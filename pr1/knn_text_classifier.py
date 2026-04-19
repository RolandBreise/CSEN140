#!/usr/bin/env python3
"""
News abstract text classification with a custom k-NN classifier.

Allowed libraries:
  - scikit-learn: TfidfVectorizer, preprocessing.normalize, train_test_split, f1_score
  - scipy.sparse: sparse TF-IDF matrices
  - nltk: stemming / lemmatization / stopwords (optional)
  - numpy: vectorized neighbor search on batched similarity blocks

The k-NN implementation itself is written here (no sklearn.neighbors).
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# -----------------------------------------------------------------------------
# NLTK (optional pieces loaded lazily)
# -----------------------------------------------------------------------------

_STEMMER = None
_LEMMATIZER = None
_STOPWORDS: Optional[set] = None


def _ensure_nltk_data() -> None:
    """Download small NLTK resources if missing (safe to call multiple times)."""
    import nltk

    corpora = {"stopwords", "wordnet", "omw-1.4"}
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg in corpora else f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def _tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def preprocess_texts(
    texts: Sequence[str],
    *,
    use_stemmer: bool,
    use_lemma: bool,
    remove_stopwords: bool,
) -> List[str]:
    """
    Return a list of space-joined token strings (one per document).
    Stemming and lemmatization are mutually exclusive in this pipeline;
    if both flags are set, lemmatization wins.
    """
    global _STEMMER, _LEMMATIZER, _STOPWORDS

    if use_stemmer or use_lemma or remove_stopwords:
        _ensure_nltk_data()

    if remove_stopwords and _STOPWORDS is None:
        from nltk.corpus import stopwords

        _STOPWORDS = set(stopwords.words("english"))

    if use_stemmer and not use_lemma and _STEMMER is None:
        from nltk.stem import PorterStemmer

        _STEMMER = PorterStemmer()

    if use_lemma and _LEMMATIZER is None:
        from nltk.stem import WordNetLemmatizer

        _LEMMATIZER = WordNetLemmatizer()

    out: List[str] = []
    for raw in texts:
        toks = _tokenize_simple(raw)
        if remove_stopwords and _STOPWORDS is not None:
            toks = [t for t in toks if t not in _STOPWORDS]
        if use_lemma and _LEMMATIZER is not None:
            toks = [_LEMMATIZER.lemmatize(t) for t in toks]
        elif use_stemmer and _STEMMER is not None:
            toks = [_STEMMER.stem(t) for t in toks]
        out.append(" ".join(toks))
    return out


# -----------------------------------------------------------------------------
# Data I/O
# -----------------------------------------------------------------------------


def load_train(path: Path) -> Tuple[List[str], List[int]]:
    labels: List[int] = []
    texts: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            m = re.match(r"^([1-4])\s+(.*)$", line)
            if not m:
                raise ValueError(f"Bad train line (expected 'd text'): {line[:80]!r}")
            labels.append(int(m.group(1)))
            texts.append(m.group(2))
    return texts, labels


def load_test(path: Path) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            texts.append(line)
    return texts


def write_predictions(path: Path, preds: Sequence[int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for y in preds:
            f.write(f"{int(y)}\n")


# -----------------------------------------------------------------------------
# Custom k-NN (batched; no sklearn.neighbors)
# -----------------------------------------------------------------------------

Metric = Literal["cosine", "euclidean"]
Voting = Literal["majority", "weighted"]


class CustomKNNClassifier:
    """
    k-NN for high-dimensional sparse features.

    - Cosine: rows are L2-normalized; similarity is sparse-dense dot product per batch.
    - Euclidean: squared distances via ||x||^2 + ||y||^2 - 2 x·y (vectorized per batch).

    Labels are assumed to be integers in {1, 2, 3, 4}.
    """

    def __init__(
        self,
        k: int,
        metric: Metric = "cosine",
        voting: Voting = "majority",
        batch_size: int = 2048,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.metric = metric
        self.voting = voting
        self.batch_size = batch_size
        self._X_train: Optional[sparse.csr_matrix] = None
        self._y_train: Optional[np.ndarray] = None
        self._train_norm_sq: Optional[np.ndarray] = None

    def fit(self, X_train: sparse.csr_matrix, y_train: Sequence[int]) -> "CustomKNNClassifier":
        self._X_train = X_train.tocsr()
        self._y_train = np.asarray(y_train, dtype=np.int64)
        if self.metric == "cosine":
            self._X_train = normalize(self._X_train, norm="l2", axis=1)
        # Precompute squared row norms for Euclidean distance
        if self.metric == "euclidean":
            x = self._X_train.astype(np.float64, copy=False)
            self._train_norm_sq = np.asarray(x.multiply(x).sum(axis=1)).ravel()
        else:
            self._train_norm_sq = None
        return self

    def predict(self, X: sparse.csr_matrix) -> np.ndarray:
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Call fit() before predict().")

        X = X.tocsr()
        n = X.shape[0]
        preds = np.empty(n, dtype=np.int64)
        Xt = self._X_train
        y = self._y_train
        k = min(self.k, Xt.shape[0])

        if self.metric == "cosine":
            Xn = normalize(X, norm="l2", axis=1)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                block = Xn[start:end] @ Xt.T
                sims = block.toarray() if sparse.issparse(block) else np.asarray(block)
                self._predict_block_from_similarity(sims, y, k, preds, start)
        else:
            x64 = X.astype(np.float64, copy=False)
            chunk_norm_sq = np.asarray(x64.multiply(x64).sum(axis=1)).ravel()
            train_norm_sq = self._train_norm_sq
            assert train_norm_sq is not None
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                chunk = x64[start:end]
                block = chunk @ Xt.T
                dots = block.toarray() if sparse.issparse(block) else np.asarray(block)
                cns = chunk_norm_sq[start:end][:, None]
                dist2 = np.maximum(cns + train_norm_sq[None, :] - 2.0 * dots, 0.0)
                # k smallest distances
                self._predict_block_from_distance(dist2, y, k, preds, start)

        return preds

    def _predict_block_from_similarity(
        self,
        sims: np.ndarray,
        y: np.ndarray,
        k: int,
        preds: np.ndarray,
        offset: int,
    ) -> None:
        """sims: (batch, n_train) cosine similarities (higher is nearer)."""
        for i in range(sims.shape[0]):
            row = sims[i]
            idx = np.argpartition(-row, k - 1)[:k]
            neigh_y = y[idx]
            if self.voting == "majority":
                preds[offset + i] = self._majority_vote(neigh_y)
            else:
                w = row[idx]
                preds[offset + i] = self._weighted_vote(neigh_y, w)

    def _predict_block_from_distance(
        self,
        dist2: np.ndarray,
        y: np.ndarray,
        k: int,
        preds: np.ndarray,
        offset: int,
    ) -> None:
        for i in range(dist2.shape[0]):
            row = dist2[i]
            idx = np.argpartition(row, k - 1)[:k]
            neigh_y = y[idx]
            if self.voting == "majority":
                preds[offset + i] = self._majority_vote(neigh_y)
            else:
                d = np.sqrt(np.maximum(row[idx], 0.0))
                w = 1.0 / (d + 1e-9)
                preds[offset + i] = self._weighted_vote(neigh_y, w)

    @staticmethod
    def _majority_vote(neigh_y: np.ndarray) -> int:
        # Labels 1..4
        counts = np.bincount(neigh_y, minlength=5)
        c = counts[1:5]
        # tie-break toward smaller class id for reproducibility
        return int(np.argmax(c) + 1)

    @staticmethod
    def _weighted_vote(neigh_y: np.ndarray, weights: np.ndarray) -> int:
        score = np.zeros(5, dtype=np.float64)
        for cls, w in zip(neigh_y, weights):
            score[int(cls)] += float(w)
        c = score[1:5]
        return int(np.argmax(c) + 1)


# -----------------------------------------------------------------------------
# Pipeline + validation
# -----------------------------------------------------------------------------


def _whitespace_analyzer(doc: str) -> List[str]:
    """Split preprocessed space-joined tokens (avoid double tokenization)."""
    if not doc:
        return []
    return doc.split()


def build_vectorizer(
    max_features: Optional[int],
    ngram_range: Tuple[int, int],
    min_df: int,
    sublinear_tf: bool,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        analyzer=_whitespace_analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
    )


def run_validation(
    X: sparse.csr_matrix,
    y: List[int],
    *,
    val_size: float,
    random_state: int,
    k: int,
    metric: Metric,
    voting: Voting,
    batch_size: int,
) -> float:
    y_arr = np.asarray(y, dtype=np.int64)
    idx = np.arange(X.shape[0])
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=val_size,
        random_state=random_state,
        stratify=y_arr,
    )
    X_tr = X[tr_idx]
    X_va = X[va_idx]
    y_tr = y_arr[tr_idx]
    y_va = y_arr[va_idx]

    knn = CustomKNNClassifier(k=k, metric=metric, voting=voting, batch_size=batch_size)
    knn.fit(X_tr, y_tr)
    pred = knn.predict(X_va)
    return float(f1_score(y_va, pred, average="macro"))


def print_efficiency_notes() -> None:
    notes = """
Efficiency notes (for large N_train × N_test k-NN on sparse TF-IDF)
-------------------------------------------------------------------
1) Never build a full (N_test × N_train) similarity matrix at once unless you
   have the RAM. This script multiplies in batches: (batch × D) @ (D × N_train).

2) Cosine k-NN on L2-normalized sparse rows reduces to a single large dot
   product per batch (fast BLAS/SciPy sparse-dense matmul).

3) Euclidean k-NN still vectorizes as:
     ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
   with ||y||^2 precomputed once for all training rows.

4) Further speedups (not implemented here): approximate nearest neighbors
   (Annoy, FAISS, nmslib), inverted-index retrieval for sparse vectors, or
   prototype / condensed NN training sets for prototyping.

5) TF-IDF controls: lowering max_features, using char n-grams only for very
   high speed experiments, or sublinear_tf=True can change accuracy/runtime.
"""
    print(notes.strip())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Custom k-NN text classifier for CSEN140-style news data.")
    here = Path(__file__).resolve().parent
    p.add_argument("--train", type=Path, default=here / "train.dat")
    p.add_argument("--test", type=Path, default=here / "test.dat")
    p.add_argument("--output", type=Path, default=here / "predictions.dat")

    p.add_argument("--k", type=int, default=15)
    p.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine")
    p.add_argument("--voting", choices=("majority", "weighted"), default="majority")
    p.add_argument("--batch-size", type=int, default=2048)

    p.add_argument("--use-stemmer", action="store_true")
    p.add_argument("--use-lemma", action="store_true")
    p.add_argument("--no-stopwords", action="store_true", help="Do not remove English stopwords.")

    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--sublinear-tf", action="store_true")

    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--val-only", action="store_true", help="Only run validation; do not write test predictions.")
    p.add_argument("--skip-val", action="store_true")
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument("--max-train-samples", type=int, default=None, help="Subsample training for quick experiments.")
    p.add_argument("--efficiency-notes", action="store_true", help="Print efficiency suggestions and exit.")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.efficiency_notes:
        print_efficiency_notes()
        return 0

    train_path: Path = args.train
    test_path: Path = args.test
    out_path: Path = args.output

    if not train_path.is_file():
        print(f"Missing train file: {train_path}", file=sys.stderr)
        return 2
    if not args.val_only and not test_path.is_file():
        print(f"Missing test file: {test_path}", file=sys.stderr)
        return 2

    texts_tr, y_tr_list = load_train(train_path)
    if args.max_train_samples is not None:
        texts_tr = texts_tr[: args.max_train_samples]
        y_tr_list = y_tr_list[: args.max_train_samples]

    remove_stop = not args.no_stopwords
    texts_tr_pp = preprocess_texts(
        texts_tr,
        use_stemmer=args.use_stemmer,
        use_lemma=args.use_lemma,
        remove_stopwords=remove_stop,
    )

    vec = build_vectorizer(
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        sublinear_tf=args.sublinear_tf,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        X_tr = vec.fit_transform(texts_tr_pp).tocsr()

    y_tr = np.asarray(y_tr_list, dtype=np.int64)

    if not args.skip_val:
        f1 = run_validation(
            X_tr,
            y_tr_list,
            val_size=args.val_size,
            random_state=args.random_state,
            k=args.k,
            metric=args.metric,
            voting=args.voting,
            batch_size=args.batch_size,
        )
        print(f"Validation macro-F1 (stratified holdout, val_size={args.val_size}): {f1:.4f}")

    if args.val_only:
        return 0

    texts_te = load_test(test_path)
    texts_te_pp = preprocess_texts(
        texts_te,
        use_stemmer=args.use_stemmer,
        use_lemma=args.use_lemma,
        remove_stopwords=remove_stop,
    )
    X_te = vec.transform(texts_te_pp).tocsr()

    knn = CustomKNNClassifier(
        k=args.k,
        metric=args.metric,
        voting=args.voting,
        batch_size=args.batch_size,
    )
    knn.fit(X_tr, y_tr)
    pred_te = knn.predict(X_te)
    write_predictions(out_path, pred_te)
    print(f"Wrote {len(pred_te)} lines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
