#!/usr/bin/env python3
"""
News abstract text classification with a custom k-NN classifier.

Allowed libraries:
  - scikit-learn: TfidfVectorizer, TruncatedSVD, chi2, preprocessing.normalize,
    train_test_split, f1_score
  - scipy.sparse: sparse BM25 / TF-IDF matrices
  - nltk: stemming / lemmatization / stopwords (optional)
  - numpy: vectorized neighbor search on batched similarity blocks

The k-NN implementation itself is written here (no sklearn.neighbors).

Strong accuracy-oriented pipeline (see pr1/p1.ipynb): rich tokens (punct strip, stem,
stopwords, synthetic word n-grams), BM25 or sublinear TF-IDF, optional chi-squared
feature selection, optional LSI (SVD), weighted voting with --sim-power.
Multi-representation ensembles and PRF from the notebook are not ported (very heavy).
"""

from __future__ import annotations

import argparse
import math
import re
import string
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

SparseOrDense = Union[sparse.csr_matrix, np.ndarray]

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
# Notebook-style representation (BM25 / sublinear TF-IDF / chi2 / LSI)
# -----------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def preprocess_rich(raw: str, *, include_trigrams: bool = True) -> List[str]:
    """
    Lowercase, strip punctuation, remove English stopwords, Porter-stem words,
    then emit unigrams + underscore bigrams (+ optional trigrams).
    Matches the core token pipeline from p1.ipynb (without GPU-specific parts).
    """
    _ensure_nltk_data()
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    global _STEMMER
    if _STEMMER is None:
        _STEMMER = PorterStemmer()
    sw = set(stopwords.words("english"))

    text = raw.lower().translate(_PUNCT_TABLE)
    unigrams = [_STEMMER.stem(t) for t in text.split() if t not in sw and len(t) > 1]
    bigrams = [f"{unigrams[i]}_{unigrams[i + 1]}" for i in range(len(unigrams) - 1)]
    toks = unigrams + bigrams
    if include_trigrams:
        tri = [
            f"{unigrams[i]}_{unigrams[i + 1]}_{unigrams[i + 2]}"
            for i in range(len(unigrams) - 2)
        ]
        toks.extend(tri)
    return toks


def preprocess_rich_corpus(texts: Sequence[str], *, include_trigrams: bool = True) -> List[List[str]]:
    return [preprocess_rich(t, include_trigrams=include_trigrams) for t in texts]


def renormalize(mat: csr_matrix) -> csr_matrix:
    """L2-normalize each row of a sparse matrix (cosine k-NN on dot product)."""
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1.astype(np.float64)
    norms[norms == 0] = 1.0
    return diags(1.0 / norms) @ mat


def build_vocab_idf(
    token_lists: Sequence[Sequence[str]],
    *,
    min_df: int,
    max_vocab: int,
    idf_mode: Literal["bm25", "tfidf"],
) -> Tuple[dict, np.ndarray, defaultdict]:
    """Vocabulary + IDF weights from training token lists (df filtered, top terms by df)."""
    N = len(token_lists)
    doc_freq: defaultdict[str, int] = defaultdict(int)
    for toks in token_lists:
        for t in set(toks):
            doc_freq[t] += 1
    terms = sorted([(t, df) for t, df in doc_freq.items() if df >= min_df], key=lambda x: -x[1])[:max_vocab]
    vocab = {t: i for i, (t, _) in enumerate(terms)}
    idf = np.zeros(len(vocab), dtype=np.float32)
    for t, idx in vocab.items():
        df = doc_freq[t]
        if idf_mode == "bm25":
            idf[idx] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        else:
            idf[idx] = math.log((N + 1) / (df + 1)) + 1
    return vocab, idf, doc_freq


def build_bm25_matrix(
    token_lists: Sequence[Sequence[str]],
    vocab: dict,
    idf: np.ndarray,
    k1: float,
    b: float,
    avgdl: float,
) -> csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for doc_idx, tokens in enumerate(token_lists):
        if not tokens:
            continue
        tf_counts = Counter(tokens)
        doc_len = len(tokens)
        for term, tf in tf_counts.items():
            if term not in vocab:
                continue
            col = vocab[term]
            score = idf[col] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            rows.append(doc_idx)
            cols.append(col)
            vals.append(float(score))
    mat = csr_matrix((vals, (rows, cols)), shape=(len(token_lists), len(vocab)), dtype=np.float32)
    return renormalize(mat)


def build_sublinear_tfidf_matrix(token_lists: Sequence[Sequence[str]], vocab: dict, idf: np.ndarray) -> csr_matrix:
    """Notebook-style sublinear TF * IDF / doc length, then row L2-normalize."""
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for doc_idx, tokens in enumerate(token_lists):
        if not tokens:
            continue
        tf_counts = Counter(tokens)
        total = len(tokens)
        for term, count in tf_counts.items():
            if term not in vocab:
                continue
            col = vocab[term]
            val = (1 + math.log(count)) / total * idf[col]
            rows.append(doc_idx)
            cols.append(col)
            vals.append(float(val))
    mat = csr_matrix((vals, (rows, cols)), shape=(len(token_lists), len(vocab)), dtype=np.float32)
    return renormalize(mat)


def apply_chi2_selection(
    train_mat: csr_matrix,
    labels: Sequence[int],
    k: int,
    test_mat: Optional[csr_matrix] = None,
) -> Tuple[csr_matrix, Optional[csr_matrix]]:
    """Select top-k features by chi-squared vs. labels; re-normalize rows."""
    y = np.asarray(labels, dtype=np.int64)
    scores, _ = chi2(train_mat, y)
    k_eff = min(k, train_mat.shape[1])
    top = np.argsort(scores)[-k_eff:]
    top.sort()
    tr = renormalize(train_mat[:, top].tocsr())
    if test_mat is None:
        return tr, None
    return tr, renormalize(test_mat[:, top].tocsr())


def l2_normalize_dense(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def apply_lsi(
    X_train: csr_matrix,
    X_test: Optional[csr_matrix],
    *,
    n_components: int,
    random_state: int,
    n_iter: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """TruncatedSVD + row L2 normalization (dense cosine k-NN), as in p1.ipynb LSI models."""
    n_comp = min(n_components, X_train.shape[1] - 1, X_train.shape[0] - 1)
    n_comp = max(n_comp, 1)
    svd = TruncatedSVD(
        n_components=n_comp,
        random_state=random_state,
        algorithm="randomized",
        n_iter=n_iter,
    )
    Xt = l2_normalize_dense(svd.fit_transform(X_train))
    if X_test is None:
        return Xt, None
    Xv = l2_normalize_dense(svd.transform(X_test))
    return Xt, Xv


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
    k-NN for sparse TF-IDF/BM25 or dense LSI features.

    - Sparse cosine: L2-normalized rows; dot product = cosine similarity.
    - Sparse Euclidean: squared distances via norms and dot products.
    - Dense cosine (e.g. after TruncatedSVD): batched matrix multiply.

    Labels are integers in {1, 2, 3, 4}. Optional sim_power sharpens weighted votes (p1.ipynb).
    """

    def __init__(
        self,
        k: int,
        metric: Metric = "cosine",
        voting: Voting = "majority",
        batch_size: int = 2048,
        sim_power: float = 1.0,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.metric = metric
        self.voting = voting
        self.batch_size = batch_size
        self.sim_power = sim_power
        self._X_train_sparse: Optional[sparse.csr_matrix] = None
        self._X_train_dense: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._train_norm_sq: Optional[np.ndarray] = None

    def fit(self, X_train: SparseOrDense, y_train: Sequence[int]) -> "CustomKNNClassifier":
        self._y_train = np.asarray(y_train, dtype=np.int64)
        if isinstance(X_train, np.ndarray):
            if self.metric == "euclidean":
                raise ValueError("Dense Euclidean k-NN is not implemented; use cosine + LSI or sparse features.")
            self._X_train_dense = np.asarray(X_train, dtype=np.float32)
            self._X_train_sparse = None
            self._train_norm_sq = None
            return self

        self._X_train_dense = None
        self._X_train_sparse = X_train.tocsr()
        if self.metric == "cosine":
            self._X_train_sparse = normalize(self._X_train_sparse, norm="l2", axis=1)
        if self.metric == "euclidean":
            x = self._X_train_sparse.astype(np.float64, copy=False)
            self._train_norm_sq = np.asarray(x.multiply(x).sum(axis=1)).ravel()
        else:
            self._train_norm_sq = None
        return self

    def predict(self, X: SparseOrDense) -> np.ndarray:
        if self._y_train is None:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(X, np.ndarray):
            return self._predict_dense_cosine(np.asarray(X, dtype=np.float32))

        X = X.tocsr()
        if self._X_train_sparse is None:
            raise RuntimeError("Internal error: sparse predict without sparse training matrix.")

        n = X.shape[0]
        preds = np.empty(n, dtype=np.int64)
        Xt = self._X_train_sparse
        y = self._y_train
        k_eff = min(self.k, Xt.shape[0])

        if self.metric == "cosine":
            Xn = normalize(X, norm="l2", axis=1)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                block = Xn[start:end] @ Xt.T
                sims = block.toarray() if sparse.issparse(block) else np.asarray(block)
                self._predict_block_from_similarity(sims, y, k_eff, preds, start)
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
                self._predict_block_from_distance(dist2, y, k_eff, preds, start)

        return preds

    def _predict_dense_cosine(self, X: np.ndarray) -> np.ndarray:
        """Cosine k-NN on row-L2-normalized dense matrices."""
        if self._X_train_dense is None:
            raise RuntimeError("Internal error: dense predict without dense training matrix.")
        tr = self._X_train_dense
        y = self._y_train
        assert y is not None
        n = X.shape[0]
        preds = np.empty(n, dtype=np.int64)
        k_eff = min(self.k, tr.shape[0])
        Xn = l2_normalize_dense(X)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            sims = Xn[start:end] @ tr.T
            self._predict_block_from_similarity(np.asarray(sims, dtype=np.float64), y, k_eff, preds, start)
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
                w = row[idx].astype(np.float64)
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
                preds[offset + i] = self._weighted_vote(neigh_y, w.astype(np.float64))

    def _majority_vote(self, neigh_y: np.ndarray) -> int:
        counts = np.bincount(neigh_y, minlength=5)
        c = counts[1:5]
        return int(np.argmax(c) + 1)

    def _weighted_vote(self, neigh_y: np.ndarray, weights: np.ndarray) -> int:
        if self.sim_power != 1.0:
            weights = np.power(np.maximum(weights, 0.0), self.sim_power)
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
    X: SparseOrDense,
    y: List[int],
    *,
    val_size: float,
    random_state: int,
    k: int,
    metric: Metric,
    voting: Voting,
    batch_size: int,
    sim_power: float,
) -> float:
    y_arr = np.asarray(y, dtype=np.int64)
    idx = np.arange(X.shape[0])
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=val_size,
        random_state=random_state,
        stratify=y_arr,
    )
    X_fit = X[tr_idx]
    X_va = X[va_idx]
    y_tr = y_arr[tr_idx]
    y_va = y_arr[va_idx]

    knn = CustomKNNClassifier(
        k=k,
        metric=metric,
        voting=voting,
        batch_size=batch_size,
        sim_power=sim_power,
    )
    knn.fit(X_fit, y_tr)
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

    p.add_argument(
        "--pipeline",
        choices=("sklearn", "bm25", "sublinear"),
        default="bm25",
        help="sklearn=TfidfVectorizer on simple preprocess; bm25/sublinear = p1.ipynb-style rich tokens + custom weighting.",
    )
    p.add_argument("--no-trigrams", action="store_true", help="Rich pipeline: uni+bigrams only (drop synthetic trigrams).")
    p.add_argument("--max-vocab", type=int, default=150_000, help="Max vocabulary size after min_df filtering (rich pipelines).")
    p.add_argument(
        "--chi2-k",
        type=int,
        default=120_000,
        help="Retain top-K features by chi-squared vs. labels (0 = disable). Inspired by p1.ipynb.",
    )
    p.add_argument("--bm25-k1", type=float, default=1.2, help="BM25 k1 (notebook grid favored ~1.2).")
    p.add_argument("--bm25-b", type=float, default=0.5, help="BM25 b (notebook grid favored ~0.5).")
    p.add_argument(
        "--lsi-components",
        type=int,
        default=0,
        help="If >0, apply TruncatedSVD then dense cosine k-NN (requires --metric cosine).",
    )
    p.add_argument("--svd-iter", type=int, default=7, help="Iterations for randomized TruncatedSVD.")

    p.add_argument("--k", type=int, default=15)
    p.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine")
    p.add_argument("--voting", choices=("majority", "weighted"), default="majority")
    p.add_argument(
        "--sim-power",
        type=float,
        default=1.0,
        help="Raise neighbor similarity weights to this power (weighted voting only; try 2.0 per p1.ipynb).",
    )
    p.add_argument("--batch-size", type=int, default=2048)

    p.add_argument("--use-stemmer", action="store_true")
    p.add_argument("--use-lemma", action="store_true")
    p.add_argument("--no-stopwords", action="store_true", help="Sklearn pipeline only: do not remove English stopwords.")

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

    if args.pipeline != "sklearn" and args.lsi_components > 0 and args.metric != "cosine":
        print("LSI requires cosine similarity; use --metric cosine.", file=sys.stderr)
        return 2

    texts_tr, y_tr_list = load_train(train_path)
    if args.max_train_samples is not None:
        texts_tr = texts_tr[: args.max_train_samples]
        y_tr_list = y_tr_list[: args.max_train_samples]

    y_tr = np.asarray(y_tr_list, dtype=np.int64)
    X_te: Optional[Union[csr_matrix, np.ndarray]] = None

    if args.pipeline == "sklearn":
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
        if not args.val_only:
            texts_te = load_test(test_path)
            texts_te_pp = preprocess_texts(
                texts_te,
                use_stemmer=args.use_stemmer,
                use_lemma=args.use_lemma,
                remove_stopwords=remove_stop,
            )
            X_te = vec.transform(texts_te_pp).tocsr()
    else:
        include_tri = not args.no_trigrams
        tokens_tr = preprocess_rich_corpus(texts_tr, include_trigrams=include_tri)
        idf_mode: Literal["bm25", "tfidf"] = "bm25" if args.pipeline == "bm25" else "tfidf"
        vocab, idf, _ = build_vocab_idf(
            tokens_tr,
            min_df=args.min_df,
            max_vocab=args.max_vocab,
            idf_mode=idf_mode,
        )
        avgdl = float(np.mean([len(t) for t in tokens_tr])) if tokens_tr else 1.0

        if args.pipeline == "bm25":
            X_tr = build_bm25_matrix(tokens_tr, vocab, idf, args.bm25_k1, args.bm25_b, avgdl)
        else:
            X_tr = build_sublinear_tfidf_matrix(tokens_tr, vocab, idf)

        if not args.val_only:
            texts_te = load_test(test_path)
            tokens_te = preprocess_rich_corpus(texts_te, include_trigrams=include_tri)
            if args.pipeline == "bm25":
                X_te = build_bm25_matrix(tokens_te, vocab, idf, args.bm25_k1, args.bm25_b, avgdl)
            else:
                X_te = build_sublinear_tfidf_matrix(tokens_te, vocab, idf)

        if args.chi2_k > 0:
            X_tr, X_te = apply_chi2_selection(X_tr, y_tr_list, args.chi2_k, X_te)

        if args.lsi_components > 0:
            X_tr, X_te = apply_lsi(
                X_tr,
                X_te,
                n_components=args.lsi_components,
                random_state=args.random_state,
                n_iter=args.svd_iter,
            )

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
            sim_power=args.sim_power,
        )
        print(f"Validation macro-F1 (stratified holdout, val_size={args.val_size}): {f1:.4f}")

    if args.val_only:
        return 0

    assert X_te is not None

    knn = CustomKNNClassifier(
        k=args.k,
        metric=args.metric,
        voting=args.voting,
        batch_size=args.batch_size,
        sim_power=args.sim_power,
    )
    knn.fit(X_tr, y_tr)
    pred_te = knn.predict(X_te)
    write_predictions(out_path, pred_te)
    print(f"Wrote {len(pred_te)} lines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
