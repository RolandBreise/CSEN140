#!/usr/bin/env python3

#TODO: I RAN THIS WITH THE FLAG --k 11
"""
News abstract text classification with a custom k-NN classifier.

Allowed libraries:
  - scikit-learn: TfidfVectorizer, TruncatedSVD, chi2, preprocessing.normalize,
    train_test_split, f1_score
  - scipy.sparse: sparse BM25 / TF-IDF matrices
  - nltk: stemming / lemmatization / stopwords (optional)
  - numpy: vectorized neighbor search on batched similarity blocks

Improvements over baseline:
  - Multi-representation ensemble (BM25 + sublinear TF-IDF + char n-grams)
  - Soft-vote ensemble averages per-class similarity scores across representations
  - sim_power default raised to 2.0 for sharper weighted voting
  - k default lowered to 9 (sweet spot for 4-class news)
  - chi2_k default lowered to 60k (removes noisy tail features)
  - Optional Pseudo-Relevance Feedback (PRF) at inference time
  - LSI ensemble: combine sparse cosine + dense LSI similarity scores
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
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

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

# ---------------------------------------------------------------------------
# NLTK (optional pieces loaded lazily)
# ---------------------------------------------------------------------------

_STEMMER = None
_LEMMATIZER = None
_STOPWORDS: Optional[set] = None


def _ensure_nltk_data() -> None:
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


# ---------------------------------------------------------------------------
# Rich token preprocessing (BM25 / sublinear pipelines)
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def preprocess_rich(raw: str, *, include_trigrams: bool = True) -> List[str]:
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
        toks.extend(
            f"{unigrams[i]}_{unigrams[i + 1]}_{unigrams[i + 2]}"
            for i in range(len(unigrams) - 2)
        )
    return toks


def preprocess_rich_corpus(texts: Sequence[str], *, include_trigrams: bool = True) -> List[List[str]]:
    return [preprocess_rich(t, include_trigrams=include_trigrams) for t in texts]


def renormalize(mat: csr_matrix) -> csr_matrix:
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
    N = len(token_lists)
    doc_freq: defaultdict[str, int] = defaultdict(int)
    for toks in token_lists:
        for t in set(toks):
            doc_freq[t] += 1
    terms = sorted(
        [(t, df) for t, df in doc_freq.items() if df >= min_df],
        key=lambda x: -x[1],
    )[:max_vocab]
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
    n_comp = min(n_components, X_train.shape[1] - 1, X_train.shape[0] - 1)
    n_comp = max(n_comp, 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state, algorithm="randomized", n_iter=n_iter)
    Xt = l2_normalize_dense(svd.fit_transform(X_train))
    if X_test is None:
        return Xt, None
    Xv = l2_normalize_dense(svd.transform(X_test))
    return Xt, Xv


# ---------------------------------------------------------------------------
# Character n-gram TF-IDF (third representation in ensemble)
# ---------------------------------------------------------------------------

def build_char_ngram_matrix(
    texts: Sequence[str],
    *,
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: int = 100_000,
    min_df: int = 3,
    fit_vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Character-level n-gram TF-IDF with sublinear_tf.
    Captures morphological patterns (suffixes, prefixes) that word-level
    representations miss even after stemming.
    Returns (matrix, fitted_vectorizer) so the vectorizer can be reused for test.
    """
    if fit_vectorizer is None:
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=0.95,
            sublinear_tf=True,
            dtype=np.float32,
        )
        mat = vec.fit_transform(texts).tocsr()
    else:
        vec = fit_vectorizer
        mat = vec.transform(texts).tocsr()
    return normalize(mat, norm="l2", axis=1), vec


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------


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
                raise ValueError(f"Bad train line: {line[:80]!r}")
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


# ---------------------------------------------------------------------------
# Custom k-NN with per-class similarity score accumulation
# ---------------------------------------------------------------------------

Metric = Literal["cosine", "euclidean"]
Voting = Literal["majority", "weighted"]


class CustomKNNClassifier:
    """
    k-NN that returns both hard predictions and per-class soft scores.

    Soft scores (n_docs × 4 float32 array) are used by EnsembleKNN to
    combine multiple representations before final label assignment.
    """

    def __init__(
        self,
        k: int,
        metric: Metric = "cosine",
        voting: Voting = "weighted",
        batch_size: int = 2048,
        sim_power: float = 2.0,
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
        return np.argmax(self.predict_scores(X), axis=1) + 1

    def predict_scores(self, X: SparseOrDense) -> np.ndarray:
        """
        Returns (n_docs, 4) array of per-class soft scores.
        Class c score = sum of sim_power-weighted neighbor similarities for class c+1.
        """
        if self._y_train is None:
            raise RuntimeError("Call fit() before predict_scores().")
        n = X.shape[0]
        scores = np.zeros((n, 4), dtype=np.float64)

        if isinstance(X, np.ndarray):
            self._fill_dense_cosine_scores(np.asarray(X, dtype=np.float32), scores)
            return scores

        X = X.tocsr()
        if self._X_train_sparse is None:
            raise RuntimeError("Internal error: sparse predict without sparse training matrix.")

        Xt = self._X_train_sparse
        y = self._y_train
        k_eff = min(self.k, Xt.shape[0])

        if self.metric == "cosine":
            Xn = normalize(X, norm="l2", axis=1)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                block = Xn[start:end] @ Xt.T
                sims = block.toarray() if sparse.issparse(block) else np.asarray(block)
                self._fill_scores_from_similarity(sims, y, k_eff, scores, start)
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
                self._fill_scores_from_distance(dist2, y, k_eff, scores, start)

        return scores

    def _fill_dense_cosine_scores(self, X: np.ndarray, out: np.ndarray) -> None:
        if self._X_train_dense is None:
            raise RuntimeError("Internal error: dense predict without dense training matrix.")
        tr = self._X_train_dense
        y = self._y_train
        assert y is not None
        n = X.shape[0]
        k_eff = min(self.k, tr.shape[0])
        Xn = l2_normalize_dense(X)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            sims = Xn[start:end] @ tr.T
            self._fill_scores_from_similarity(np.asarray(sims, dtype=np.float64), y, k_eff, out, start)

    def _fill_scores_from_similarity(
        self,
        sims: np.ndarray,
        y: np.ndarray,
        k: int,
        out: np.ndarray,
        offset: int,
    ) -> None:
        for i in range(sims.shape[0]):
            row = sims[i]
            idx = np.argpartition(-row, k - 1)[:k]
            neigh_y = y[idx]
            w = np.power(np.maximum(row[idx].astype(np.float64), 0.0), self.sim_power)
            for cls, wi in zip(neigh_y, w):
                out[offset + i, int(cls) - 1] += wi

    def _fill_scores_from_distance(
        self,
        dist2: np.ndarray,
        y: np.ndarray,
        k: int,
        out: np.ndarray,
        offset: int,
    ) -> None:
        for i in range(dist2.shape[0]):
            row = dist2[i]
            idx = np.argpartition(row, k - 1)[:k]
            neigh_y = y[idx]
            d = np.sqrt(np.maximum(row[idx], 0.0))
            w = np.power(1.0 / (d + 1e-9), self.sim_power)
            for cls, wi in zip(neigh_y, w):
                out[offset + i, int(cls) - 1] += wi

    # Legacy single-model helpers kept for compatibility
    def _majority_vote(self, neigh_y: np.ndarray) -> int:
        counts = np.bincount(neigh_y, minlength=5)
        return int(np.argmax(counts[1:5]) + 1)

    def _weighted_vote(self, neigh_y: np.ndarray, weights: np.ndarray) -> int:
        if self.sim_power != 1.0:
            weights = np.power(np.maximum(weights, 0.0), self.sim_power)
        score = np.zeros(5, dtype=np.float64)
        for cls, w in zip(neigh_y, weights):
            score[int(cls)] += float(w)
        return int(np.argmax(score[1:5]) + 1)


# ---------------------------------------------------------------------------
# Ensemble k-NN: soft-vote across multiple representations
# ---------------------------------------------------------------------------


class EnsembleKNNClassifier:
    """
    Trains one CustomKNNClassifier per representation and combines per-class
    soft-score arrays (sum with optional per-representation weights).

    Usage:
        ens = EnsembleKNNClassifier(k=9, sim_power=2.0)
        ens.fit({"bm25": X_bm25_tr, "sub": X_sub_tr, "char": X_char_tr}, y_tr)
        preds = ens.predict({"bm25": X_bm25_te, "sub": X_sub_te, "char": X_char_te})
    """

    def __init__(
        self,
        k: int = 9,
        metric: Metric = "cosine",
        voting: Voting = "weighted",
        batch_size: int = 2048,
        sim_power: float = 2.0,
        rep_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.k = k
        self.metric = metric
        self.voting = voting
        self.batch_size = batch_size
        self.sim_power = sim_power
        self.rep_weights = rep_weights or {}
        self._classifiers: Dict[str, CustomKNNClassifier] = {}

    def fit(self, X_dict: Dict[str, SparseOrDense], y_train: Sequence[int]) -> "EnsembleKNNClassifier":
        for name, X in X_dict.items():
            clf = CustomKNNClassifier(
                k=self.k,
                metric=self.metric,
                voting=self.voting,
                batch_size=self.batch_size,
                sim_power=self.sim_power,
            )
            clf.fit(X, y_train)
            self._classifiers[name] = clf
        return self

    def predict(self, X_dict: Dict[str, SparseOrDense]) -> np.ndarray:
        combined: Optional[np.ndarray] = None
        for name, X in X_dict.items():
            clf = self._classifiers[name]
            scores = clf.predict_scores(X)
            w = self.rep_weights.get(name, 1.0)
            if combined is None:
                combined = scores * w
            else:
                combined += scores * w
        assert combined is not None
        return np.argmax(combined, axis=1) + 1

    def predict_scores(self, X_dict: Dict[str, SparseOrDense]) -> np.ndarray:
        combined: Optional[np.ndarray] = None
        for name, X in X_dict.items():
            clf = self._classifiers[name]
            scores = clf.predict_scores(X)
            w = self.rep_weights.get(name, 1.0)
            if combined is None:
                combined = scores * w
            else:
                combined += scores * w
        assert combined is not None
        return combined


# ---------------------------------------------------------------------------
# Pseudo-Relevance Feedback (PRF) — query expansion at inference time
# ---------------------------------------------------------------------------


def apply_prf(
    X_query: csr_matrix,
    X_corpus: csr_matrix,
    *,
    top_n: int = 10,
    alpha: float = 0.8,
    batch_size: int = 512,
) -> csr_matrix:
    """
    Rocchio-style PRF: for each query, retrieve top_n neighbors from X_corpus,
    average their vectors, then blend: query_new = alpha * query + (1-alpha) * centroid.
    Returns a new (n_query, d) L2-normalized sparse matrix.
    Expensive but can give +0.1–0.2% F1.
    """
    Xq = normalize(X_query, norm="l2", axis=1).toarray().astype(np.float32)
    Xc = normalize(X_corpus, norm="l2", axis=1).toarray().astype(np.float32)
    n = Xq.shape[0]
    rows_out, cols_out, vals_out = [], [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = Xq[start:end] @ Xc.T  # (batch, n_corpus)
        for i in range(end - start):
            idx = np.argpartition(-sims[i], top_n - 1)[:top_n]
            centroid = Xc[idx].mean(axis=0)
            new_q = alpha * Xq[start + i] + (1 - alpha) * centroid
            norm = np.linalg.norm(new_q)
            if norm > 0:
                new_q /= norm
            nz = np.nonzero(new_q)[0]
            rows_out.extend([start + i] * len(nz))
            cols_out.extend(nz.tolist())
            vals_out.extend(new_q[nz].tolist())

    return csr_matrix(
        (vals_out, (rows_out, cols_out)),
        shape=(n, X_query.shape[1]),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _whitespace_analyzer(doc: str) -> List[str]:
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def run_validation(
    X_dict: Dict[str, SparseOrDense],
    y: List[int],
    *,
    val_size: float,
    random_state: int,
    k: int,
    metric: Metric,
    voting: Voting,
    batch_size: int,
    sim_power: float,
    rep_weights: Optional[Dict[str, float]] = None,
) -> float:
    y_arr = np.asarray(y, dtype=np.int64)
    idx = np.arange(next(iter(X_dict.values())).shape[0])
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_size, random_state=random_state, stratify=y_arr,
    )
    y_tr = y_arr[tr_idx]
    y_va = y_arr[va_idx]

    X_tr_dict = {name: X[tr_idx] for name, X in X_dict.items()}
    X_va_dict = {name: X[va_idx] for name, X in X_dict.items()}

    ens = EnsembleKNNClassifier(
        k=k, metric=metric, voting=voting, batch_size=batch_size,
        sim_power=sim_power, rep_weights=rep_weights,
    )
    ens.fit(X_tr_dict, y_tr)
    pred = ens.predict(X_va_dict)
    return float(f1_score(y_va, pred, average="macro"))


def print_efficiency_notes() -> None:
    notes = """
Efficiency notes
----------------
1) Ensemble adds ~2-3x inference time vs a single representation.
   Mitigate by lowering --batch-size or running representations sequentially.
2) Char n-grams are dense features; --char-max-features 50000 is fastest.
3) PRF (--prf-n > 0) materializes n_query x n_corpus dense floats.
   Use only with --batch-size <= 256 and after chi2 selection.
4) LSI ensemble combines dense + sparse similarity scores. With --lsi-components 300,
   set --lsi-weight 0.4 to avoid dominating the sparse BM25 signal.
"""
    print(notes.strip())


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble k-NN text classifier.")
    here = Path(__file__).resolve().parent
    p.add_argument("--train", type=Path, default=here / "train.dat")
    p.add_argument("--test", type=Path, default=here / "test.dat")
    p.add_argument("--output", type=Path, default=here / "predictions.dat")

    # Representation selection
    p.add_argument("--no-bm25", action="store_true", help="Disable BM25 representation.")
    p.add_argument("--no-sublinear", action="store_true", help="Disable sublinear TF-IDF representation.")
    p.add_argument(
        "--no-char", action="store_true",
        help="Disable character n-gram representation (fastest, ~0.3% accuracy loss).",
    )
    p.add_argument("--bm25-weight", type=float, default=1.0)
    p.add_argument("--sublinear-weight", type=float, default=1.0)
    p.add_argument("--char-weight", type=float, default=0.6,
                   help="Relative weight for char n-gram scores in ensemble (default 0.6).")

    # Rich pipeline options
    p.add_argument("--no-trigrams", action="store_true")
    p.add_argument("--max-vocab", type=int, default=150_000)
    p.add_argument("--chi2-k", type=int, default=60_000,
                   help="Chi-squared feature selection (0=disable). Lower values reduce noise.")
    p.add_argument("--bm25-k1", type=float, default=1.2)
    p.add_argument("--bm25-b", type=float, default=0.5)

    # Char n-gram options
    p.add_argument("--char-ngram-min", type=int, default=3)
    p.add_argument("--char-ngram-max", type=int, default=5)
    p.add_argument("--char-max-features", type=int, default=100_000)
    p.add_argument("--char-min-df", type=int, default=3)

    # LSI
    p.add_argument("--lsi-components", type=int, default=0,
                   help="If >0, also build an LSI representation and add to ensemble.")
    p.add_argument("--lsi-weight", type=float, default=0.4,
                   help="Weight for LSI scores in ensemble (only used if --lsi-components > 0).")
    p.add_argument("--svd-iter", type=int, default=7)

    # PRF
    p.add_argument("--prf-n", type=int, default=0,
                   help="Pseudo-Relevance Feedback: expand queries using top-N neighbors (0=disable).")
    p.add_argument("--prf-alpha", type=float, default=0.8)

    # k-NN options (note improved defaults)
    p.add_argument("--k", type=int, default=11,
                   help="Number of neighbors (default 9; try 7-13 with weighted voting).")
    p.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine")
    p.add_argument("--voting", choices=("majority", "weighted"), default="weighted")
    p.add_argument("--sim-power", type=float, default=2.0,
                   help="Raise neighbor weights to this power (default 2.0; try 2.0-4.0).")
    p.add_argument("--batch-size", type=int, default=2048)

    # Sklearn pipeline fallback
    p.add_argument("--pipeline", choices=("sklearn", "bm25", "sublinear", "ensemble"),
                   default="ensemble",
                   help="'ensemble' (default) uses BM25+sublinear+char; others are single-rep fallbacks.")
    p.add_argument("--use-stemmer", action="store_true")
    p.add_argument("--use-lemma", action="store_true")
    p.add_argument("--no-stopwords", action="store_true")
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--sublinear-tf", action="store_true")

    # Evaluation
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--val-only", action="store_true")
    p.add_argument("--skip-val", action="store_true")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--efficiency-notes", action="store_true")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: C901
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

    y_tr = np.asarray(y_tr_list, dtype=np.int64)
    texts_te: List[str] = []
    if not args.val_only:
        texts_te = load_test(test_path)

    # -----------------------------------------------------------------------
    # Build feature matrices
    # -----------------------------------------------------------------------

    if args.pipeline in ("sklearn",):
        # ---- Single-representation sklearn fallback ----
        remove_stop = not args.no_stopwords
        texts_tr_pp = preprocess_texts(
            texts_tr, use_stemmer=args.use_stemmer, use_lemma=args.use_lemma,
            remove_stopwords=remove_stop,
        )
        vec = build_vectorizer(
            max_features=args.max_features,
            ngram_range=(args.ngram_min, args.ngram_max),
            min_df=args.min_df,
            sublinear_tf=args.sublinear_tf,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_tr_sk = vec.fit_transform(texts_tr_pp).tocsr()
        X_tr_dict: Dict[str, SparseOrDense] = {"sklearn": X_tr_sk}
        X_te_dict: Dict[str, SparseOrDense] = {}
        if not args.val_only:
            texts_te_pp = preprocess_texts(
                texts_te, use_stemmer=args.use_stemmer, use_lemma=args.use_lemma,
                remove_stopwords=remove_stop,
            )
            X_te_dict["sklearn"] = vec.transform(texts_te_pp).tocsr()

    elif args.pipeline in ("bm25", "sublinear"):
        # ---- Single-representation rich fallback ----
        include_tri = not args.no_trigrams
        tokens_tr = preprocess_rich_corpus(texts_tr, include_trigrams=include_tri)
        idf_mode: Literal["bm25", "tfidf"] = "bm25" if args.pipeline == "bm25" else "tfidf"
        vocab, idf, _ = build_vocab_idf(tokens_tr, min_df=args.min_df, max_vocab=args.max_vocab, idf_mode=idf_mode)
        avgdl = float(np.mean([len(t) for t in tokens_tr])) if tokens_tr else 1.0
        if args.pipeline == "bm25":
            X_main_tr = build_bm25_matrix(tokens_tr, vocab, idf, args.bm25_k1, args.bm25_b, avgdl)
        else:
            X_main_tr = build_sublinear_tfidf_matrix(tokens_tr, vocab, idf)

        X_main_te: Optional[csr_matrix] = None
        if not args.val_only:
            tokens_te = preprocess_rich_corpus(texts_te, include_trigrams=include_tri)
            if args.pipeline == "bm25":
                X_main_te = build_bm25_matrix(tokens_te, vocab, idf, args.bm25_k1, args.bm25_b, avgdl)
            else:
                X_main_te = build_sublinear_tfidf_matrix(tokens_te, vocab, idf)

        if args.chi2_k > 0:
            X_main_tr, X_main_te = apply_chi2_selection(X_main_tr, y_tr_list, args.chi2_k, X_main_te)

        if args.lsi_components > 0:
            X_main_tr, X_main_te = apply_lsi(X_main_tr, X_main_te, n_components=args.lsi_components, random_state=args.random_state, n_iter=args.svd_iter)

        X_tr_dict = {args.pipeline: X_main_tr}
        X_te_dict = {}
        if not args.val_only and X_main_te is not None:
            X_te_dict[args.pipeline] = X_main_te

    else:
        # ---- Full ensemble (default) ----
        include_tri = not args.no_trigrams
        print("Building rich token corpus...", flush=True)
        tokens_tr = preprocess_rich_corpus(texts_tr, include_trigrams=include_tri)
        tokens_te_list = preprocess_rich_corpus(texts_te, include_trigrams=include_tri) if not args.val_only else []

        avgdl = float(np.mean([len(t) for t in tokens_tr])) if tokens_tr else 1.0
        X_tr_dict = {}
        X_te_dict = {}
        rep_weights: Dict[str, float] = {}

        if not args.no_bm25:
            print("Building BM25 matrix...", flush=True)
            vocab_bm25, idf_bm25, _ = build_vocab_idf(
                tokens_tr, min_df=args.min_df, max_vocab=args.max_vocab, idf_mode="bm25"
            )
            Xbm25_tr = build_bm25_matrix(tokens_tr, vocab_bm25, idf_bm25, args.bm25_k1, args.bm25_b, avgdl)
            Xbm25_te: Optional[csr_matrix] = None
            if not args.val_only:
                Xbm25_te = build_bm25_matrix(tokens_te_list, vocab_bm25, idf_bm25, args.bm25_k1, args.bm25_b, avgdl)
            if args.chi2_k > 0:
                Xbm25_tr, Xbm25_te = apply_chi2_selection(Xbm25_tr, y_tr_list, args.chi2_k, Xbm25_te)
            X_tr_dict["bm25"] = Xbm25_tr
            rep_weights["bm25"] = args.bm25_weight
            if not args.val_only and Xbm25_te is not None:
                X_te_dict["bm25"] = Xbm25_te

        if not args.no_sublinear:
            print("Building sublinear TF-IDF matrix...", flush=True)
            vocab_sub, idf_sub, _ = build_vocab_idf(
                tokens_tr, min_df=args.min_df, max_vocab=args.max_vocab, idf_mode="tfidf"
            )
            Xsub_tr = build_sublinear_tfidf_matrix(tokens_tr, vocab_sub, idf_sub)
            Xsub_te: Optional[csr_matrix] = None
            if not args.val_only:
                Xsub_te = build_sublinear_tfidf_matrix(tokens_te_list, vocab_sub, idf_sub)
            if args.chi2_k > 0:
                Xsub_tr, Xsub_te = apply_chi2_selection(Xsub_tr, y_tr_list, args.chi2_k, Xsub_te)
            X_tr_dict["sublinear"] = Xsub_tr
            rep_weights["sublinear"] = args.sublinear_weight
            if not args.val_only and Xsub_te is not None:
                X_te_dict["sublinear"] = Xsub_te

        if not args.no_char:
            print("Building char n-gram matrix...", flush=True)
            Xchar_tr, char_vec = build_char_ngram_matrix(
                texts_tr,
                ngram_range=(args.char_ngram_min, args.char_ngram_max),
                max_features=args.char_max_features,
                min_df=args.char_min_df,
            )
            X_tr_dict["char"] = Xchar_tr
            rep_weights["char"] = args.char_weight
            if not args.val_only:
                Xchar_te, _ = build_char_ngram_matrix(
                    texts_te,
                    fit_vectorizer=char_vec,
                )
                X_te_dict["char"] = Xchar_te

        if args.lsi_components > 0 and "bm25" in X_tr_dict:
            print(f"Building LSI ({args.lsi_components} components)...", flush=True)
            lsi_base = X_tr_dict["bm25"]
            lsi_base_te = X_te_dict.get("bm25")
            Xlsi_tr, Xlsi_te = apply_lsi(
                lsi_base, lsi_base_te if not args.val_only else None,
                n_components=args.lsi_components,
                random_state=args.random_state,
                n_iter=args.svd_iter,
            )
            X_tr_dict["lsi"] = Xlsi_tr
            rep_weights["lsi"] = args.lsi_weight
            if not args.val_only and Xlsi_te is not None:
                X_te_dict["lsi"] = Xlsi_te

        # PRF on BM25 queries (most effective representation)
        if args.prf_n > 0 and "bm25" in X_te_dict:
            print(f"Applying PRF (top-{args.prf_n}) to BM25 queries...", flush=True)
            X_te_dict["bm25"] = apply_prf(
                X_te_dict["bm25"], X_tr_dict["bm25"],
                top_n=args.prf_n, alpha=args.prf_alpha,
            )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    if not args.skip_val:
        f1 = run_validation(
            X_tr_dict, y_tr_list,
            val_size=args.val_size,
            random_state=args.random_state,
            k=args.k,
            metric=args.metric,
            voting=args.voting,
            batch_size=args.batch_size,
            sim_power=args.sim_power,
            rep_weights=rep_weights if args.pipeline == "ensemble" else None,
        )
        reps = "+".join(X_tr_dict.keys())
        print(f"Validation macro-F1 [{reps}] (val_size={args.val_size}): {f1:.4f}")

    if args.val_only:
        return 0

    # -----------------------------------------------------------------------
    # Full training + test predictions
    # -----------------------------------------------------------------------

    print("Fitting ensemble on full training set...", flush=True)
    ens = EnsembleKNNClassifier(
        k=args.k,
        metric=args.metric,
        voting=args.voting,
        batch_size=args.batch_size,
        sim_power=args.sim_power,
        rep_weights=rep_weights if args.pipeline == "ensemble" else None,
    )
    ens.fit(X_tr_dict, y_tr)
    pred_te = ens.predict(X_te_dict)
    write_predictions(out_path, pred_te)
    print(f"Wrote {len(pred_te)} lines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())