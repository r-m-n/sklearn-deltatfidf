import six
import numbers
from collections import defaultdict, Counter

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, \
    _make_int_array, _document_frequency
from sklearn.utils.fixes import frombuffer_empty


class DeltaTfidfTransformer(TfidfTransformer):
    def fit(self, X_pos, X_neg, y):
        if not sp.issparse(X_pos):
            X_pos = sp.csc_matrix(X_pos)
        if not sp.issparse(X_neg):
            X_neg = sp.csc_matrix(X_neg)
        if self.use_idf:
            n_samples, n_features = X_pos.shape
            counter = Counter(y)
            n_pos_samples = counter[1]
            n_neg_samples = counter[-1]
            df_pos = _document_frequency(X_pos)
            df_neg = _document_frequency(X_neg)

            # perform idf smoothing if required
            df_pos += int(self.smooth_idf)
            df_neg += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)
            n_pos_samples += int(self.smooth_idf)
            n_neg_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(float(n_pos_samples) / df_pos) - np.log(float(n_neg_samples) / df_neg) + 1.0
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features,
                                        n=n_features, format='csr')

        return self


class DeltaTfidfVectorizer(TfidfVectorizer):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        super(TfidfVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self._tfidf = DeltaTfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    def _count_vocab(self, raw_documents, fixed_vocab, y=None):
        if not y:
            return super(DeltaTfidfVectorizer, self)._count_vocab(raw_documents, fixed_vocab)

        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = _make_int_array()
        values = _make_int_array()
        pos_values = _make_int_array()
        neg_values = _make_int_array()
        indptr.append(0)

        for i, doc in enumerate(raw_documents):
            feature_counter = defaultdict(int)
            pos_feature_counter = defaultdict(int)
            neg_feature_counter = defaultdict(int)
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    feature_counter[feature_idx] += 1
                    pos_feature_counter[feature_idx] += int(y[i] == 1)
                    neg_feature_counter[feature_idx] += int(y[i] == -1)
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            pos_values.extend(pos_feature_counter.values())
            neg_values.extend(neg_feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = frombuffer_empty(values, dtype=np.intc)
        pos_values = frombuffer_empty(pos_values, dtype=np.intc)
        neg_values = frombuffer_empty(neg_values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X.sort_indices()

        X_pos = sp.csr_matrix((pos_values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X_pos.sort_indices()

        X_neg = sp.csr_matrix((neg_values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X_neg.sort_indices()

        return vocabulary, X, X_pos, X_neg

    def _sort_features(self, X, X_pos, X_neg, vocabulary):
        sorted_features = sorted(six.iteritems(vocabulary))
        map_index = np.empty(len(sorted_features), dtype=np.int32)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode='clip')
        X_pos.indices = map_index.take(X_pos.indices, mode='clip')
        X_neg.indices = map_index.take(X_neg.indices, mode='clip')
        return X, X_pos, X_neg

    def _limit_features(self, X, X_pos, X_neg, vocabulary, high=None, low=None,
                        limit=None):
        if high is None and low is None and limit is None:
            return X, set()

        # Calculate a mask based on document frequencies
        dfs = _document_frequency(X)
        tfs = np.asarray(X.sum(axis=0)).ravel()
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low
        if limit is not None and mask.sum() > limit:
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(six.iteritems(vocabulary)):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return X[:, kept_indices], X_pos[:, kept_indices], X_neg[:, kept_indices], removed_terms

    def _fit_transform(self, raw_documents, y):
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X, X_pos, X_neg = self._count_vocab(raw_documents,
                                          self.fixed_vocabulary_, y)

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            X, X_pos, X_neg = self._sort_features(X, X_pos, X_neg, vocabulary)

            n_doc = X.shape[0]
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            X, X_pos, X_neg, self.stop_words_ = self._limit_features(X, X_pos, X_neg,
                                                                     vocabulary,
                                                                     max_doc_count,
                                                                     min_doc_count,
                                                                     max_features)

            self.vocabulary_ = vocabulary

        return X, X_pos, X_neg

    def fit(self, raw_documents, y):
        X, X_pos, X_neg = self._fit_transform(raw_documents, y)
        self._tfidf.fit(X_pos, X_neg, y)
        return self

    def fit_transform(self, raw_documents, y):
        X, X_pos, X_neg = self._fit_transform(raw_documents, y)
        self._tfidf.fit(X_pos, X_neg, y)
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents):
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self._check_vocabulary()

        # use the same matrix-building strategy as fit_transform
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
        return X
