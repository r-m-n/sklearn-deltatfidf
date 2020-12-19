from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer, _document_frequency


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
