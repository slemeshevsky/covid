from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y
import numpy as np

class ConstrainedLinearRegression(LinearRegression, RegressorMixin):

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, nonnegative=False, tol=1e-15):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.nonnegative = nonnegative
        self.tol = tol

    def fit(self, X, y, min_coef=None, max_coef=None):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=False)
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize, copy=self.copy_X)
        self.min_coef_ = min_coef if min_coef is not None else np.repeat(-np.inf, X.shape[1])
        self.max_coef_ = max_coef if max_coef is not None else np.repeat(np.inf, X.shape[1])

        if self.nonnegative:
            self.min_coef_ = np.clip(self.min_coef_, 0, None)

        beta = np.zeros(X.shape[1]).astype(float)
        prev_beta = beta + 1
        hessian = np.dot(X.transpose(), X)

        counter = 0
        max_counter = 100000
        #while not(np.linalg.norm(np.dot(X, beta) - y) < self.tol):
        while not (np.abs(prev_beta - beta) < self.tol).all():
            #counter = counter + 1
            #if counter % 1000 == 0:
            #    print(counter)
            #if counter >= max_counter:
            #    break

            prev_beta = beta.copy()
            for i in range(len(beta)):
                grad = np.dot(np.dot(X, beta) - y, X)

                a1 = self.max_coef_[i]
                a2 = self.min_coef_[i]
                a3 = beta[i]-grad[i] / hessian[i,i]
                a4 = np.maximum(a2, a3)
                a5 = np.minimum(a1, a4)

                beta[i] = a5

        self.coef_ = beta
        self._set_intercept(X_offset, y_offset, X_scale)
        return self
