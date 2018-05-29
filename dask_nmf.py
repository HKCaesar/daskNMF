import numpy as np
import dask.array as da


# nndsvd/a not implemented in dask yet

def initialize_da(X, k, init = 'random', W=None, H=None):
    n_components = k
    n_samples, n_features = X.shape
    if init == 'random':
        avg = da.sqrt(X.mean() / n_components)
        H = avg * da.random.RandomState(42).normal(0,1,size=(n_components, n_features),chunks=(n_components,X.chunks[1][0]))
        W = avg * da.random.RandomState(42).normal(0,1,size=(n_samples, n_components),chunks=(n_samples,n_components))

        H = da.fabs(H)
        W = da.fabs(W)
        return W, H

    if init == 'nndsvd' or init == 'nndsvda':
    # not converted to da yet
        raise NotImplementedError

    if init == 'custom':
        return W, H


# random/NNDSVD/A initialization from sklearn
def initialize(X, k, init, W=None, H=None):

    n_components = k
    n_samples, n_features = X.shape

    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)
        H = avg * np.random.RandomState(42).normal(0,1,size=(n_components, n_features))
        W = avg * np.random.RandomState(42).normal(0,1,size=(n_samples, n_components))

        np.fabs(H, H)
        np.fabs(W, W)
        return W, H

    if init == 'nndsvd' or init == 'nndsvda':

        from scipy.linalg import svd

        U, S, V = svd(X, full_matrices = False)
        W, H = np.zeros(U.shape), np.zeros(V.shape)

        # The leading singular triplet is non-negative
        # so it can be used as is for initialization.
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

        def norm(x):
            x = x.ravel()
            return(np.dot(x,x))

        for j in range(1, n_components):
            x, y = U[:, j], V[j, :]

            # extract positive and negative parts of column vectors
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

            # and their norms
            x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
            x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

            # choose update
            if m_p > m_n:
                u = x_p / x_p_nrm
                v = y_p / y_p_nrm
                sigma = m_p
            else:
                u = x_n / x_n_nrm
                v = y_n / y_n_nrm
                sigma = m_n

            lbd = np.sqrt(S[j] * sigma)
            W[:, j] = lbd * u
            H[j, :] = lbd * v

        eps=1e-6

        if init == 'nndsvd':
            W[W < eps] = 0
            H[H < eps] = 0

        if init == 'nndsvda':
            avg = X.mean()
            W[W == 0] = avg
            H[H == 0] = avg

        return W, H

    if init == 'custom':
        if np.min(H) < 0 or np.min(W) < 0:
            raise ValueError('H and W should be nonnegative')
        return W, H



# loss
def Frobenius_loss(M,H,W):
    return(linalg(M - np.dot(W,H)))
