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

    if init == 'random_vcol':

        import math
        #p_c = options.get('p_c', int(ceil(1. / 5 * X.shape[1])))
        #p_r = options.get('p_r', int(ceil(1. / 5 * X.shape[0])))
        p_c = int(math.ceil(1. / 5 * X.shape[1]))
        p_r = int(math.ceil(1. / 5 * X.shape[0]))
        prng = np.random.RandomState(42)


        #W = da.zeros((X.shape[0], n_components), chunks = (X.shape[0],n_components))
        #H = da.zeros((n_components, X.shape[1]), chunks = (n_components,X.chunks[1][0]))

        W = []
        H = []

        for i in range(k):
            W.append (X[:, prng.randint(low=0, high=X.shape[1], size=p_c)].mean(axis=1).compute())
            H.append (X[prng.randint(low=0, high=X.shape[0], size=p_r), :].mean(axis=0).compute())
        W = np.stack(W, axis=1)
        H = np.stack(H, axis=0)



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



    if init == 'random_vcol':

        import math
        #p_c = options.get('p_c', int(ceil(1. / 5 * X.shape[1])))
        #p_r = options.get('p_r', int(ceil(1. / 5 * X.shape[0])))
        p_c = int(math.ceil(1. / 5 * X.shape[1]))
        p_r = int(math.ceil(1. / 5 * X.shape[0]))
        prng = np.random.RandomState(42)


        W = np.mat(np.zeros((X.shape[0], k)))
        H = np.mat(np.zeros((k, X.shape[1])))

        for i in range(k):
            W[:, i] = X[:, prng.randint(
                    low=0, high=X.shape[1], size=p_c)].mean(axis=1)
            H[i, :] = X[
                    self.prng.randint(low=0, high=V.shape[0], size=p_r), :].mean(axis=0)

        return W, H
#-----------------------------------
# Updates Dask
#
def update_H_da(M,H,W):
    denominator = da.dot(W.T,da.dot(W,H))
    denominator_new = da.where(da.fabs(denominator) < EPSILON,EPSILON,denominator)
    H_new = H*da.dot(W.T,M)/denominator_new
    return(H_new)


def update_W_da(M,H,W):
    denominator = da.dot(W,da.dot(H,H.T))
    denominator_new = da.where(da.fabs(denominator) < EPSILON,EPSILON,denominator)
    W_new = W*da.dot(M,H.T)/denominator_new
    return(W_new)

#-----------------------------------
# Updates Numpy
#


def update_W(M,H,W):
    denominator = (np.dot(W,np.dot(H,H.T)))
    denominator[np.abs(denominator) < EPSILON] = EPSILON
    W_new = W*np.dot(M,H.T)/denominator
    return(W_new)

def update_H(M,H,W):
    denominator = (np.dot(W.T,np.dot(W,H)))
    denominator[np.abs(denominator) < EPSILON] = EPSILON
    H_new = H*np.dot(W.T,M)/denominator
    return(H_new)

#---------------------
# fitting functions

# numpy fitting function
EPSILON = np.finfo(np.float32).eps
def fit(M, k, nofit, init, W=None, H=None):

    W, H = initialize(M, k, init, W, H)



    err = []
    for it in range(nofit):
        W = update_W(M,H,W)
        #print(np.sum(np.isnan(W)))
        H = update_H(M,H,W)
        err.append(linalg.norm(M - np.dot(W,H)))
        if it%10==0:
            print('Iteration '+str(it)+': error = '+ str(err[it]))
    return(W, H, err)

# dask fitting function
def fit_da(M, k, nofit, init='random', W=None, H=None):

    from dask import compute

    W, H = initialize_da(M, k, init, W, H)

    err = []
    for it in range(nofit):
        W = update_W_da(M,H,W)
        H = update_H_da(M,H,W)

        err.append(da.linalg.norm(M - da.dot(W,H)))

    return(W,H,err)

# loss
def Frobenius_loss(M,H,W):
    return(linalg(M - np.dot(W,H)))
