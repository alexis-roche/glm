# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import scipy.stats as sp_stats

TINY = 1e-50
DOFMAX = 1e10


def zscore(pvalue):
    """ Return the z-score corresponding to a given p-value.
    """
    pvalue = np.minimum(np.maximum(pvalue, TINY), 1. - TINY)
    z = sp_stats.norm.isf(pvalue)
    return z


class GLM(object):

    def __init__(self, Y=None, X=None, formula=None, axis=0):

        # Check dimensions
        if Y == None:
            return
        else:
            self._fit(Y, X, formula, axis)

    def _fit(self, Y, X, formula=None, axis=0):

        if Y.shape[axis] != X.shape[0]:
            raise ValueError('Response and predictors are inconsistent')

        # Find axis
        self._axis = axis

        # Switch on models / methods
        out = ols(Y, X, axis=axis)

        # Finalize
        self.beta, self.nvbeta, self.s2, self.dof = out
        self.s2 = self.s2.squeeze()

    def save(self, file):
        """ Save fit into a .npz file
        """
        np.savez(file,
             beta=self.beta,
             nvbeta=self.nvbeta,
             s2=self.s2,
             dof=self.dof,
             method=self.method,
             axis=self._axis)

    def contrast(self, c, kind='t'):
        """ Specify and estimate a constrast

        c must be a numpy.ndarray (or anything that numpy.asarray
        can cast to a ndarray).
        For a F contrast, c must be q x p
        where q is the number of contrast vectors and
        p is the total number of regressors.
        """
        c = np.asarray(c)
        #dim = len(c.shape)
        if c.ndim == 1:
            dim = 1
        else:
            dim = c.shape[0]
        axis = self._axis
        ndims = len(self.beta.shape)

        # Compute the contrast estimate: c*B
        B = np.rollaxis(self.beta, axis, ndims)
        con = np.inner(c, B) # shape = q, X

        # Compute the variance of the contrast estimate: s2 * (c' * nvbeta * c)
        s2 = self.s2.squeeze()
        nvbeta = self.nvbeta
        if dim == 1:
            nvcon = np.inner(c, np.inner(c, nvbeta))
        else:
            nvcon = np.dot(c, np.inner(nvbeta, c)) # q, q

        # Create contrast instance
        c = Contrast(dim, kind)
        c.effect = con
        c.nvar = nvcon
        c.s2 = s2
        c.dof = self.dof
        return c


class Contrast(object):

    def __init__(self, dim, kind='t'):
        """tiny is a numerical constant for computations.
        """
        self.dim = dim
        self.effect = None
        self.nvar = None
        self.s2 = None
        self.dof = None
        if dim > 1:
            if kind is 't':
                kind = 'F'
        self.kind = kind
        self._stat = None
        self._pvalue = None
        self._baseline = 0

    def get_variance(self):
        if self.dim == 1:
            vcon = self.nvar.squeeze() * self.s2
        else:
            aux = self.nvar.shape
            vcon = np.resize(self.nvar, self.s2.shape + aux) # X, q, q
            vcon = vcon.T.reshape(aux + (self.s2.size,)) * \
                self.s2.reshape((self.s2.size,)) # q, q, Xflat
            vcon = vcon.reshape(aux + self.s2.shape) # q, q, X
        return vcon

    variance = property(get_variance)

    def summary(self):
        """
        Return a dictionary containing the estimated contrast effect,
        the associated ReML-based estimation variance, and the estimated
        degrees of freedom (variance of the variance).
        """
        return {'effect': self.effect,
                'variance': self.variance,
                'dof': self.dof}

    def stat(self, baseline=0.0):
        """
        Return the decision statistic associated with the test of the
        null hypothesis: (H0) 'contrast equals baseline'
        """
        self._baseline = baseline

        # Case: one-dimensional contrast ==> t or t**2
        if self.dim == 1:
            # avoids division by zero
            t = (self.effect - baseline) / np.sqrt(
                np.maximum(self.variance, TINY))
            if self.kind == 'F':
                t = t ** 2
        # Case: F contrast
        elif self.kind == 'F':
            # F = |t|^2/q ,  |t|^2 = e^t v-1 e
            aux = self.effect - baseline
            aux = aux.reshape((aux.shape[0], np.prod(aux.shape[1:])))
            A = np.linalg.inv(self.nvar)
            t = np.sum(aux * np.dot(A, aux), 0)
            t /= np.maximum(self.s2.reshape((self.s2.size, )) * self.dim, 
                            TINY)
            t = t.reshape(aux.shape[1:])
        # Case: tmin (conjunctions)
        elif self.kind == 'tmin':
            vdiag = self.variance.reshape([self.dim ** 2] + list(
                    self.variance.shape[2:]))[:: self.dim + 1]
            t = (self.effect - baseline) / np.sqrt(
                np.maximum(vdiag, TINY))
            t = t.min(0)

        # Unknwon stat
        else:
            raise ValueError('Unknown statistic kind')
        self._stat = t
        return t

    def pvalue(self, baseline=0.0):
        """
        Return a parametric approximation of the p-value associated
        with the null hypothesis: (H0) 'contrast equals baseline'
        """
        if self._stat == None or not self._baseline == baseline:
            self._stat = self.stat(baseline)
        # Valid conjunction as in Nichols et al, Neuroimage 25, 2005.
        if self.kind in ['t', 'tmin']:
            p = sp_stats.t.sf(self._stat, np.minimum(self.dof, DOFMAX))
        elif self.kind == 'F':
            p = sp_stats.f.sf(self._stat, self.dim, np.minimum(
                    self.dof, DOFMAX))
        else:
            raise ValueError('Unknown statistic kind')
        self._pvalue = p
        return p

    def zscore(self, baseline=0.0):
        """
        Return a parametric approximation of the z-score associated
        with the null hypothesis: (H0) 'contrast equals baseline'
        """
        if self._pvalue == None or not self._baseline == baseline:
            self._pvalue = self.pvalue(baseline)

        # Avoid inf values kindly supplied by scipy.
        z = zscore(self._pvalue)
        return z

    def __add__(self, other):
        if self.dim != other.dim:
            return None
        con = Contrast(self.dim)
        con.kind = self.kind
        con.effect = self.effect + other.effect
        con.variance = self.variance + other.variance
        con.dof = self.dof + other.dof
        return con

    def __rmul__(self, other):
        k = float(other)
        con = Contrast(self.dim)
        con.kind = self.kind
        con.effect = k * self.effect
        con.variance = k ** 2 * self.variance
        con.dof = self.dof
        return con

    __mul__ = __rmul__

    def __div__(self, other):
        return self.__rmul__(1 / float(other))


def ols(Y, X, axis=0):
    """Essentially, compute pinv(X)*Y
    """
    ndims = len(Y.shape)
    pX = np.linalg.pinv(X)
    beta = np.rollaxis(np.inner(pX, np.rollaxis(Y, axis, ndims)), 0, axis + 1)
    nvbeta = np.inner(pX, pX)
    res = Y - np.rollaxis(
        np.inner(X, np.rollaxis(beta, axis, ndims)), 0, axis + 1)
    n = res.shape[axis]
    s2 = (res ** 2).sum(axis) / float(n - X.shape[1])
    dof = float(X.shape[0] - X.shape[1])
    return beta, nvbeta, s2, dof


def load(file):
    """Load a fitted glm
    """
    from os.path import splitext
    if splitext(file)[1] == '':
        file = file + '.npz'
    fmod = np.load(file)
    mod = glm()
    mod.beta = fmod['beta']
    mod.nvbeta = fmod['nvbeta']
    mod.s2 = fmod['s2']
    mod.dof = fmod['dof']
    mod.method = str(fmod['method'])
    mod._axis = int(fmod['axis'])
    return mod
