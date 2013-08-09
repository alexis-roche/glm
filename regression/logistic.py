import numpy as np
from scipy.optimize import fmin_ncg
from scipy.stats import norm
from scipy.linalg import cholesky
from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian


force_positive = lambda x: np.maximum(x, 1e-25)


def logistic(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(object):

    def __init__(self, y, X, prior_var=None):
        """
        Logistic regression object

        y: 1d array of +/- 1, length nb subjects
        X: nb subjects x nb features 2d array

        Will compute the distribution of
        w: 1d array, length nb features
        """
        self.y = y
        self.X = X
        self.prior_var = prior_var
        self.cache = {'w': None, 'f': None, 'p': None, 'map': None}

    def update_cache(self, w):
        if not w is self.cache['w']:
            f = np.dot(self.X, w)
            yf = self.y * f
            self.cache['f'] = f
            self.cache['yf'] = yf
            self.cache['p'] = logistic(yf)
            self.cache['w'] = w

    def posterior(self, w):
        self.update_cache(w)
        return np.prod(self.cache['p'])

    def log_posterior(self, w):
        self.update_cache(w)
        return np.sum(np.log(self.cache['p'])) + self.log_prior(w)

    def log_posterior_grad(self, w):
        self.update_cache(w)
        a = self.y * (1 - self.cache['p'])
        return np.dot(self.X.T, a) + self.log_prior_grad(w)

    def log_posterior_hess(self, w):
        self.update_cache(w)
        a = self.cache['p'] * (1 - self.cache['p'])
        return -np.dot(a * self.X.T, self.X) + self.log_prior_hess(w)

    def log_prior(self, w):
        if self.prior_var is None:
            return 0
        tmp = -.5 * w.size * np.log(2 * np.pi * self.prior_var)
        return tmp - .5 * np.sum(w ** 2) / self.prior_var

    def log_prior_grad(self, w):
        if self.prior_var is None:
            return 0
        return -w / self.prior_var

    def log_prior_hess(self, w):
        if self.prior_var is None:
            return 0
        log_prior_hess = np.zeros((w.size, w.size))
        np.fill_diagonal(log_prior_hess, -1 / self.prior_var)
        return log_prior_hess

    def map(self, tol=1e-8):
        """
        Compute the maximum a posteriori regression coefficients.
        """
        cost = lambda w: -self.log_posterior(w)
        grad = lambda w: -self.log_posterior_grad(w)
        hess = lambda w: -self.log_posterior_hess(w)
        w0 = np.zeros(self.X.shape[1])
        w = fmin_ncg(cost, w0, grad, fhess=hess, avextol=tol, disp=False)
        self.cache['map'] = w
        return w

    def accuracy(self, w, klass=None):
        """
        Compute the correct classification rate for a given set of
        regression coefficients.
        """
        self.update_cache(w)
        if klass in (-1, 1):
            msk = np.where(self.y == klass)
        else:
            msk = slice(0, self.y.size)
        yf = self.cache['yf'][msk]
        y = self.y[msk]
        errors = np.where(yf < 0)
        return float(y.size - len(errors[0])) / y.size

    def laplace_approx(self):
        """
        Compute the Laplace approximation to the unnormalized
        posterior distribution of regression coefficients.
        """
        w = self.cache['map']
        if w is None:
            w = self.map()
        invV = -self.log_posterior_hess(w)
        logK = self.log_posterior(w)
        return Gaussian(w, np.linalg.pinv(invV), K=np.exp(logK))

    def ep_approx(self, niters, post_loop=False):
        """
        Approximate the unnormalized posterior distribution of
        regression coefficients using the EP algorithm under a probit
        model.
        """
        EP = ExpectationPropagation(self.y, self.X, self.prior_var)
        EP.run(niters, post_loop=post_loop)
        return EP.posterior_approx()

    def vb_approx(self, niters):
        """
        Approximate the unnormalized posterior distribution of
        regression coefficients using the variational bound method
        proposed by Jaakkola & Jordan, 1996, under a logistic model.
        """
        VB = VariationalBound(self.y, self.X, self.prior_var)
        VB.run(niters)
        return VB.update_approx()

    def sampling_approx(self, ndraws, method='VS'):
        fit0 = self.laplace_approx()
        target = lambda w: self.log_posterior(w)
        vs = VariationalSampler(target, (fit0.m, fit0.V), ndraws)
        if method == 'VS':
            f = vs.fit(minimizer='quasi_newton')
        elif method == 'IS':
            f = vs.fit(objective='l')
        elif method == 'BMC':
            var = np.trace(vs.kernel.V) / vs.kernel.dim
            f = vs.fit(objective='gp', var=var)
        else:
            raise ValueError('unknown sampling method')
        if method in ('IS', 'VS'):
            print('Evidence rel. error: %f' %\
                      (np.sqrt(f.var_integral[0, 0]) / f.fit.Z))
        return f.fit


class ProbitRegression(LogisticRegression):

    def __init__(self, y, X, prior_var=None):
        self.y = y
        self.X = X
        self.prior_var = prior_var
        self.cache = {'w': None, 'f': None, 'p': None, 'map': None}

    def update_cache(self, w):
        if not w is self.cache['w']:

            f = np.dot(self.X, w)
            yf = self.y * f
            self.cache['f'] = f
            self.cache['yf'] = yf
            self.cache['p'] = norm.cdf(yf)
            self.cache['aux'] = norm.pdf(yf) / self.cache['p']
            self.cache['w'] = w

    def log_posterior_grad(self, w):
        self.update_cache(w)
        a = self.y * self.cache['aux']
        return np.dot(self.X.T, a) + self.log_prior_grad(w)

    def log_posterior_hess(self, w):
        self.update_cache(w)
        a = self.cache['aux'] * (self.cache['yf'] + self.cache['aux'])
        return -np.dot(a * self.X.T, self.X) + self.log_prior_hess(w)


class ExpectationPropagation(object):

    def __init__(self, y, X, prior_var):
        self.y = y
        self.X = X
        self.fac_nu = np.zeros(y.size)
        self.fac_tau = np.zeros(y.size)
        self.fac_logZ = np.zeros(y.size)
        self.K = prior_var * np.dot(X, X.T)
        self.Sigma = self.K.copy()
        self.mu = np.zeros(y.size)

    def update_factor(self, i):
        """
        See Rasmussen and Williams (2006), chap. 3, p. 55-58
        """
        # Compute context (eq 3.56)
        tau = 1 / force_positive(self.Sigma[i, i])
        ctx_tau = force_positive(tau - self.fac_tau[i])
        ctx_nu = tau * self.mu[i] - self.fac_nu[i]
        ctx_var = 1 / ctx_tau
        ctx_mu = ctx_nu * ctx_var

        # Compute marginal target mean and variance (eq 3.58)
        y = self.y[i]
        tmp = force_positive(np.sqrt(1 + ctx_var))
        z = (y * ctx_mu) / tmp
        Z = force_positive(norm.cdf(z))
        tmp2 = norm.pdf(z) / Z
        mu = ctx_mu + (y * ctx_var) * tmp2 / tmp
        var = force_positive(ctx_var - (ctx_var ** 2 / (1 + ctx_var))\
                                 * tmp2 * (z + tmp2))

        # Update factor parameters (eq 3.59)
        dtau = 1 / var - ctx_tau - self.fac_tau[i]
        self.fac_tau[i] += dtau
        self.fac_nu[i] = mu / var - ctx_nu
        fac_var = 1 / force_positive(self.fac_tau[i])
        fac_mu = self.fac_nu[i] / force_positive(self.fac_tau[i])
        self.fac_logZ[i] = np.log(Z) + .5 * np.log(2 * np.pi)\
            + .5 * np.log(ctx_var + fac_var)\
            + .5 * (ctx_mu - fac_mu) ** 2 / (ctx_var + fac_var)

        # Update global covariance approximation
        si = np.reshape(self.Sigma[:, i], (self.y.size, 1))
        step = dtau / (1 + dtau * self.Sigma[i, i])
        self.Sigma -= step * np.dot(si, si.T)
        self.mu = np.dot(self.Sigma, self.fac_nu)

    def post_loop(self):
        s = np.sqrt(np.maximum(self.fac_tau, 0))
        B = np.eye(self.y.size) + (s * (self.K * s).T)
        b, P = np.linalg.eigh(B)
        b = force_positive(b)
        #b = np.maximum(b, 1)
        V = np.dot(np.dot(P * (b ** -.5), P.T), (s * self.K.T).T)
        Sigma = self.K - np.dot(V.T, V)
        print('*******************')
        print b.min(), b.max()
        print np.max(np.abs(Sigma - self.Sigma))
        self.Sigma = Sigma
        self.mu = np.dot(self.Sigma, self.fac_nu)

    def loop(self):
        for i in range(self.y.size):
            self.update_factor(i)

    def run(self, niters, post_loop=False):
        for it in range(niters):
            self.loop()
            if post_loop:
                self.post_loop()

    def logZ(self):
        """
        Careful here, we need to deal with factors that have
        potentially flat Gaussian approximations (zero precision,
        i.e. fac_tau=0). A direct implementation of Rasmussen's
        equation (3.65) turns out numerically unstable. We instead
        resort to the implementation described in Sec 3.6.3, from
        p. 59 onwards.

        We start with computing B = I + DKD, with D =
        diag(fac_tau**(1/2)), and its Cholesky decomposition, i.e. the
        upper triangular matrix L such that: B = L.T L

        Next we have four terms to compute and sum up.

        Term 1
        ------
        It is:

        .5 * log(1 + fac_tau/ctx_tau) - log(L_ii)

        with:

        1 + fac_tau/ctx_tau = tau / (tau - fac_tau)

        the log of which is: -log(1 - fac_tau/tau)
                 = -log(1 - diag(Sigma)*fac_tau)

        Term 2
        ------
        We need the matrix:

        K - K D B^-1 D K - diag(Sigma)

        Term 3
        ------
        We need to compute the context means, which are found to be:

        (mu - diag(Sigma) fac_nu) / (1 - diag(Sigma) fac_tau)

        Term 4
        ------
        We again need the context means as well as the context
        variances, which are

        diag(Sigma) / (1 - diag(Sigma) fac_tau)
        """
        app_diag_lr = lambda s, A: s.reshape(1, len(s))\
            * A * d.reshape(len(s), 1)
        d = self.fac_tau ** .5
        B = np.eye(len(d)) + app_diag_lr(d, self.K)
        L = cholesky(B)
        Linv = np.linalg.inv(L)
        Binv = np.dot(Linv, Linv.T)
        diag_var = np.diagonal(self.Sigma)
        # term 1
        aux = 1 - diag_var * self.fac_tau
        term1 = -.5 * np.sum(np.log(aux)) - np.sum(np.log(np.diagonal(L)))
        # term 2
        A = self.K - np.dot(np.dot(self.K, app_diag_lr(d, Binv)), self.K)\
            - np.diag(diag_var)
        term2 = .5 * np.sum(self.fac_nu * np.dot(A, self.fac_nu))
        # term 3
        ctx_mu = (self.mu - diag_var * self.fac_nu) / aux
        tmp = aux * (self.fac_tau * ctx_mu - 2 * self.fac_nu)
        term3 = .5 * np.sum(ctx_mu * tmp)
        # term 4
        ctx_var = diag_var / aux
        tmp = (self.y * ctx_mu) / np.sqrt(1 + ctx_var)
        term4 = np.sum(np.log(norm.cdf(tmp)))
        return term1 + term2 + term3 + term4

    def oldlogZ(self):
        fac_var = 1 / force_positive(self.fac_tau)
        fac_mu = fac_var * self.fac_nu
        A = self.K + np.diag(fac_var)
        a, P = np.linalg.eigh(A)
        a = force_positive(a)
        log_det = np.sum(np.log(a))
        maha = np.dot(fac_mu, np.dot(P * (1 / a), np.dot(P.T, fac_mu)))
        return -.5 * log_det - .5 * maha - self.y.size\
            * .5 * np.log(2 * np.pi) + self.fac_logZ.sum()

    def posterior_approx(self):
        """
        We have: f = Xw
        => w = Af with A = pinv(X)
        E(w) = A mu
        Var(w) = A Sigma A.T
        """
        A = np.linalg.pinv(self.X)
        Z = np.exp(self.logZ())
        m = np.dot(A, self.mu)
        V = np.dot(np.dot(A, self.Sigma), A.T)
        return Gaussian(m, V, Z=Z)


def gamma(x):
    """
    x / (sigma(x) - .5)
    """
    x = np.maximum(np.abs(x), 1e-5)
    return (logistic(x) - .5) / x


class VariationalBound(object):

    def __init__(self, y, X, prior_var):
        self.y = y
        self.X = X
        self.prior_var = prior_var
        self.XTX = np.dot(X.T, X)
        self.XTX_inv = np.linalg.inv(self.XTX)
        XTy = np.dot(X.T, y)
        self.beta = np.dot(self.XTX_inv, XTy)
        self.dot = np.dot(XTy.T, self.beta)
        self._xeta = 2.0
        self._Sigma = None
        self._mu = None

    def update_likelihood_approx(self):
        gam = gamma(self._xeta)
        fac = 1. / gam
        Sigma = fac * self.XTX_inv
        mu = .5 * fac * self.beta
        """
        normalizing constant
        """
        n = len(self.y)
        logK = n * np.log(logistic(self._xeta))
        logK += .5 * n * (gam * self._xeta ** 2 - self._xeta)
        logK += .125 * fac * self.dot
        return Gaussian(mu, Sigma, K=np.exp(logK))

    def update_approx(self):
        g = self.update_likelihood_approx()
        dim = len(g.m)
        prior = Gaussian(np.zeros(dim), self.prior_var * np.eye(dim))
        g = g * prior
        self._Sigma = g.V
        self._mu = g.m
        return g

    def update_xeta(self):
        dim = len(self._mu)
        aux = self._Sigma + np.dot(np.reshape(self._mu, (dim, 1)),
                                   np.reshape(self._mu, (1, dim)))
        xeta2 = np.trace(np.dot(self.XTX, aux)) / len(self.y)
        self._xeta = np.sqrt(xeta2)
        print self._xeta

    def run(self, niters):
        for it in range(niters):
            self.update_approx()
            self.update_xeta()
