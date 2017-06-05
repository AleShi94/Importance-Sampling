import numpy as np
import scipy as sp

def _dot(a, b):
    return sum(np.inner(x.ravel(), y.ravel()) for x, y in zip(a, b))


class ExponentialFamilyDistribution(object):
    def __init__(self, *theta):
        self.theta = theta

    def A(self):
        return NotImplemented

    def T(self, x):
        return NotImplemented

    def h(self, x):
        return NotImplemented

    def pdf(self, x):
        return self.h(x) * np.exp(_dot(self.theta, self.T(x)) - self.A())

    def sample(self):
        raise NotImplemented


class GaussianDistribution(ExponentialFamilyDistribution):
    def __init__(self, *theta):
        super(GaussianDistribution, self).__init__(*theta)

        m, S = theta
        self.m = m
        self.S = S
        self.cov = np.linalg.pinv(S)
        self.mu = self.cov.dot(m)
        self.k = m.shape[0]

    def A(self):
        return 0.5 * (np.inner(self.m, self.cov.dot(self.m)) - np.log(np.linalg.det(self.S)))

    def grad_A(self):
        # \nabla_S - log(det(S)) = - S^{-1} = - cov
        # \nabla_S^{-1} m^T S^{-1} m =  mm^T
        # \nabla_S m^T S^{-1} m =  -S^{-1} mm^T S^{-1}

        t = self.mu
        return (self.mu, -0.5 * (np.outer(t, t) + self.cov))

    def T(self, x):
        return (x, -0.5 * np.outer(x, x))

    def h(self, x):
        return (2 * np.pi) ** (-self.k / 2.)

    def sample(self):
        return np.random.multivariate_normal(mean=self.mu, cov=self.cov)



class DirichletDistribution(ExponentialFamilyDistribution):
    def __init__(self, theta):
        super(DirichletDistribution, self).__init__(theta)


        # thetas are original parameters alpha of dirichlet distribution 
        self.theta = theta

    def A(self):
        return np.log(np.sum(sp.special.gamma(self.theta))) - np.log(sp.special.gamma(np.sum(self.theta)))

    def grad_A(self):
        grad = [sp.special.psi(self.theta[i]) / sp.special.gamma(self.theta[i]) - sp.special.psi(np.sum(self.theta[i])) / sp.special.gamma(np.sum(self.theta[i])) for i in range(len(self.theta))]
        return grad

    def T(self, x):
        return np.log(x)

    def h(self, x):
        return 1./np.exp(np.log(x).sum())

    def sample(self):
        return np.random.dirichlet(self.theta)
