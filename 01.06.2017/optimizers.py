from tqdm import tqdm_notebook
import numpy as np


####### Projected gradient method #######

def step(old_theta, step_size, grad):
    """returns old_theta - step_size * grad"""
    return tuple(t - step_size * g for t, g in zip(old_theta, grad))


def projected_gradient(obj, f_class, projector, C, theta0, n_iters=5000):
    estimations = []
    thetas = [theta0]

    cumsum = 0
    theta = theta0
    for i in tqdm_notebook(range(1, n_iters + 1)):
        f = f_class(*theta)
        x = f.sample()
        # print(obj(x))
        cumsum += obj(x) / f.pdf(x)
        estimations.append(cumsum / i)

        mult = (obj(x) / f.pdf(x)) ** 2
        g = tuple(mult * (a - b) for a, b in zip(f.grad_A(), f.T(x)))
        theta = projector(*step(theta, C, g))
        thetas.append(theta)

    return thetas, np.array(estimations)


####### Mirror descent method #######

def mirror(x, g, alpha, dgf='xlogx'):
    n = len(x)
    updated_x = [x[i] * np.exp(- alpha * g[i]) for i in range(n)]
    return updated_x


def mirror_update(obj, f_class, mirror, C, theta0, n_iters=5000):
    estimations = []
    thetas = [theta0]

    cumsum = 0
    theta = theta0
    # print theta
    for i in tqdm_notebook(range(1, n_iters + 1)):
        f = f_class(*theta)
        x = f.sample()
        cumsum += obj(x) / f.pdf(x)
        estimations.append(cumsum / i)

        tmp = (obj(x) / f.pdf(x)) ** 2
        g = tuple(tmp * (a - b) for a, b in zip(f.grad_A(), f.T(x)))
        # print "gradient: ", g, "\t value: ", theta 
        alpha = C / i ** 0.5
        theta = mirror(theta, g, alpha)  # - C / i ** 0.5 *
        thetas.append((theta, ))

    return thetas, np.array(estimations)

def projected_mirror(obj, f_class, mirror, projector, C, theta0, n_iters=5000):
    estimations = []
    thetas = [theta0]

    cumsum = 0
    theta = theta0
    # print theta
    for i in tqdm_notebook(range(1, n_iters + 1)):
        f = f_class(*theta)
        x = f.sample()
        cumsum += obj(x) / f.pdf(x)
        estimations.append(cumsum / i)

        tmp = (obj(x) / f.pdf(x)) ** 2
        g = tuple(tmp * (a - b) for a, b in zip(f.grad_A(), f.T(x)))
        # print "gradient: ", g, "\t value: ", theta 
        alpha = C / i ** 0.5
        theta = projector(*mirror(theta, g, alpha))  # - C / i ** 0.5 *
        thetas.append((theta, ))

    return thetas, np.array(estimations)


####### ADMM method #######
import scipy as sp


def __ravel(x):
    return np.concatenate([c.ravel() for c in x])


def __reshape(x, shapes):
    result = []
    shift = 0
    for shape in shapes:
        result.append(x[shift: shift + np.sum(shape)].reshape(shape))
        shift += np.sum(shape)
    return tuple(result)


def admm(obj, f_class, projector, beta, C, theta0, n_iters=1000):
    """realisation of https://arxiv.org/pdf/1211.0632.pdf"""
    estimations = []
    thetas = [theta0]

    cumsum = 0
    theta = theta0
    y = theta0
    theta_shape = tuple([x.shape for x in theta])
    l = tuple([np.zeros_like(x) for x in theta])
    for i in tqdm_notebook(range(1, n_iters + 1)):
        f = f_class(*y)
        x = f.sample()

        cumsum += obj(x) / f.pdf(x)
        estimations.append(cumsum / i)
        
        mult = (obj(x) / f.pdf(x)) ** 2
        g = tuple(mult * (a - b) for a, b in zip(f.grad_A(), f.T(x)))
        
        eta = C / i ** 0.5
        theta = tuple([(beta * y_part + l_part + theta_part / eta - g_part) / (beta + 1. / eta) 
                       for y_part, l_part, theta_part, g_part in zip(y, l, theta, g)])
        y = projector(*theta)
        l = step(l, beta, step(theta, 1, y))

        thetas.append(y)

    return thetas, np.array(estimations)
def adamc_step(old_theta, step_size, grad):
    """returns old_theta - step_size * grad"""
    return tuple(t - step_size * g for t, g in zip(old_theta, grad))

def adamc(obj, dist, projector, C, theta0, n_iter = 1e4):
    """Alena's realisation"""
    #cur_theta = theta0
    estim = []
    thetas = [theta0]
    summ = 0
    n_iter = int(n_iter)
    for i in tqdm_notebook(range(1, n_iter + 1)):
        cur_dist = dist(*thetas[-1])
        x = cur_dist.sample()
        summand = obj(x) /cur_dist.pdf(x)
        summ += summand
        estim.append(summ / i)
        grad = ((g_a - t)*(summand **2) for g_a, t in zip(cur_dist.grad_A(), cur_dist.T(x)))
        new_theta = projector(*adamc_step(thetas[-1], C / np.sqrt(i), grad))
        thetas.append(new_theta)
    return thetas, estim
