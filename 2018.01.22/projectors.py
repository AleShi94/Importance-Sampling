import numpy as np


class DummyProjector(object):
    def __call__(self, *x):
        raise NotImplemented


class EyeProjector(DummyProjector):
    def __call__(self, x):
        return x


class ConstantProjector(DummyProjector):
    def __init__(self, c):
        self.c = c

    def __call__(self, x):
        return self.c


class CoordProjector(DummyProjector):
    def __init__(self, *coord_projectors):
        self.projectors = coord_projectors

    def __call__(self, *x):
        return tuple(proj(c) for c, proj in zip(x, self.projectors))


class BoxProjector(DummyProjector):
    def __init__(self, min_values, max_values):
        self.min = min_values
        self.max = max_values

    def __call__(self, x):
        return np.clip(x, a_min=self.min, a_max=self.max)


class EigenBoxProjector(BoxProjector):
    def __call__(self, x):
        eig, v = np.linalg.eig(x)
        eig = np.clip(eig, a_min=self.min, a_max=self.max)
        return v.dot(np.diag(eig)).dot(v.T)

    
class SimplexProjector(DummyProjector):
    def __init__(self, epsilon):
        self.eps = epsilon
       
    def __call__(self, x):
        
        dim = len(x)
        num_eps = None
        
        x = x - np.min(x) + self.eps
        
        while np.around(np.sum(x), decimals= 8) != 1.0:
            diff = 1 - np.sum(x)
            num_eps = np.argwhere(x == self.eps).size
            
            if diff < 0:
                change = diff / (dim - num_eps)
                x = np.where(x + change < self.eps, self.eps, x + change)
            else:
                change = diff / dim
                x = x + change
        return x