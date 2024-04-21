import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import numpy as np

rng = np.random.default_rng(424)


class Agent:
    def __init__(self, func, point=None):
        if not point:
            point = func.domain_random()
        self.point = point
        self.value = func(point)
        
    def move(self, func, rhs, rel=False):
        if rel:
            self.point += rhs
        else:
            self.point = rhs
        self.value = func(self.point)


class Func:
    def __init__(self, expr, domain):
        self.eval_calls = 0
        self.rand_calls = 0
        self.expr = expr
        self.domain = domain
        
    def __call__(self, point):
        self.eval_calls += 1
        return self.expr(*point)

    def within_domain(self, point):
        return abs(point[0]) < self.domain and abs(point[1]) < self.domain
    
    def domain_random(self):
        self.rand_calls += 1
        return rng.uniform(-self.domain, self.domain, (2,))

    def neigh_random(self, point, neigh):
        self.rand_calls += 1
        return np.array(
            [
                rng.uniform(
                    max(-self.domain, point[0] - neigh),
                    min(self.domain, point[0] + neigh),
                ),
                rng.uniform(
                    max(-self.domain, point[1] - neigh),
                    min(self.domain, point[1] + neigh),
                ),
            ]
        )
    
    def plot(self, ax, grid_density=64):
        x = np.linspace(-self.domain, self.domain, grid_density)
        y = np.linspace(-self.domain, self.domain, grid_density)
        grid = np.meshgrid(x, y)
        z = self(grid)

        ls = LightSource(90, 1000)
        rgb = ls.shade(z, cmap=cm.magma, vert_exag=0.1, blend_mode="soft")
        ax.plot_surface(*grid, z, facecolors=rgb, antialiased=True, shade=True)



def ackley():
    A = 20
    B = 0.2
    C = 2 * np.pi
    return Func(
        lambda x, y: -A * np.exp(-B * np.sqrt((x**2 + y**2) / 2))
        - np.exp((np.cos(C * x) + np.cos(C * y)) / 2),
        40,
    )


def griewank():
    A = 4000
    return Func(lambda x, y: (x**2 + y**2) / A - np.cos(x) * np.cos(y / np.sqrt(2)), 20)


def rosenbrock():
    A = 100
    return Func(lambda x, y: A * (y - x**2) ** 2 + (x - 1) ** 2, 10)


def schwefel():
    return Func(
        lambda x, y: -(x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y)))),
        500,
    )
