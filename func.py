import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import numpy as np
from numpy import ndarray
from enum import Enum
from copy import deepcopy as cp
from functools import reduce
from operator import mul
import typing as tp

rng = np.random.default_rng(424)


class Clamping(Enum):
    ALLOW_CROSSING = 0
    STOP_ON_BOUNDARY = 1
    STAND_STILL = 2
    FOREACH_COMPONENT = 3


def clamp(lo: float, value: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


class HyperCube:

    __slots__ = ("lo", "hi", "dims", "diameter")

    # lo: list[int]
    # hi: list[int]
    # dims: int
    # diameter: float

    def __init__(self, lo: list[int], hi: list[int]):
        assert isinstance(lo, list) and isinstance(hi, list) and len(lo) == len(hi)

        self.lo = lo
        self.hi = hi
        self.dims = len(lo)
        self.diameter = np.sqrt(
            sum((self.hi[i] - self.lo[i]) ** 2 for i in range(self.dims))
        )

    def _is_within(self, point: ndarray) -> bool:
        return all(self.lo[i] <= point[i] <= self.hi[i] for i in range(self.dims))

    def _assert_ndarray_point(self, point: ndarray):
        assert (
            isinstance(point, ndarray)
            and len(point.shape) == 1
            and point.shape[0] == self.dims
        )

    def _assert_meshgrid_or_point(self, arr: ndarray):
        assert isinstance(arr, ndarray) and (
            len(arr.shape) == 1 and arr.shape[0] == self.dims or len(arr.shape) == 3
        )

    def _move_clamping(
        self, point: ndarray, by: ndarray, clamping=Clamping.STOP_ON_BOUNDARY
    ) -> tuple[ndarray, bool]:

        result = point + by
        if self._is_within(result):
            return result, False

        match clamping:
            case Clamping.ALLOW_CROSSING:
                return result, True

            case Clamping.STOP_ON_BOUNDARY:
                alpha = min(
                    (
                        ((self.hi[i] - point[i]) / by[i])
                        if by[i] > 0
                        else ((point[i] - self.lo[i]) / (-by[i]))
                    )
                    for i in range(self.dims)
                    if by[i] != 0
                )
                return point + by * alpha, True

            case Clamping.STAND_STILL:
                return point, True

            case Clamping.FOREACH_COMPONENT:
                for i in range(self.dims):
                    result[i] = clamp(self.lo[i], result[i], self.hi[i])
                return result, True

    def _random_point(
        self,
        nearto: ndarray = None,
        dist: float = 0,
        cubic: bool = True,
    ) -> ndarray:
        if nearto is None:
            return np.array(
                [rng.uniform(self.lo[i], self.hi[i]) for i in range(self.dims)]
            )

        if cubic:
            by = rng.uniform(-dist, dist, size=self.dims)
        else:
            r = rng.uniform(0, dist)
            a = rng.uniform(-np.pi / 2, np.pi / 2, size=self.dims - 1)
            a[0] = rng.uniform(0, 2 * np.pi)
            by = np.array(
                [r * reduce(mul, (np.cos(a[j]) for j in range(self.dims - 1)))]
                + [
                    r
                    * np.sin(i)
                    * reduce(mul, (np.cos(a[j]) for j in range(i + 1, self.dims - 1)))
                    for i in range(self.dims - 1)
                ]
            )

        point, clamped = self._move_clamping(nearto, by, Clamping.STOP_ON_BOUNDARY)
        return point


class Func:

    __slots__ = (
        "name",
        "expr",
        "eval_calls",
        "hcube",
        "exact_min_point",
        "exact_min_value",
        "found_min_point",
        "found_min_value",
    )

    def __init__(
        self,
        name: str,
        expr: tp.Callable,
        hcube_lo: list[int],
        hcube_hi: list[int],
        min_point: ndarray = None,
    ):
        self.name = name
        self.expr = expr
        self.eval_calls = 0
        self.hcube = HyperCube(hcube_lo, hcube_hi)

        if min_point is not None:
            self.hcube._assert_ndarray_point(min_point)
            assert self.hcube._is_within(min_point)
            self.exact_min_point = min_point
            self.exact_min_value = self(min_point)
        else:
            self.exact_min_point = None
            self.exact_min_value = None

        self.found_min_point = self.random_point()
        self.found_min_value = self(self.found_min_point)


    def __call__(self, point: ndarray):
        self.eval_calls += 1
        return self.expr(point)

    def dims(self) -> float:
        return self.hcube.dims

    def diameter(self) -> float:
        return self.hcube.diameter

    def is_within(self, point: ndarray) -> bool:
        self.hcube._assert_ndarray_point(point)
        return self.hcube._is_within(point)

    def random_point(
        self, nearto: ndarray = None, dist: float = 0, cubic: bool = True
    ) -> ndarray:
        if nearto is not None:
            self.hcube._assert_ndarray_point(nearto)
            assert self.hcube._is_within(nearto) and dist > 0
        return self.hcube._random_point(nearto, dist, cubic)

    def move(
        self, point: ndarray, by: ndarray, clamping=Clamping.STOP_ON_BOUNDARY
    ) -> tuple[ndarray, bool]:
        return self.hcube._move_clamping(point, by, clamping)

    def plot(self, ax, grid_density=64, cmap=cm.Oranges, azdegree=135, altdegree=180):
        if self.hcube.dims != 2:
            return

        lo = self.hcube.lo
        hi = self.hcube.hi

        x = np.linspace(lo[0], hi[0], grid_density)
        y = np.linspace(lo[1], hi[1], grid_density)
        grid = np.array(np.meshgrid(x, y))
        z = self(grid)

        ax.view_init(30, 60)
        ls = LightSource(azdegree, altdegree)
        rgb = ls.shade(z, cmap=cmap, vert_exag=0.1, blend_mode="soft")
        ax.plot_surface(*grid, z, facecolors=rgb, antialiased=True, shade=True)
        ax.set_title(self.name, fontsize=20, y=-0.05)
        # ax.plot_surface(*grid, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # ax.contour3D(*grid, z, 50, cmap="magma")
        # ax.plot_wireframe(*grid, z, cmap="magma")


def ackley(dims: int):
    A = 20
    B = 0.2
    C = 2 * np.pi
    return Func(
        "Acley",
        lambda x: -A * np.exp(-B * np.sqrt(sum(x_i**2 for x_i in x) / dims))
        - np.exp(np.sum(np.cos(C * x_i) for x_i in x) / dims)
        + A
        + np.exp(1),
        # [-32.768, 32.768]
        [-30] * dims,
        [30] * dims,
        np.zeros(shape=dims),
    )


def griewank(dims: int):
    A = 4000
    return Func(
        "Griewank",
        lambda x: np.sum(x_i**2 for x_i in x) / A
        - reduce(
            np.multiply, (np.cos(x[i - 1] / np.sqrt(i)) for i in range(1, dims + 1))
        )
        + 1,
        [-80] * dims,
        [80] * dims,
        np.zeros(shape=dims),
    )


def schwefel(dims: int):
    A = 418.9829
    return Func(
        "Schwefel",
        lambda x: A * dims - np.sum(x_i * np.sin(np.sqrt(np.abs(x_i))) for x_i in x),
        [-500] * dims,
        [500] * dims,
        np.array([420.9687] * dims),
    )


def rosenbrock(dims: int):
    A = 100
    return Func(
        "Rosenbrok",
        lambda x: np.sum(
            A * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(dims - 1)
        ),
        [-2.048] * dims,
        [2.048] * dims,
        np.ones(shape=dims),
    )


def skin(dims: int):
    assert dims % 2 == 0
    return Func(
        "Skin",
        lambda x: np.sum(
            (np.cos(2 * x[2 * i] ** 2) - 1.1) ** 2
            + (np.sin(x[2 * i] / 2) - 1.2) ** 2
            - (np.cos(2 * x[2 * i + 1] ** 2) - 1.1) ** 2
            + (np.sin(x[2 * i + 1] / 2) - 1.2) ** 2
            for i in range(dims // 2)
        ),
        # [-5, 5]
        [-2] * dims,
        [4] * dims,
        np.array([3.07021, 3.315935] * (dims // 2)),
    )


def forest(dims: int):
    assert dims % 2 == 0
    return Func(
        "Forest",
        lambda x: np.sum(
            np.sin(np.sqrt(np.abs(x[2 * i] - 1.3) + np.abs(x[2 * i + 1] - 2)))
            + np.cos(
                np.sqrt(np.abs(np.sin(x[2 * i])))
                + np.sqrt(np.abs(np.sin(x[2 * i + 1] - 2)))
            )
            ** 4
            for i in range(dims // 2)
        ),
        # [-50, -18]
        [-40] * dims,
        [-20] * dims,
        np.array([-25.132741228718345, -32.55751918948773] * (dims // 2)),
    )


def megacity(dims: int):
    assert dims % 2 == 0
    return Func(
        "Megacity",
        lambda x: np.sum(
            np.floor(
                (
                    np.sin(np.sqrt(np.abs(x[2 * i] - 1.3) + np.abs(x[2 * i + 1] - 2)))
                    + np.cos(
                        np.sqrt(np.abs(np.sin(x[2 * i])))
                        + np.sqrt(np.abs(np.sin(x[2 * i + 1] - 2)))
                    )
                )
                ** 4
            )
            for i in range(dims // 2)
        ),
        # [-15, 15]
        [0] * dims,
        [10] * dims,
        np.array([3.16, 1.990] * (dims // 2))
    )


targets = [ackley, griewank, schwefel, rosenbrock, skin, forest, megacity]


ackley_func = ackley(2)
