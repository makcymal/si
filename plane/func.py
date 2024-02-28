from matplotlib.colors import LightSource
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from collections.abc import Callable
import sys

sys.path.append("..")
from abstract import AbstractFunc, DomainClamping, rng


class Func(AbstractFunc):
    """
    Функция на плоскости.

    Функция = вычисление значения + область определения.
    Про свою область определения знает только функция, поэтому
    она отвечает за генерацию новых точек в ней и движения агентов.

    Область определения - квадрат с центром в (0,0) и заданной стороной.
    """

    def __init__(self, expr: Callable, domain_lim: float):
        """
        Инициализация функции.

        Наследованные атрибуты:

        eval_calls - сколько раз было вызвано вычисление функции. Требуется для отслеживания
          эффективности алгоритмов.
        rand_calls - сколько раз была вызвана генерация случайной точки. Требуется также для
          отслеживания эффективности алгоритмов, хотя eval_calls предпочтительней, т.к. не все
          алгоритмы постоянно генерируют новые точки.

        Собственные атрибуты:

        expr - лямбда-выражение для вычисления функции
        domain_lim - половина стороны квадрата области определения
        """

        super().__init__()
        self.expr = expr
        self.domain_lim = domain_lim

    def __call__(self, coord: ndarray) -> float:
        """
        Вызов экземляра объекта как функции в некоторой точке (точках)

        Может быть передана одна точка, а может быть и целое множество.
        При этом множество можно передать как numpy.ndarray, тогда выражение
        для одной точки из арифметических операций и numpy-функций будет выполнено
        поэлементно для каждой точки всего множества.
        """

        self.eval_calls += 1
        return self.expr(*coord)

    def within_domain(self, coord: ndarray) -> bool:
        """
        Возвращает True, если точка находится внутри области определения, False иначе
        """

        return abs(coord[0]) < self.domain_lim and abs(coord[1]) < self.domain_lim

    def move_coord(self, coord: ndarray, by: ndarray, clamp: DomainClamping) -> bool:
        """
        Двигает данный coord на вектор shift.

        При выходе за границу области определения обращается к clamp.
        Предполагается, что coord представляет одну точку.

        Так как coord - не примитив, то передается по ссылке и может быть
        изменен внутри функции, и изменения будут видны извне.

        Возвращает True, если была попытка выхода за границу области определения,
        False иначе
        """

        if clamp == DomainClamping.ALLOW_CROSSING:
            coord += by
            return self.within_domain(coord)

        else:
            after = coord + by
            if self.within_domain(after):
                coord = after
                return True

            if clamp == DomainClamping.STAND_STILL:
                pass

            elif clamp == DomainClamping.STOP_ON_BOUNDARY:
                x = abs(coord[0])
                y = abs(coord[1])
                x_shrink = (self.domain_lim - x) / (abs(after[0]) - x)
                y_shrink = (self.domain_lim - y) / (abs(after[1]) - y)
                coord += min(x_shrink, y_shrink) * by

            elif clamp == DomainClamping.FOR_EACH_COMPONENT:
                if after[0] < -self.domain_lim:
                    after[0] = -self.domain_lim
                elif after[0] > self.domain_lim:
                    after[0] = self.domain_lim
                if after[1] < -self.domain_lim:
                    after[1] = -self.domain_lim
                elif after[1] > self.domain_lim:
                    after[1] = self.domain_lim

            return False

    def domain_random(self) -> ndarray:
        """Случайная точка в области определения с равномерным распределением"""

        self.rand_calls += 1
        return rng().uniform(-self.domain_lim, self.domain_lim, (2,))

    def neigh_random(self, coord: ndarray, neigh: float) -> ndarray:
        """
        Случайная точка в окрестности размера neigh точки coord в пересечении с
        областью определения с равномерным распределением
        """

        self.rand_calls += 1
        return np.array(
            [
                rng().uniform(
                    max(-self.domain_lim, coord[0] - neigh),
                    min(self.domain_lim, coord[0] + neigh),
                ),
                rng().uniform(
                    max(-self.domain_lim, coord[1] - neigh),
                    min(self.domain_lim, coord[1] + neigh),
                ),
            ]
        )

    def domain_size(self) -> float:
        """Сторона квадрата области определения, нужна для пчел"""
        return self.domain_lim

    def dim(self) -> int:
        """Размерность функции, кол-во принимаемых агрументов"""
        return 2

    def plot(self, grid_density=64, name=None):
        """
        Рисует график
        Если вызвать из юпитера, график отобразиться под вызовом
        Если передать имя, то график сохранится в файл
        """

        x = np.linspace(-self.domain_lim, self.domain_lim, grid_density)
        y = np.linspace(-self.domain_lim, self.domain_lim, grid_density)
        grid = np.meshgrid(x, y)
        z = self(grid)

        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

        ls = LightSource(90, 1000)
        rgb = ls.shade(z, cmap=cm.magma, vert_exag=0.1, blend_mode="soft")
        ax.plot_surface(*grid, z, facecolors=rgb, antialiased=True, shade=True)

        if name:
            fig.savefig(f"{name}.png", dpi=400)
