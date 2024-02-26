from matplotlib.colors import LightSource
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from collections.abc import Callable
import sys

sys.path.append("..")
from cfg import rng
from abstract_func import AbstractFunc


class Func2d(AbstractFunc):
  """
  Функция на плоскости
  Область определения - квадрат с центром в (0,0) и заданной стороной
  """
  
  def __init__(self, expr: Callable, domain_lim: float):
    super().__init__()
    self.expr = expr
    self.domain_lim = domain_lim


  def __call__(self, point: ndarray) -> float:
    """Вызов экземляра объекта как функции в некоторой точке"""
    
    super().__call__()
    return self.expr(*point)


  def domain_random(self) -> ndarray:
    """Генерирует случайную точку в области определения с равномерным распределением"""
    
    super().domain_random()
    return rng().uniform(-self.domain_lim, self.domain_lim, (2,))


  def neigh_random(self, point: ndarray, diam: float) -> ndarray:
    """Генерирует случайную точку в области определения из окрестности другой точки"""
    
    super().neigh_random()
    return np.array(
      [
        rng().uniform(
          max(-self.domain_lim, point[0] - diam), min(self.domain_lim, point[0] + diam)
        ),
        rng().uniform(
          max(-self.domain_lim, point[1] - diam), min(self.domain_lim, point[1] + diam)
        ),
      ]
    )
  

  def domain_size(self) -> float:
    """Сторона квадрата области определения, нужна для пчел"""
    
    return self.domain_lim


  def dim(self) -> int:
    """Размерность функции"""
    
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
