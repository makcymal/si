import numpy as np
from func2d import Func2d


def ackley():
  A = 20
  B = 0.2
  C = 2 * np.pi
  return Func2d(
    lambda x, y: -A * np.exp(-B * np.sqrt((x ** 2 + y ** 2) / 2)) - np.exp((np.cos(C * x) + np.cos(C * y)) / 2),
    40
  )


def griewank():
  A = 4000
  return Func2d(
    lambda x, y: (x ** 2 + y ** 2) / A - np.cos(x) * np.cos(y / np.sqrt(2)),
    20
  )


def rosenbrock():
  A = 100
  return Func2d(
    lambda x, y: A * (y - x ** 2) ** 2 + (x - 1) ** 2,
    10
  )


def schwefel():
  return Func2d(
    lambda x, y: -(x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y)))),
    500
  )
