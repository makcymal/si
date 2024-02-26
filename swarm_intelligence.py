from typing import List, Iterable, Callable
from copy import deepcopy as cp
from math import sqrt

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from abstract_func import AbstractFunc
from cfg import rng


class Agent:
  """
  Агент - элементарная составляющая роя, будь то частица, пчела, муравей etc
  По сути просто точка в области определения функции + значение в ней
  Соответственно, зависит от самой функции
  """

  def __init__(self, func: AbstractFunc, coord: ndarray = None):
    """
    Новая точка + значение в ней
    AbstractFunc знает о своей области определения и умеет генерировать случайные точки в ней
    coord задано - используется как заданная точка, нет - генерируется случайная
    """

    if not coord:
      coord = func.domain_random()
    self.coord = coord
    self.value = func(coord)

  def goto(self, func: AbstractFunc, coord: ndarray, rel=False):
    """
    Переход в новую точку
    Относительный переход (rel == True) - к текущей координате прибавляется coord
    Абсолютный переход (rel == False) - текущая координата приравнивается к coord
    """

    if not rel:
      self.coord = coord
    else:
      self.coord += coord
    self.value = func(coord)

  def l1_dist(self, other: ndarray) -> float:
    """
    L1 норма разности между текущей и заданной точкой
    Сумма модулей компонент разности
    """
    return np.sum(np.abs(self.coord - other))

  def l2_dist(self, other: ndarray) -> float:
    """
    L2 норма разности между текущей и заданной точкой
    Корень суммы квадратов компонент разности
    """
    return np.sqrt(np.sum((self.coord - other) ** 2))

  def l_inf_dist(self, other: ndarray) -> float:
    """
    Linf норма разности между текущей и заданной точкой
    Максимум модулей компонент разности
    """
    return np.max(np.abs(self.coord - other))


def best_agent(swarm: List[Agent], minimize=True) -> int:
  cmp = (lambda a, b: a < b) if minimize else (lambda a, b: a > b)
  best_idx = 0
  for curr_idx, agent in enumerate(swarm):
    if cmp(agent.value, swarm[best_idx].value):
      best_idx = curr_idx
  return best_idx


def benchmark(
  func: AbstractFunc,
  algos: Iterable[Callable],
  legend: Iterable[str],
  n_iters=100,
  eval_calls_lim=1000,
  title=None,
):
  """
  Измеряет среднюю эффективность работы алгоритмов и рисует график
  Каждый алгоритм тратит некоторое количество раз вычисляет функцию (eval_calls)
  Чем меньше значение алгоритм находит к некоторому фиксированному eval_calls, тем лучше
  График сохраняется в папку benchmarks с именем title

  func - минимизируемая функция
  algos - список алгоритмов, переданных как функции
  legend - названия алгоритмов, появляются только в легенде графика
  n_iters - столько раз строится график для каждого алгоритма, а затем усредняется
  eval_calls_lim - когда выполнение алгоритма останавливается
  title - название графика и файла с ним
  """

  fig, axes = plt.subplots()
  lines = []

  for algo in algos:
    anchors = []

    for i in range(n_iters):
      for minima in algo(func, eval_calls_lim):
        anchors.append((func.eval_calls, minima.value))

    anchors.sort(key=lambda x: x[0])
    x = []
    y = []
    i = 0
    while i < len(anchors):
      if i == 0 or anchors[i][0] != x[-1]:
        x.append(anchors[i][0])

        value_sum = 0
        j = i + 1
        while j < len(anchors) and anchors[j][0] == x[-1]:
          value_sum += anchors[j][1]
          j += 1

        y.append(value_sum / (j - i))
        i = j

    lines.append(axes.plot(x, y)[0])

  axes.legend(tuple(lines), tuple(legend), loc="upper right")

  axes.set_xlabel("eval_calls")
  axes.set_ylabel("found_min")
  axes.set_title(title)
  axes.grid()
  fig.savefig(f"benchmarks/{title}.png", dpi=400)


def random_search(func: AbstractFunc, n_gener: 1000):
  """
  Самый тупой поиск простым перебором случайных точек
  Нужен здесь только для сравнения
  """

  minima = func.domain_random()
  for gener in range(n_gener):
    curr = func.domain_random()
    if curr.value < minima.value:
      minima = curr
  return minima


def random_search_bench(func: AbstractFunc, eval_calls: int):
  """
  То же самое, только выполняет фиксированное число eval_calls
  Нужен здесь только для симметрии,
  предыдущая функция с тем же параметром выполнится столько же раз
  """

  func.eval_calls = 0

  minima = Agent(func)
  while func.eval_calls < eval_calls:
    curr = Agent(func)
    if curr.value < minima.value:
      minima = curr

    yield minima


def bee_colony(
  func: AbstractFunc,
  n_gener=100,
  shrink_cf=0.9,
  n_scouts=34,
  n_elites=7,
  n_extras=2,
  n_elite_sites=1,
  n_extra_sites=2,
  allow_intersections=True,
):
  
  """
  Алгоритм пчелиной колонии
  
  Источники:
  The Bees Algorithm - A Novel Tool for Complex Optimisation Problems
  Handbook of Swarm Intelligence
  https://fmin.xyz/docs/methods/zom/bee_algorithm.html
  https://jenyay.net/Programming/Bees
  
  n_gener - кол-во поколений
  shrink_cf - коэф. сжатия области поиска
  n_scouts - кол-во пчел разведчиков в каждом поколении
  n_elites - кол-во элитных пчел на каждый элитный участок
  n_extras - кол-во выбранных пчел на каждый выбранный участок
    (в источниках используется термин выбранный/selected, но я заменил на extra)
  n_elite_sites - кол-во элитных участков
  n_extra_sites - кол-во выбранных участков
  allow_intersections - разрешить ли участкам пересекаться
  """

  def select_sites(swarm, n, allow_intersections=True, env_size=None):
    """
    Выбирает участки в зависимости от того, могут ли они пересекаться
    Подробнее об этом https://jenyay.net/Programming/Bees
    Мол если запретить пересечения, риск застревания в локальном минимуме уменьшается
    Сейчас работает некорректно - что делать, если участки так часто пересекаются, что
    из них всех не наберется нужное количество для следующей популяции?
    """
    
    if allow_intersections:
      return swarm[:n]
    else:
      i = 1
      selected = [swarm[0]]
      while len(selected) < n and i < len(swarm):
        for bee in selected:
          if swarm[i].l_inf_dist(bee.coord) < env_size:
            break
        else:
          selected.append(swarm[i])
        i += 1


  n_total = n_scouts + n_elite_sites * n_elites + n_extra_sites * n_extras
  swarm = [Agent(func) for _ in range(n_total)]
  swarm.sort(key=lambda bee: bee.value)

  minima = cp(swarm[0])
  neigh_diam = 2 * func.domain_size() / (n_elite_sites + n_extra_sites)

  for gener in range(n_gener):
    bee = n_elite_sites + n_extra_sites

    for guide in range(n_elite_sites):
      for _ in range(n_elites):
        swarm[bee].goto(func, func.neigh_random(swarm[guide].coord, neigh_diam))
        bee += 1
      swarm[guide].goto(func, func.domain_random())

    for guide in range(n_elite_sites, n_elite_sites + n_extra_sites):
      for _ in range(n_extras):
        swarm[bee].goto(func, func.neigh_random(swarm[guide].coord, neigh_diam))
        bee += 1
      swarm[guide].goto(func, func.domain_random())

    while bee < n_total:
      swarm[bee].goto(func, func.domain_random())
      bee += 1

    swarm.sort(key=lambda bee: bee.value)
    if swarm[0].value < minima.value:
      minima = cp(swarm[0])
    neigh_diam *= shrink_cf

  return minima


def bee_colony_bench(
  func: AbstractFunc,
  eval_calls=100,
  shrink_cf=0.9,
  n_scouts=34,
  n_elites=7,
  n_extras=2,
  n_elite_sites=1,
  n_extra_sites=2,
):
  
  """
  То же самое, но заканчивает выполнение при достижении заданного количества eval_calls
  Используется функцией benchmark
  Функция-генератор, возвращает лучший найденный максимум после каждой популяции
  """

  func.eval_calls = 0

  n_total = n_scouts + n_elite_sites * n_elites + n_extra_sites * n_extras
  swarm = [Agent(func) for _ in range(n_total)]
  swarm.sort(key=lambda bee: bee.value)

  minima = cp(swarm[0])
  neigh_diam = 2 * func.domain_size() / (n_elite_sites + n_extra_sites)

  while func.eval_calls < eval_calls:
    bee = n_elite_sites + n_extra_sites

    for guide in range(n_elite_sites):
      for _ in range(n_elites):
        swarm[bee].goto(func, func.neigh_random(swarm[guide].coord, neigh_diam))
        bee += 1
      swarm[guide].goto(func, func.domain_random())

    for guide in range(n_elite_sites, n_elite_sites + n_extra_sites):
      for _ in range(n_extras):
        swarm[bee].goto(func, func.neigh_random(swarm[guide].coord, neigh_diam))
        bee += 1
      swarm[guide].goto(func, func.domain_random())

    while bee < n_total:
      swarm[bee].goto(func, func.domain_random())
      bee += 1

    swarm.sort(key=lambda bee: bee.value)
    if swarm[0].value < minima.value:
      minima = cp(swarm[0])
    neigh_diam *= shrink_cf

    yield minima


def particle_swarm(
  func: AbstractFunc,
  n_gener=100,
  n_particles=50,
  inertia_cf=0.5,
  cognitive_cf=0.1,
  social_cf=0.1,
) -> Agent:
  
  """
  Алгоритм роя частиц
  
  Источники:
  Анализ и оптимизация в задачах дизайна устройств невидимости материальных тел
  Handbook of Swarm Intelligence
  https://www.mql5.com/ru/articles/11386
  https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization
  
  n_gener - кол-во поколений
  n_particles - кол-во частиц в поколении
  inertia_cf - коэф. сохранения скорости из предыдущего поколения
  cognitive_cf - коэф. стремления к лучшей точке, найденной отдельной частицой
    во всех поколениях
  social_cf - коэф. стремления к лучшей точке, найденной всеми частицами
    в данном поколении (возможно, нужно во всех поколения смотреть)
    
  Работает очень плохо, хуже чем оптимизация пчелами, а должна быть не хуже  
  """
  
  swarm = [Agent(func) for _ in range(n_particles)]
  each_best = cp(swarm)
  glob_best = best_agent(swarm)
  minima = Agent(func)

  velocities = np.zeros(shape=(n_particles, func.dim()))
  for i, velocity in enumerate(velocities):
    velocity = swarm[i].coord - func.domain_random()

  for gener in range(n_gener):
    for i, velocity in enumerate(velocities):
      velocity = (
        inertia_cf * velocity
        + cognitive_cf * rng().random() * (each_best[i].coord - swarm[i].coord)
        + social_cf * rng().random() * (swarm[glob_best].coord - swarm[i].coord)
      )
      swarm[i].coord += velocity

    for accum, curr in zip(each_best, swarm):
      if curr.value < accum.value:
        accum = cp(curr)

    glob_best = best_agent(swarm)
    if swarm[glob_best].value < minima.value:
      minima = cp(swarm[glob_best])

  return minima


def particle_swarm_bench(
  func: AbstractFunc,
  eval_calls=100,
  n_particles=50,
  inertia_cf=0.5,
  cognitive_cf=0.1,
  social_cf=0.1,
):
  
  """
  То же самое, но заканчивает выполнение при достижении заданного количества eval_calls
  Используется функцией benchmark
  Функция-генератор, возвращает лучший найденный максимум после каждой популяции
  """

  func.eval_calls = 0

  swarm = [Agent(func) for _ in range(n_particles)]
  each_best = cp(swarm)
  glob_best = best_agent(swarm)
  minima = Agent(func)

  velocities = np.zeros(shape=(n_particles, func.dim()))
  for i, velocity in enumerate(velocities):
    velocity = swarm[i].coord - func.domain_random()

  while func.eval_calls < eval_calls:
    for i, velocity in enumerate(velocities):
      velocity = (
        inertia_cf * velocity
        + cognitive_cf * rng().random() * (each_best[i].coord - swarm[i].coord)
        + social_cf * rng().random() * (swarm[glob_best].coord - swarm[i].coord)
      )
      swarm[i].goto(func, swarm[i].coord + velocity)

    for accum, curr in zip(each_best, swarm):
      if curr.value < accum.value:
        accum = cp(curr)

    glob_best = best_agent(swarm)
    if swarm[glob_best].value < minima.value:
      minima = cp(swarm[glob_best])

    yield minima
