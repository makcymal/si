import numpy as np


class rng:
  """
  Random Number Generator - синглтон - один инстанс на все модули
  Доки numpy говорят использовать объект np.random.default_rng(),
  но создавать его везде каждый раз, передавать как аргумент было неудобно, поэтому вот
  """
  
  _instance = None
  
  def __new__(cls):
    if not cls._instance:
      cls._instance = np.random.default_rng()
    return cls._instance
  
  def reseed(seed: int):
    rng._instance = np.random.default_rng(seed)
    
