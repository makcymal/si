from collections.abc import Iterable
from abc import ABC, abstractmethod


class AbstractFunc(ABC):
  @abstractmethod
  def __init__(self):
    self.eval_calls = 0
    self.rand_calls = 0

  @abstractmethod
  def __call__(self):
    self.eval_calls += 1

  @abstractmethod
  def domain_random(self):
    self.rand_calls += 1

  @abstractmethod
  def neigh_random(self):
    self.rand_calls += 1

  @abstractmethod
  def domain_size(self):
    pass

  @abstractmethod
  def dim(self):
    pass
