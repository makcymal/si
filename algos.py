from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from func import Func, Clamping
from copy import deepcopy as cp
import typing as tp

rng = np.random.default_rng(424)


class Agent:

    __slots__ = ("point", "value")

    # point: ndarray
    # value: float

    def __init__(self, func: Func, point=None):
        if not point:
            point = func.random_point()
            self.value = func(point)
        else:
            self.point = point
            self.value = func(point)

    def move(
        self, func: Func, rhs: ndarray, rel=True, clamping=Clamping.STOP_ON_BOUNDARY
    ) -> bool:
        assert isinstance(rhs, ndarray) and rhs.shape == self.point.shape

        if not rel:
            rhs -= self.point
        self.point, clamped = func.move(self.point, rhs, clamping)
        self.value = func(self.point)
        return clamped

    def __eq__(self, other: tp.Self) -> bool:
        return self.point == other.point


@dataclass(slots=True, frozen=True)
class RandomSearch:
    
    def optimize(self, func: Func, eval_calls_lim: int) -> dict:
        func.eval_calls = 0
        evolution = {}

        minima = Agent(func)
        while func.eval_calls < eval_calls_lim:
            agent = Agent(func)
            if agent.value < minima.value:
                minima = agent
                evolution[func.eval_calls] = minima.value

        return evolution


class Particle(Agent):

    __slots__ = ("best_point", "best_value", "velocity")

    def __init__(self, func: Func, point=None):
        super().__init__(func, point)
        self.best_point = cp(self.point)
        self.best_value = self.value
        self.velocity = func.random_point() - self.point

    def update_best(self):
        if self.value < self.best_value:
            self.best_point = cp(self.point)
            self.best_value = self.value

    def move(
        self,
        func: Func,
        rhs: ndarray = None,
        rel=True,
        clamping=Clamping.STOP_ON_BOUNDARY,
    ) -> bool:

        if rhs is None:
            self.point, clamped = func.move(self.point, self.velocity)
            self.value = func(self.point)
            self.update_best()
            return clamped

        assert isinstance(rhs, ndarray) and rhs.shape == self.point.shape

        if not rel:
            rhs -= self.point

        self.point, clamped = func.move(self.point, rhs, clamping)
        self.value = func(self.point)
        self.update_best()
        return clamped


@dataclass(slots=True, frozen=True)
class ParticleSwarm:

    n_particles: int
    inertia_cf: float  # 0.721
    cognitive_cf: float  # 1.193
    social_cf: float  # 1.193

    def optimize(self, func: Func, eval_calls_lim: int) -> dict:
        func.eval_calls = 0
        evolution = {}

        swarm = [Particle(func) for _ in range(self.n_particles)]
        minima = cp(min(swarm, key=lambda idx: swarm[idx].value))

        while func.eval_calls < eval_calls_lim:
            for particle in swarm:
                particle.velocity *= self.inertia_cf
                particle.velocity += (
                    self.cognitive_cf
                    * rng.random()
                    * (particle.best_point - particle.point)
                )
                particle.velocity += (
                    self.social_cf * rng.random() * (minima.point - particle.point)
                )

                particle.move(func)

                if particle.value < minima.value:
                    minima = cp(particle)
                    evolution[func.eval_calls] = minima.value

        return evolution


@dataclass(slots=True, frozen=True)
class BeeHive:

    shrink_cf: float
    n_scouts: int
    n_elites: int
    n_extras: int
    n_elite_sites: int
    n_extra_sites: int

    def optimize(self, func: Func, eval_calls_lim: int) -> dict:
        func.eval_calls = 0
        evolution = {}

        n_total = (
            self.n_scouts
            + self.n_elite_sites * self.n_elites
            + self.n_extra_sites * self.n_extras
        )
        swarm = [Agent(func) for _ in range(n_total)]
        swarm.sort(key=lambda agent: agent.value)

        minima = cp(swarm[0])
        nearto_dist = 2 * func.diameter / (self.n_elite_sites + self.n_extra_sites)

        while func.eval_calls < eval_calls_lim:
            bee = self.n_elite_sites + self.n_extra_sites

            for guide in range(self.n_elite_sites):
                for _ in range(self.n_elites):
                    swarm[bee].move(
                        func,
                        func.random_point(swarm[guide].point, nearto_dist),
                        rel=False,
                    )
                    bee += 1
                swarm[guide].move(func, func.random_point(), rel=False)

            for guide in range(
                self.n_elite_sites, self.n_elite_sites + self.n_extra_sites
            ):
                for _ in range(self.n_extras):
                    swarm[bee].move(
                        func,
                        func.random_point(swarm[guide].point, nearto_dist),
                        rel=False,
                    )
                    bee += 1
                swarm[guide].move(func, func.random_point(), rel=False)

            while bee < n_total:
                swarm[guide].move(func, func.random_point(), rel=False)
                bee += 1

            nearto_dist *= self.shrink_cf

            swarm.sort(key=lambda bee: bee.value)
            if swarm[0].value < minima.value:
                minima = cp(swarm[0])
                evolution[func.eval_calls] = minima.value

        return evolution


class Ant(Agent):

    __slots__ = ("velocity",)

    def __init__(self, func: Func, point: ndarray = None):
        super().__init__(func, point)
        self.velocity = None

    def move(
        self,
        func: Func,
        rhs: ndarray = None,
        rel=True,
        clamping=Clamping.STOP_ON_BOUNDARY,
    ) -> bool:

        if rhs is None and self.velocity is not None:
            self.point, clamped = func.move(self.point, self.velocity, clamping)
            self.value = func(self.point)
            self.velocity = None
            return clamped

        assert isinstance(rhs, ndarray) and rhs.shape == self.point.shape

        if not rel:
            rhs -= self.point

        self.point, clamped = func.move(self.point, self.velocity, clamping)
        self.value = func(self.point)
        self.velocity = None
        return clamped


@dataclass(slots=True, frozen=True)
class AntHill:

    n_ants: int
    pher_cf: float
    dist_cf: float
    skip_cf: float

    def optimize(self, func: Func, eval_calls_lim: int) -> dict:
        func.eval_calls = 0
        evolution = {}

        hill = [Ant(func) for _ in range(self.n_ants)]
        minima = cp(min(hill, key=lambda ant: ant.value))

        for ant in hill:
            prob = np.zeros(shape=self.n_ants)
            prob_sum = 0
            
            for i, neigh in enumerate(hill):
                if ant == neigh:
                    continue

                dist = sum(d**2 for d in neigh.point - ant.point) ** 0.5
                prob[i] = neigh.value ** self.pher_cf * (1 / dist) ** self.dist_cf
                prob_sum += prob[i]
            
            prob /= prob_sum
            neigh = np.random.choice(hill, p=prob)
            ant.velocity = rng.beta(2, 5) * self.skip_cf * (neigh.point - ant.point)
            
        for ant in hill:
            ant.move(func)
            if (ant.value < minima.value):
                minima = cp(ant)
                evolution[func.eval_calls] = minima.value
                
        return evolution                        
