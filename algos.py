from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from func import Func, Clamping
from copy import deepcopy as cp
import typing as tp

rng = np.random.default_rng(424)


# << Agent >>


class Agent:

    __slots__ = ("point", "value")

    def __init__(self, func: Func, point=None):
        if not point:
            self.point = func.random_point()
            self.value = func(self.point)
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


# == Agent ==


# << RandomSearch >>


@dataclass(slots=True, frozen=True)
class RandomSearch:

    name: str

    def optimize(self, func: Func, eval_calls_lim: int) -> dict:
        func.eval_calls = 0
        evolution = {}
        minima = Agent(func)
        evolution[func.eval_calls] = minima.value

        while func.eval_calls < eval_calls_lim:
            agent = Agent(func)
            if agent.value < minima.value:
                minima = agent
                evolution[func.eval_calls] = minima.value

        return evolution


# == RandomSearch ==


# << ParticleSwarm >>


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

    name: str
    n_particles: int
    inertia_cf: float  # 0.721
    cognitive_cf: float  # 1.193
    social_cf: float  # 1.193

    def optimize(self, func: Func, eval_calls_lim: int) -> dict:
        func.eval_calls = 0
        evolution = {}

        swarm = [Particle(func) for _ in range(self.n_particles)]
        minima = cp(min(swarm, key=lambda particle: particle.value))
        evolution[func.eval_calls] = minima.value

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

    def optimize_viz(self, func: Func, eval_calls_lim: int):
        func.eval_calls = 0

        swarm = [Particle(func) for _ in range(self.n_particles)]
        minima = cp(min(swarm, key=lambda particle: particle.value))

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

            x = [particle.point[0] for particle in swarm]
            y = [particle.point[1] for particle in swarm]
            yield x, y


# == ParticleSwarm ==


# << TurboBees >>


@dataclass(slots=True, frozen=True)
class TurboBees:

    name: str
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
        evolution[func.eval_calls] = minima.value

        nearto_dist = func.diameter() / (self.n_elite_sites + self.n_extra_sites)

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

            for guide in range(i1 := self.n_elite_sites, i1 + self.n_extra_sites):
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


# == TurboBees ==


# << BeeHive >>


class BeeSwarm:

    __slots__ = (
        "bees",
        "best",
        "center",
    )

    def __init__(self, func: Func, n_bees: int):
        self.bees = [Agent(func) for _ in range(n_bees)]
        self.best = min(range(n_bees), key=lambda idx: self.bees[idx].value)
        self.center = self.bees[self.best]

    def move_to_center(self, func: Func, radius: float):
        for idx, bee in enumerate(self.bees):
            if idx == self.best:
                continue
            bee.move(
                func,
                func.random_point(nearto=self.center.point, dist=radius, cubic=False),
            )

    def find_best(self):
        self.best = min(range(len(self.bees)), key=lambda idx: self.bees[idx].value)
        self.center = self.bees[self.best]

    def go_scout(self, func: Func):
        for bee in self.bees:
            bee.move(func, func.random_point(), rel=False)


@dataclass(slots=True, frozen=True)
class BeesHive:

    name: str
    n_swarms: int
    n_bees_in_swarm: int
    swarm_radius_frac: float

    def optimize(self, func: Func, eval_calls_lim: int = 1000):
        func.eval_calls = 0
        evolution = {}

        swarms = [BeeSwarm(func, self.n_bees_in_swarm) for _ in range(self.n_swarms)]
        swarms.sort(key=lambda swarm: swarm.bees[swarm.best].value)
        minima = cp(swarms[0].bees[swarms[0].best])
        evolution[func.eval_calls] = minima.value

        radius = func.diameter() / 2 * self.swarm_radius_frac

        while func.eval_calls < eval_calls_lim:
            for i in range(self.n_swarms - 1):
                for j in range(i):
                    dist = (
                        sum(
                            (c1 - c2) ** 2
                            for c1, c2 in zip(
                                swarms[i].center.point, swarms[j].center.point
                            )
                        )
                        ** 0.5
                    )
                    if dist == 0:
                        swarms[i].center.move(func, func.random_point())
                    elif dist < 2 * radius:
                        alpha = 2 * radius / dist - 1
                        swarms[i].center.move(
                            func,
                            (swarms[i].center.point - swarms[j].center.point) * alpha,
                        )
                swarms[i].move_to_center(func, radius)

            swarms[-1].go_scout(func)

            for swarm in swarms:
                swarm.find_best()

            swarms.sort(key=lambda swarm: swarm.bees[swarm.best].value)
            if swarms[0].bees[swarms[0].best].value < minima.value:
                minima = cp(swarms[0].bees[swarms[0].best])
                evolution[func.eval_calls] = minima.value

        return evolution

    def optimize_viz(self, func: Func, eval_calls_lim: int = 1000):
        func.eval_calls = 0

        swarms = [BeeSwarm(func, self.n_bees_in_swarm) for _ in range(self.n_swarms)]
        swarms.sort(key=lambda swarm: swarm.bees[swarm.best].value)
        minima = cp(swarms[0].bees[swarms[0].best])

        radius = func.diameter() / 2 * self.swarm_radius_frac

        while func.eval_calls < eval_calls_lim:
            for i in range(self.n_swarms - 1):
                for j in range(i):
                    dist = (
                        sum(
                            (c1 - c2) ** 2
                            for c1, c2 in zip(
                                swarms[i].center.point, swarms[j].center.point
                            )
                        )
                        ** 0.5
                    )
                    if dist == 0:
                        swarms[i].center.move(func, func.random_point())
                    elif dist < 2 * radius:
                        alpha = 2 * radius / dist - 1
                        swarms[i].center.move(
                            func,
                            (swarms[i].center.point - swarms[j].center.point) * alpha,
                        )
                swarms[i].move_to_center(func, radius)

            swarms[-1].go_scout(func)

            for swarm in swarms:
                swarm.find_best()

            swarms.sort(key=lambda swarm: swarm.bees[swarm.best].value)
            if swarms[0].bees[swarms[0].best].value < minima.value:
                minima = cp(swarms[0].bees[swarms[0].best])

            x = sum([bee.point[0] for bee in swarm.bees] for swarm in swarms)
            y = sum([bee.point[1] for bee in swarm.bees] for swarm in swarms)
            yield x, y


# == BeeHive ==


# << AntHill >>


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

    name: str
    n_ants: int
    pher_cf: float
    dist_cf: float
    skip_cf: float

    def optimize(self, func: Func, eval_calls_lim: int) -> dict:
        func.eval_calls = 0
        evolution = {}

        hill = [Ant(func) for _ in range(self.n_ants)]
        minima = cp(min(hill, key=lambda ant: ant.value))
        evolution[func.eval_calls] = minima.value

        while func.eval_calls < eval_calls_lim:
            for ant in hill:
                prob = np.zeros(shape=self.n_ants)
                prob_sum = 0

                for i, neigh in enumerate(hill):
                    if all(ant.point == neigh.point):
                        continue

                    dist = sum(d**2 for d in neigh.point - ant.point) ** 0.5
                    prob[i] = (neigh.value - func.exact_min_value) ** self.pher_cf * (
                        1 / dist
                    ) ** self.dist_cf
                    prob_sum += prob[i]

                prob /= prob_sum
                neigh = np.random.choice(hill, p=prob)
                ant.velocity = rng.beta(2, 5) * self.skip_cf * (neigh.point - ant.point)

            for ant in hill:
                ant.move(func)
                if ant.value < minima.value:
                    minima = cp(ant)
                    evolution[func.eval_calls] = minima.value

        return evolution

    def optimize_viz(self, func: Func, eval_calls_lim: int):
        func.eval_calls = 0

        hill = [Ant(func) for _ in range(self.n_ants)]
        minima = cp(min(hill, key=lambda ant: ant.value))

        while func.eval_calls < eval_calls_lim:
            for ant in hill:
                prob = np.zeros(shape=self.n_ants)
                prob_sum = 0

                for i, neigh in enumerate(hill):
                    if all(ant.point == neigh.point):
                        continue

                    dist = sum(d**2 for d in neigh.point - ant.point) ** 0.5
                    prob[i] = (neigh.value - func.exact_min_value) ** self.pher_cf * (
                        1 / dist
                    ) ** self.dist_cf
                    prob_sum += prob[i]

                prob /= prob_sum
                neigh = np.random.choice(hill, p=prob)
                ant.velocity = rng.beta(2, 5) * self.skip_cf * (neigh.point - ant.point)

            for ant in hill:
                ant.move(func)
                if ant.value < minima.value:
                    minima = cp(ant)

            x = [ant.point[0] for ant in hill]
            y = [ant.point[1] for ant in hill]
            yield x, y


# == AntHill ==
