import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import numpy as np
from func import Agent
from copy import deepcopy as cp

rng = np.random.default_rng(424)


def best_agent(swarm, minimize=True) -> int:
    cmp = (lambda a, b: a < b) if minimize else (lambda a, b: a > b)
    best_idx = 0
    for curr_idx, agent in enumerate(swarm):
        if cmp(agent.value, swarm[best_idx].value):
            best_idx = curr_idx
    return best_idx


def particle_swarm_bench(
    func,
    eval_calls,
    n_particles=50,
    inertia_cf=0.721,
    cognitive_cf=1.193,
    social_cf=1.193,
):
    func.eval_calls = 0

    swarm = [Agent(func) for _ in range(n_particles)]
    glob_best = swarm[best_agent(swarm)]
    each_best = cp(swarm)
    
    minima = Agent(func)

    velocities = np.zeros(shape=(n_particles, 2))
    for i in range(n_particles):
        velocities[i] = (func.domain_random() - swarm[i].point) / 2

    while func.eval_calls < eval_calls:
        for i in range(n_particles):
            velocities[i] = (
                inertia_cf * velocities[i]
                + cognitive_cf * rng.random() * (each_best[i].point - swarm[i].point)
                + social_cf * rng.random() * (glob_best.point - swarm[i].point)
            )
            if func.within_domain(swarm[i].point + velocities[i]):
                swarm[i].move(func, velocities[i], rel=True)
            else:
                func.eval_calls += 1
                velocities[i] = 0

        glob_best = cp(swarm[0])
        for i in range(n_particles):
            if swarm[i].value < each_best[i].value:
                each_best[i] = cp(swarm[i])
                
            if swarm[i].value < glob_best.value:
                glob_best = cp(swarm[i])
                if glob_best.value < minima.value:
                    minima = cp(glob_best)
                
        yield minima


def bee_colony_bench(
    func,
    eval_calls=100,
    shrink_cf=0.9,
    n_scouts=34,
    n_elites=7,
    n_extras=2,
    n_elite_sites=1,
    n_extra_sites=2,
):
    func.eval_calls = 0

    n_total = n_scouts + n_elite_sites * n_elites + n_extra_sites * n_extras
    swarm = [Agent(func) for _ in range(n_total)]
    swarm.sort(key=lambda bee: bee.value)

    minima = cp(swarm[0])
    neigh_diam = 2 * func.domain / (n_elite_sites + n_extra_sites)

    while func.eval_calls < eval_calls:
        bee = n_elite_sites + n_extra_sites

        for guide in range(n_elite_sites):
            for _ in range(n_elites):
                swarm[bee].move(func, func.neigh_random(swarm[guide].point, neigh_diam))
                bee += 1
            swarm[guide].move(func, func.domain_random())

        for guide in range(n_elite_sites, n_elite_sites + n_extra_sites):
            for _ in range(n_extras):
                swarm[bee].move(func, func.neigh_random(swarm[guide].point, neigh_diam))
                bee += 1
            swarm[guide].move(func, func.domain_random())

        while bee < n_total:
            swarm[bee].move(func, func.domain_random())
            bee += 1

        swarm.sort(key=lambda bee: bee.value)
        if swarm[0].value < minima.value:
            minima = cp(swarm[0])
        neigh_diam *= shrink_cf

        yield minima
