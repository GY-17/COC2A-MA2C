import copy
import os
import pickle
import random
from dataclasses import dataclass

import numpy as np


def _optimizer_variables(optimizer):
    variables = getattr(optimizer, "variables", [])
    return list(variables() if callable(variables) else variables)


def capture_random_state():
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
    }


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def seed_python_and_numpy(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))


def _copy_weights(model):
    return [np.array(value, copy=True) for value in model.get_weights()]


def _copy_optimizer_state(optimizer):
    return [np.array(variable.numpy(), copy=True)
            for variable in _optimizer_variables(optimizer)]


def _restore_optimizer_state(optimizer, values):
    variables = _optimizer_variables(optimizer)
    if len(variables) != len(values):
        raise RuntimeError(
            "Optimizer variable layout changed during CMA-ES candidate evaluation: "
            f"current={len(variables)}, snapshot={len(values)}."
        )
    for variable, value in zip(variables, values):
        variable.assign(value)


@dataclass
class MACA2CTrainingSnapshot:
    actor_weights: list
    critic_weights: list
    actor_optimizer_states: list
    critic_optimizer_state: list
    random_state: dict

    @classmethod
    def capture(cls, trainers):
        if not trainers:
            raise ValueError("At least one MACA2C trainer is required.")
        return cls(
            actor_weights=[_copy_weights(trainer.policy) for trainer in trainers],
            critic_weights=_copy_weights(trainers[0].critic),
            actor_optimizer_states=[
                _copy_optimizer_state(trainer.actor_opt) for trainer in trainers
            ],
            critic_optimizer_state=_copy_optimizer_state(trainers[0].critic_opt),
            random_state=capture_random_state(),
        )

    def restore(self, trainers, restore_rng=True):
        if len(trainers) != len(self.actor_weights):
            raise ValueError("Trainer count does not match the saved snapshot.")

        for trainer, weights, optimizer_state in zip(
                trainers, self.actor_weights, self.actor_optimizer_states):
            trainer.policy.set_weights(weights)
            _restore_optimizer_state(trainer.actor_opt, optimizer_state)
            trainer.buffer.reset()
            trainer.policy.reset_states()

        trainers[0].critic.set_weights(self.critic_weights)
        _restore_optimizer_state(
            trainers[0].critic_opt, self.critic_optimizer_state
        )
        if restore_rng:
            restore_random_state(self.random_state)


class CMAESWeightOptimizer:
    """Thin pycma ask/tell wrapper for the flattened coordination weights."""

    def __init__(self, initial_parameters, sigma, population_size, seed):
        try:
            import cma
        except ImportError as exc:
            raise ImportError(
                "CMA-ES optimization requires the 'cma' package. "
                "Install the project dependencies before enabling "
                "maca2c_cma_enabled."
            ) from exc

        initial = np.asarray(initial_parameters, dtype=np.float64).reshape(-1)
        options = {
            "popsize": int(population_size),
            "seed": int(seed),
            "verbose": -9,
        }
        self.strategy = cma.CMAEvolutionStrategy(initial, float(sigma), options)
        self.population_size = int(population_size)
        self.generation = 0
        self.last_generation_best = np.array(initial, copy=True)

    def ask(self):
        return [np.asarray(candidate, dtype=np.float64)
                for candidate in self.strategy.ask(self.population_size)]

    def tell(self, candidates, fitnesses):
        if len(candidates) != len(fitnesses):
            raise ValueError("Each CMA-ES candidate must have one fitness value.")
        self.strategy.tell(candidates, [float(value) for value in fitnesses])
        best_index = int(np.argmin(fitnesses))
        self.last_generation_best = np.asarray(
            candidates[best_index], dtype=np.float64
        ).copy()
        self.generation += 1
        return self.last_generation_best.copy()

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as file_obj:
            pickle.dump(self, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
