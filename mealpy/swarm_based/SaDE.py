import numpy as np
from mealpy.optimizer import Optimizer


class SaDE(Optimizer):
    """
    The original version of: Self-adaptive Differential Evolution Algorithm
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Create a new solution for SaDE algorithm

        Returns:
            list: wrapper of solution with format [position, target]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        return [position, target]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Initialize learning period and strategy probabilities
        learning_period = 50
        num_strategies = 4
        strategy_probs = np.ones(num_strategies) / num_strategies

        # Initialize success and failure memories
        success_memory = np.zeros((learning_period, num_strategies))
        failure_memory = np.zeros((learning_period, num_strategies))

        pop_new = []
        for idx in range(0, self.pop_size):
            # Select strategy based on probabilities
            strategy = np.random.choice(range(num_strategies), p=strategy_probs)

            # Generate CR and F values
            CR = np.random.normal(0.5, 0.1)
            CR = np.clip(CR, 0, 1)

            F = np.random.normal(0.5, 0.3)
            F = np.clip(F, 0, 2)

            # Select random solutions for mutation
            idxs = [idx]
            while len(idxs) < 5:
                idx_rand = np.random.randint(0, self.pop_size)
                if idx_rand not in idxs:
                    idxs.append(idx_rand)

            x_r1 = self.pop[idxs[1]][self.ID_POS]
            x_r2 = self.pop[idxs[2]][self.ID_POS]
            x_r3 = self.pop[idxs[3]][self.ID_POS]
            x_r4 = self.pop[idxs[4]][self.ID_POS]

            # Apply selected strategy
            if strategy == 0:
                # DE/rand/1
                pos_new = x_r1 + F * (x_r2 - x_r3)
            elif strategy == 1:
                # DE/rand-to-best/2
                pos_new = self.pop[idx][self.ID_POS] + F * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                          F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
            elif strategy == 2:
                # DE/rand/2
                pos_new = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - self.pop[idx][self.ID_POS])
            else:
                # DE/current-to-rand/1
                pos_new = self.pop[idx][self.ID_POS] + F * (x_r1 - self.pop[idx][self.ID_POS]) + F * (x_r2 - x_r3)

            # Crossover
            pos_new = np.where(np.random.random(self.problem.n_dims) < CR, pos_new, self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

            agent = [pos_new, None]
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent[self.ID_TAR] = self.get_target_wrapper(pos_new)

            # Update success/failure memory
            if agent[self.ID_TAR][self.ID_FIT] < self.pop[idx][self.ID_TAR][self.ID_FIT]:
                success_memory[epoch % learning_period, strategy] += 1
            else:
                failure_memory[epoch % learning_period, strategy] += 1

        # Update population
        self.pop = self.update_target_wrapper_population(pop_new)

        # Update strategy probabilities after learning period
        if epoch > 0 and epoch % learning_period == 0:
            total_success = np.sum(success_memory, axis=0)
            total_failure = np.sum(failure_memory, axis=0)
            strategy_probs = (total_success + 1e-10) / (total_success + total_failure + 2e-10)
            strategy_probs = strategy_probs / np.sum(strategy_probs)