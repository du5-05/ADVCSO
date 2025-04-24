import numpy as np
from mealpy.optimizer import Optimizer

class LSHADE_cnEpSin(Optimizer):
    """
    The original version of: LSHADE with Convergence and Sinusoidal Parameters (LSHADE-cnEpSin)
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

        # Algorithm parameters
        self.memory_size = 5
        self.p_best_rate = 0.11
        self.arc_rate = 1.4
        self.min_pop_size = 4

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Create a new solution for LSHADE-cnEpSin algorithm

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
        # Calculate current population size
        current_pop_size = round((self.min_pop_size - self.pop_size) * epoch / self.epoch + self.pop_size)

        # Initialize memory
        memory_cr = np.ones(self.memory_size) * 0.5
        memory_f = np.ones(self.memory_size) * 0.5
        memory_pos = 0

        # Sort population
        pop_sorted = sorted(self.pop, key=lambda x: x[self.ID_TAR][self.ID_FIT])

        # Initialize archive
        archive = []
        archive_size = int(self.arc_rate * current_pop_size)

        pop_new = []
        for idx in range(0, current_pop_size):
            # Generate CR and F values
            ri = np.random.randint(self.memory_size)
            cr = np.random.normal(memory_cr[ri], 0.1)
            cr = np.clip(cr, 0, 1)

            # Generate F using convergence and sinusoidal parameters
            f_sin = 0.5 * (np.sin(2 * np.pi * epoch / self.epoch) + 1)
            f_conv = (1 - epoch / self.epoch) ** 2
            f = f_sin * f_conv + np.random.normal(memory_f[ri], 0.1)
            f = np.clip(f, 0, 1)

            # Select p-best solutions
            p = max(2, int(round(current_pop_size * self.p_best_rate)))
            p_best_idx = np.random.randint(p)
            x_best = pop_sorted[p_best_idx][self.ID_POS]

            # Select random solutions for mutation
            idxs = [idx]
            while len(idxs) < 4:
                idx_rand = np.random.randint(current_pop_size)
                if idx_rand not in idxs:
                    idxs.append(idx_rand)

            x_r1 = self.pop[idxs[1]][self.ID_POS]

            # Select solution from extended population (including archive)
            if len(archive) > 0:
                pop_and_archive = self.pop + archive
                idx_r2 = np.random.randint(len(pop_and_archive))
                x_r2 = pop_and_archive[idx_r2][self.ID_POS]
            else:
                x_r2 = self.pop[idxs[2]][self.ID_POS]

            # DE/current-to-pbest/1
            pos_new = self.pop[idx][self.ID_POS] + f * (x_best - self.pop[idx][self.ID_POS]) + f * (x_r1 - x_r2)

            # Crossover
            pos_new = np.where(np.random.random(self.problem.n_dims) < cr, pos_new, self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

            # Create new solution
            agent = [pos_new, None]
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent[self.ID_TAR] = self.get_target_wrapper(pos_new)

            # Update archive
            if agent[self.ID_TAR][self.ID_FIT] < self.pop[idx][self.ID_TAR][self.ID_FIT]:
                archive.append(self.pop[idx])
                if len(archive) > archive_size:
                    archive.pop(np.random.randint(len(archive)))

        # Update population
        self.pop = self.update_target_wrapper_population(pop_new)

        # Update memories
        successful_cr = []
        successful_f = []
        delta_f = []

        for idx, agent in enumerate(pop_new):
            if agent[self.ID_TAR][self.ID_FIT] < self.pop[idx][self.ID_TAR][self.ID_FIT]:
                successful_cr.append(cr)
                successful_f.append(f)
                delta_f.append(abs(agent[self.ID_TAR][self.ID_FIT] - self.pop[idx][self.ID_TAR][self.ID_FIT]))

        if len(successful_cr) > 0:
            weights = delta_f / np.sum(delta_f)
            memory_cr[memory_pos] = np.sum(weights * np.array(successful_cr))
            memory_f[memory_pos] = np.sum(weights * np.array(successful_f) ** 2) / np.sum(
                weights * np.array(successful_f))
            memory_pos = (memory_pos + 1) % self.memory_size