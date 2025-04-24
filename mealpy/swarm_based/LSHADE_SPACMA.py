import numpy as np
from scipy.stats import cauchy
from mealpy.optimizer import Optimizer


class LSHADE_SPACMA(Optimizer):
    """
    The original version of: Success-history based Parameter Adaptation for Differential Evolution with SPACMA
    """

    def __init__(self, epoch=10000, pop_size=100, memory_size=5, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.memory_size = self.validator.check_int("memory_size", memory_size, [2, 100])

        # Default parameters for L-SHADE
        self.p_best_rate = 0.11
        self.arc_rate = 2.0
        self.memory = None

        # Default parameters for SPACMA
        self.G = None
        self.k = None
        self.c1 = None
        self.c2 = None
        self.w_min = 0.4
        self.w_max = 0.9

        # Initialize indices
        self.ID_POS = 0  # Index of position/solution
        self.ID_TAR = 1  # Index of target/fitness value
        self.ID_FIT = 0  # Index of fitness value

        self.set_parameters(["epoch", "pop_size", "memory_size"])
        self.sort_flag = False
        self.archive = []

    def initialization(self):
        """Override the initialization method"""
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)

        # Sort population and get best and worst solutions
        pop_sorted = sorted(self.pop, key=lambda x: x[self.ID_TAR][self.ID_FIT])
        self.g_best = pop_sorted[0].copy()
        self.g_worst = pop_sorted[-1].copy()

        # Initialize history
        self.history.list_global_best.append(self.g_best.copy())
        self.history.list_global_worst.append(self.g_worst.copy())
        self.history.list_current_best.append(self.g_best.copy())
        self.history.list_current_worst.append(self.g_worst.copy())
        self.history.list_epoch_time.append(0)
        self.history.list_diversity.append(0)

    def after_initialization(self):
        """
        Initialize the algorithm's specific parameters after the problem is set
        """
        # Initialize memory for L-SHADE
        self.memory = {
            'mu_f': np.ones(self.memory_size) * 0.5,
            'mu_cr': np.ones(self.memory_size) * 0.5,
            'mu_pos': 0
        }

        # Initialize parameters for SPACMA
        self.G = self.epoch
        self.k = 0
        self.c1 = 2 / ((self.problem.n_dims + 1.3) ** 2 + self.pop_size)
        self.c2 = 2 / ((self.problem.n_dims + 1.3) ** 2 + self.pop_size)

    def is_better(self, target1, target2):
        """
        Compare which target is better
        """
        if self.problem.minmax == "min":
            return target1[self.ID_FIT] < target2[self.ID_FIT]
        return target1[self.ID_FIT] > target2[self.ID_FIT]

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Create a new solution for LSHADE_SPACMA algorithm
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        return [position, target]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        """
        # Update generation counter
        self.k = epoch

        # Calculate weights
        w = self.w_max - (self.w_max - self.w_min) * (self.k / self.G)

        # Sort population and get the best solution
        pop_sorted = sorted(self.pop, key=lambda x: x[self.ID_TAR][self.ID_FIT])
        pop_best = pop_sorted[0]
        current_worst = pop_sorted[-1]

        # Initialize archive
        archive_size = int(round(self.arc_rate * self.pop_size))

        pop_new = []
        success_f = []
        success_cr = []
        success_fits = []

        for idx in range(0, self.pop_size):
            # Select random memory index
            mem_idx = np.random.randint(0, self.memory_size)

            # Generate CR and F values
            cr = np.random.normal(self.memory['mu_cr'][mem_idx], 0.1)
            cr = np.clip(cr, 0, 1)

            f = cauchy.rvs(loc=self.memory['mu_f'][mem_idx], scale=0.1, size=1)[0]
            while f <= 0:
                f = cauchy.rvs(loc=self.memory['mu_f'][mem_idx], scale=0.1, size=1)[0]
            f = min(f, 1.0)

            # Generate new position using SPACMA
            r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
            while r1 == idx or r2 == idx:
                r1, r2 = np.random.choice(self.pop_size, 2, replace=False)

            # SPACMA position update
            pos_new = (self.pop[idx][self.ID_POS] +
                       w * (self.pop[r1][self.ID_POS] - self.pop[r2][self.ID_POS]) +
                       self.c1 * np.random.random() * (pop_best[self.ID_POS] - self.pop[idx][self.ID_POS]) +
                       self.c2 * np.random.random() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]))

            # Boundary handling
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

            # Create new solution and evaluate
            target_new = self.get_target_wrapper(pos_new)
            agent = [pos_new, target_new]

            # Selection
            if self.compare_agent(agent, self.pop[idx]):
                pop_new.append(agent)
                success_f.append(f)
                success_cr.append(cr)
                success_fits.append(target_new[self.ID_FIT])

                # Update archive
                if len(self.archive) < archive_size:
                    self.archive.append(self.pop[idx].copy())
                else:
                    if np.random.random() < self.arc_rate:
                        idx_replaced = np.random.randint(0, archive_size)
                        self.archive[idx_replaced] = self.pop[idx].copy()

                # Update g_best if needed
                if self.compare_agent(agent, self.g_best):
                    self.g_best = agent.copy()
                    self.history.list_global_best.append(self.g_best.copy())
            else:
                pop_new.append(self.pop[idx])

        # Update g_worst if needed
        if self.problem.minmax == "min":
            if current_worst[self.ID_TAR][self.ID_FIT] > self.g_worst[self.ID_TAR][self.ID_FIT]:
                self.g_worst = current_worst.copy()
                self.history.list_global_worst.append(self.g_worst.copy())
        else:
            if current_worst[self.ID_TAR][self.ID_FIT] < self.g_worst[self.ID_TAR][self.ID_FIT]:
                self.g_worst = current_worst.copy()
                self.history.list_global_worst.append(self.g_worst.copy())

        # Update memory
        if len(success_f) > 0:
            weights = np.array(success_fits)
            weights = weights / np.sum(weights)

            self.memory['mu_f'][self.memory['mu_pos']] = np.sum(np.array(success_f) * weights)
            self.memory['mu_cr'][self.memory['mu_pos']] = np.sum(np.array(success_cr) * weights)
            self.memory['mu_pos'] = (self.memory['mu_pos'] + 1) % self.memory_size

        # Update population
        self.pop = pop_new

        # Update history
        self.history.list_current_best.append(pop_best.copy())
        self.history.list_current_worst.append(current_worst.copy())