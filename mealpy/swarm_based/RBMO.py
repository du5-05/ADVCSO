import numpy as np
from mealpy.optimizer import Optimizer


class OriginalRBMO(Optimizer):
    """
    The original version of: Red-billed Blue Magpie Optimizer (RBMO)
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

        # Define indices for solution components
        self.ID_POS = 0  # Index of position/solution
        self.ID_TAR = 1  # Index of target/fitness value
        self.ID_VEL = 2  # Index of velocity
        self.ID_FIT = 0  # Index of fitness value in target

    def get_best_solution(self, pop):
        """
        Get the best solution from the population

        Args:
            pop: The population of solutions

        Returns:
            The best solution found
        """
        sorted_pop = sorted(pop, key=lambda x: x[self.ID_TAR][self.ID_FIT])
        if self.problem.minmax == "min":
            return sorted_pop[0]
        else:
            return sorted_pop[-1]

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Create a new solution for RBMO algorithm

        Returns:
            list: wrapper of solution with format [position, target, velocity]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        velocity = np.zeros(self.problem.n_dims)
        return [position, target, velocity]

    def initialization(self):
        """Override the initialization method"""
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)

        # Sort population and get best and worst solutions
        self.g_best = self.get_best_solution(self.pop)

        # Initialize history
        self.history.list_global_best.append(self.g_best.copy())

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate adaptive parameters
        a = 2 * (1 - epoch / self.epoch)  # Linearly decreased from 2 to 0

        pop_new = []
        for idx in range(0, self.pop_size):
            # Select random solution
            rand_idx = np.random.randint(0, self.pop_size)
            while rand_idx == idx:
                rand_idx = np.random.randint(0, self.pop_size)

            # Calculate velocity
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()

            vel_new = (self.pop[idx][self.ID_VEL] +
                       r1 * a * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) +
                       r2 * a * (self.pop[rand_idx][self.ID_POS] - self.pop[idx][self.ID_POS]))

            # Update position
            pos_new = self.pop[idx][self.ID_POS] + vel_new

            # Local search
            if r3 < 0.5:
                pos_new = self.g_best[self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * \
                          (self.pop[idx][self.ID_POS] - self.pop[rand_idx][self.ID_POS])

            # Boundary handling
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

            # Create new solution
            agent = [pos_new, None, vel_new]
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent[self.ID_TAR] = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(agent, self.pop[idx])

        # Update population
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        # Update global best solution
        current_best = self.get_best_solution(self.pop)
        if self.compare_agent(current_best, self.g_best):
            self.g_best = current_best.copy()
            self.history.list_global_best.append(self.g_best.copy())