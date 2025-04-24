import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSDO(Optimizer):
    """
    The original version of: Sled Dog-inspired Optimizer (SDO) 雪橇犬优化算法
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
        Create a new solution for SDO algorithm

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

        # Sort population and get best solution
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
        w = 0.9 - (0.9 - 0.4) * epoch / self.epoch  # Inertia weight
        c1 = 2 * np.random.random()  # Cognitive coefficient
        c2 = 2 * np.random.random()  # Social coefficient

        pop_new = []
        for idx in range(0, self.pop_size):
            # Get current position, velocity
            pos_current = self.pop[idx][self.ID_POS].copy()
            vel_current = self.pop[idx][self.ID_VEL].copy()

            # Select random sled dogs
            r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
            while r1 == idx or r2 == idx:
                r1, r2 = np.random.choice(self.pop_size, 2, replace=False)

            # Update velocity
            vel_new = (w * vel_current +
                       c1 * np.random.random() * (self.g_best[self.ID_POS] - pos_current) +
                       c2 * np.random.random() * (self.pop[r1][self.ID_POS] - self.pop[r2][self.ID_POS]))

            # Velocity clamping
            vel_new = np.clip(vel_new, -1, 1)

            # Update position
            pos_new = pos_current + vel_new

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