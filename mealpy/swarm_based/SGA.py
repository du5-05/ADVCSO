import numpy as np
from mealpy.optimizer import Optimizer

class OriginalSGA(Optimizer):
    """
    The original version of: Snow Geese Algorithm (SGA) 雪雁优化算法
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

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Create a new solution for SGA algorithm

        Returns:
            list: wrapper of solution with format [position, target, velocity]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        velocity = np.zeros(self.problem.n_dims)
        return [position, target, velocity]

    def brownian_motion(self, dim):
        """
        Generate Brownian motion

        Args:
            dim: dimension of the problem

        Returns:
            numpy.array: Brownian motion vector
        """
        T = 1
        r = T / dim
        dw = np.sqrt(r) * np.random.randn(dim)
        return np.cumsum(dw)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate coefficient
        coe = (4 * (epoch / self.epoch)) / np.exp(4 * (epoch / self.epoch))
        fi = np.random.random() * 2 * np.pi

        pop_new = []
        for idx in range(0, self.pop_size):
            # Update velocity
            acc = ((self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) -
                   1.29 * self.pop[idx][self.ID_VEL] ** 2 * np.sin(fi)) * 1e-2
            vel_new = coe * self.pop[idx][self.ID_VEL] + acc

            # Calculate center position
            weights = np.array([x[self.ID_TAR][self.ID_FIT] for x in self.pop])
            weights = weights / np.sum(weights)
            xc = np.sum([self.pop[i][self.ID_POS] * weights[i] for i in range(self.pop_size)], axis=0)

            # Update position
            pos_new = self.pop[idx][self.ID_POS] + vel_new

            if fi < np.pi:
                if idx <= self.pop_size / 5:  # First 20%
                    pos_new = pos_new + np.random.uniform(-2, 2) * (self.g_best[self.ID_POS] - pos_new) + vel_new
                elif idx < 4 * self.pop_size / 5:  # Middle 60%
                    pos_new = (pos_new + np.random.uniform(-2, 2) * (self.g_best[self.ID_POS] - pos_new) +
                               np.random.uniform(-1.5, 1.5) * (xc - pos_new) -
                               np.random.uniform(-1, 1) * (self.pop[-1][self.ID_POS] + pos_new) + vel_new)
                else:  # Last 20%
                    pos_new = (pos_new + np.random.uniform(-2, 2) * (self.g_best[self.ID_POS] - pos_new) +
                               np.random.uniform(-1.5, 1.5) * (xc - pos_new) + vel_new)
            else:
                if np.random.random() > 0.5:
                    pos_new = pos_new + (pos_new - self.g_best[self.ID_POS]) * np.random.random()
                else:
                    pos_new = self.g_best[self.ID_POS] + (
                                pos_new - self.g_best[self.ID_POS]) * np.random.random() * self.brownian_motion(
                        self.problem.n_dims)

            # Boundary handling
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

            # Create new solution
            agent = [pos_new, None, vel_new]
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent[self.ID_TAR] = self.get_target_wrapper(pos_new)

        # Update population
        self.pop = self.update_target_wrapper_population(pop_new)