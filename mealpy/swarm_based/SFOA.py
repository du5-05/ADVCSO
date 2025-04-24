import numpy as np
from math import gamma
from mealpy.optimizer import Optimizer


class OriginalSFOA(Optimizer):
    """
    The original version of: Superb Fairy-wren Optimization Algorithm (SFOA) 壮丽细尾鹩莺优化算法
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Create a new solution for SFOA algorithm

        Returns:
            list: wrapper of solution with format [position, target]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        return [position, target]

    def levy(self, n, m, beta=1.5):
        """
        Implementation of Levy flight based on levy.m

        Args:
            n: Number of steps
            m: Number of dimensions
            beta: Power law index (1 < beta < 2)

        Returns:
            numpy.array: n levy steps in m dimensions
        """
        # Calculate sigma_u using gamma function
        num = float(gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0))
        den = float(gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
        sigma_u = (num / den) ** (1.0 / beta)

        # Generate random numbers from normal distribution
        u = np.random.normal(0, sigma_u, (n, m))
        v = np.random.normal(0, 1, (n, m))

        # Calculate levy steps
        z = u / (np.abs(v) ** (1.0 / beta))
        return z

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        c = 0.8
        r1 = np.random.random()
        r2 = np.random.random()
        w = (np.pi / 2) * (epoch / self.epoch)
        k = 0.2 * np.sin(np.pi / 2 - w)
        # Generate levy flight steps for all solutions at once
        levy_steps = self.levy(self.pop_size, self.problem.n_dims)
        m = epoch / self.epoch * 2
        p = np.sin(self.problem.ub - self.problem.lb) * 2 + (self.problem.ub - self.problem.lb) * m

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS].copy()

            T = 0.5
            if np.random.random() < T:
                pos_new = self.generate_position(self.problem.lb, self.problem.ub)
            else:
                s = r1 * 20 + r2 * 20
                if s > 20:
                    pos_new = self.g_best[self.ID_POS] + pos_new * levy_steps[idx] * k
                else:
                    pos_new = self.g_best[self.ID_POS] * c + (self.g_best[self.ID_POS] - pos_new) * p

            # Boundary handling
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

            # Create new solution
            agent = [pos_new, None]
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent[self.ID_TAR] = self.get_target_wrapper(pos_new)

        # Update population
        self.pop = self.update_target_wrapper_population(pop_new)