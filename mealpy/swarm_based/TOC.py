import numpy as np
from mealpy.optimizer import Optimizer


class OriginalTOC(Optimizer):
    """
    The original version of: Tornado Optimizer with Coriolis Force (TOC)
    """

    def __init__(self, epoch=10000, pop_size=100, nto=5, nt=2, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.nto = self.validator.check_int("nto", nto, [2, 20])
        self.nt = self.validator.check_int("nt", nt, [1, nto - 1])

        self.set_parameters(["epoch", "pop_size", "nto", "nt"])
        self.sort_flag = False

        # Fixed parameters from the paper
        self.b_r = 100000
        self.omega = 0.7292115E-04

    def create_solution(self, lb=None, ub=None):
        """
        Create a solution for TOC algorithm
        """
        position = self.generate_position(lb, ub)
        position = self.amend_position(position, lb, ub)
        target = self.get_target_wrapper(position)
        return [position, target]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        """
        # Calculate adaptive parameters
        nu = (0.1 * np.exp(-0.1 * (epoch / self.epoch) ** 0.1)) ** 16
        ay = (self.epoch - (epoch ** 2 / self.epoch)) / self.epoch

        # Sort population
        pop = sorted(self.pop, key=lambda x: x[self.ID_TAR][self.ID_FIT])

        # Split population into Tornadoes, Thunderstorms and Windstorms
        To = self.nto - self.nt  # Number of tornadoes
        nw = self.pop_size - self.nto  # Number of windstorms

        tornado_pop = pop[:To]
        thunderstorm_pop = pop[1:self.nto]
        windstorm_pop = pop[self.nto:self.nto + nw]

        # Get the best tornado
        g_best = tornado_pop[0]

        # Evolution process
        pop_new = []

        # Update windstorms
        for idx in range(len(windstorm_pop)):
            pos_new = windstorm_pop[idx][self.ID_POS].copy()

            # Calculate Coriolis force parameters
            Rl = 2 / (1 + np.exp((-epoch + self.epoch / 2) / 2))
            Rr = -2 / (1 + np.exp((-epoch + self.epoch / 2) / 2))

            if np.random.random() > 0.5:
                delta = np.random.choice([-1, 1])
                rr = 1 + np.random.random() * 3
                wr = (2 * np.random.random() - (1 + np.random.random())) / rr
                c = self.b_r * delta * wr

                # Left rotation
                f = 2 * self.omega * np.sin(-1 + 2 * np.random.random())
                zeta = np.random.randint(0, To)
                phi = tornado_pop[zeta][self.ID_POS] - pos_new

                if Rl >= 0:
                    phi = np.where(phi >= 0, -phi, phi)

                CFl = ((f ** 2 * Rl ** 2) / 4) - Rl * phi
                CFl = np.abs(CFl)

                pos_new = pos_new + 0.5 * (c * (f * Rl) / 2 + np.sqrt(CFl))
            else:
                delta = np.random.choice([-1, 1])
                rr = 1 + np.random.random() * 3
                wr = (2 * np.random.random() - (1 + np.random.random())) / rr
                c = self.b_r * delta * wr

                # Right rotation
                f = 2 * self.omega * np.sin(-1 + 2 * np.random.random())
                zeta = np.random.randint(0, To)
                phi = tornado_pop[zeta][self.ID_POS] - pos_new

                if Rr <= 0:
                    phi = np.where(phi <= 0, -phi, phi)

                CFr = ((f ** 2 * Rr ** 2) / 4) - Rr * phi
                CFr = np.abs(CFr)

                pos_new = pos_new + 0.5 * (c * (f * Rr) / 2 + np.sqrt(CFr))

            # Random formation check
            if np.linalg.norm(pos_new - tornado_pop[0][self.ID_POS]) < nu:
                delta = np.random.choice([-1, 1])
                pos_new = pos_new - (2 * ay * (
                            np.random.random() * (self.problem.ub - self.problem.lb) - self.problem.lb)) * delta

            # Boundary handling
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            pop_new.append([pos_new, target])

        # Update thunderstorms
        for idx in range(len(thunderstorm_pop)):
            pos_new = thunderstorm_pop[idx][self.ID_POS].copy()

            # Evolution
            alpha = abs(2 * ay * np.random.random() - np.random.random())
            r1, r2 = np.random.choice(len(thunderstorm_pop), 2, replace=False)

            pos_new = pos_new + 2 * alpha * (thunderstorm_pop[r1][self.ID_POS] - pos_new) + \
                      2 * alpha * (g_best[self.ID_POS] - thunderstorm_pop[r2][self.ID_POS])

            # Random formation check
            if np.linalg.norm(pos_new - g_best[self.ID_POS]) < nu:
                delta = np.random.choice([-1, 1])
                pos_new = pos_new - (2 * ay * (
                            np.random.random() * (self.problem.ub - self.problem.lb) - self.problem.lb)) * delta

            # Boundary handling
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            pop_new.append([pos_new, target])

        # Update tornadoes
        for idx in range(len(tornado_pop)):
            if idx == 0:  # Best tornado
                pop_new.append(tornado_pop[idx])
                continue

            pos_new = tornado_pop[idx][self.ID_POS].copy()

            # Evolution
            alpha = abs(2 * ay * np.random.random() - np.random.random())
            pos_new = pos_new + 2 * alpha * (g_best[self.ID_POS] - pos_new)

            # Random formation check
            if np.linalg.norm(pos_new - g_best[self.ID_POS]) < nu:
                delta = np.random.choice([-1, 1])
                pos_new = pos_new - (2 * ay * (
                            np.random.random() * (self.problem.ub - self.problem.lb) - self.problem.lb)) * delta

            # Boundary handling
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            pop_new.append([pos_new, target])

        # Update population
        self.pop = pop_new