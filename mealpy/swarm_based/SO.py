import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSO(Optimizer):
    """
    The original version of: Snake Optimizer (SO) 蛇优化算法
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

        # Algorithm parameters
        self.vec_flag = [1, -1]
        self.threshold = 0.25
        self.threshold2 = 0.6
        self.c1 = 0.5
        self.c2 = 0.05
        self.c3 = 2.0

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Create a new solution for SO algorithm

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
        # Calculate temperature
        temp = np.exp(-(epoch / self.epoch))
        q = self.c1 * np.exp((epoch - self.epoch) / self.epoch)
        q = min(q, 1.0)

        # Divide population into males and females
        nm = self.pop_size // 2  # Number of males
        nf = self.pop_size - nm  # Number of females

        pop_males = self.pop[:nm]
        pop_females = self.pop[nm:]

        pop_new = []

        # Exploration phase (no food)
        if q < self.threshold:
            # Update males
            for idx in range(nm):
                pos_new = pop_males[idx][self.ID_POS].copy()

                rand_idx = np.random.randint(nm)
                x_rand = pop_males[rand_idx][self.ID_POS]
                flag = np.random.choice(self.vec_flag)

                am = np.exp(
                    -pop_males[rand_idx][self.ID_TAR][self.ID_FIT] / (pop_males[idx][self.ID_TAR][self.ID_FIT] + 1e-10))
                pos_new = x_rand + flag * self.c2 * am * (
                            (self.problem.ub - self.problem.lb) * np.random.random() + self.problem.lb)

                # Create new solution
                agent = [pos_new, None]
                pop_new.append(agent)

            # Update females
            for idx in range(nf):
                pos_new = pop_females[idx][self.ID_POS].copy()

                rand_idx = np.random.randint(nf)
                x_rand = pop_females[rand_idx][self.ID_POS]
                flag = np.random.choice(self.vec_flag)

                af = np.exp(-pop_females[rand_idx][self.ID_TAR][self.ID_FIT] / (
                            pop_females[idx][self.ID_TAR][self.ID_FIT] + 1e-10))
                pos_new = x_rand + flag * self.c2 * af * (
                            (self.problem.ub - self.problem.lb) * np.random.random() + self.problem.lb)

                # Create new solution
                agent = [pos_new, None]
                pop_new.append(agent)

        else:  # Exploitation phase (food exists)
            if temp > self.threshold2:  # Hot
                for idx in range(self.pop_size):
                    pos_new = self.pop[idx][self.ID_POS].copy()
                    flag = np.random.choice(self.vec_flag)

                    pos_new = self.g_best[self.ID_POS] + self.c3 * flag * temp * np.random.random() * \
                              (self.g_best[self.ID_POS] - pos_new)

                    # Create new solution
                    agent = [pos_new, None]
                    pop_new.append(agent)

            else:  # Cold
                if np.random.random() > 0.6:  # Fight
                    # Update males
                    for idx in range(nm):
                        pos_new = pop_males[idx][self.ID_POS].copy()

                        fm = np.exp(-np.min([x[self.ID_TAR][self.ID_FIT] for x in pop_females]) / \
                                    (pop_males[idx][self.ID_TAR][self.ID_FIT] + 1e-10))
                        pos_new = pos_new + self.c3 * fm * np.random.random() * \
                                  (q * self.g_best[self.ID_POS] - pos_new)

                        # Create new solution
                        agent = [pos_new, None]
                        pop_new.append(agent)

                    # Update females
                    for idx in range(nf):
                        pos_new = pop_females[idx][self.ID_POS].copy()

                        ff = np.exp(-np.min([x[self.ID_TAR][self.ID_FIT] for x in pop_males]) / \
                                    (pop_females[idx][self.ID_TAR][self.ID_FIT] + 1e-10))
                        pos_new = pos_new + self.c3 * ff * np.random.random() * \
                                  (q * self.g_best[self.ID_POS] - pos_new)

                        # Create new solution
                        agent = [pos_new, None]
                        pop_new.append(agent)

                else:  # Mating
                    # Update males
                    for idx in range(nm):
                        pos_new = pop_males[idx][self.ID_POS].copy()

                        mm = np.exp(-pop_females[idx][self.ID_TAR][self.ID_FIT] / \
                                    (pop_males[idx][self.ID_TAR][self.ID_FIT] + 1e-10))
                        pos_new = pos_new + self.c3 * np.random.random() * mm * \
                                  (q * pop_females[idx][self.ID_POS] - pos_new)

                        # Create new solution
                        agent = [pos_new, None]
                        pop_new.append(agent)

                    # Update females
                    for idx in range(nf):
                        pos_new = pop_females[idx][self.ID_POS].copy()

                        mf = np.exp(-pop_males[idx][self.ID_TAR][self.ID_FIT] / \
                                    (pop_females[idx][self.ID_TAR][self.ID_FIT] + 1e-10))
                        pos_new = pos_new + self.c3 * np.random.random() * mf * \
                                  (q * pop_males[idx][self.ID_POS] - pos_new)

                        # Create new solution
                        agent = [pos_new, None]
                        pop_new.append(agent)

                    # Random egg laying
                    if np.random.choice(self.vec_flag) == 1:
                        # Replace worst male
                        worst_male_idx = np.argmax([x[self.ID_TAR][self.ID_FIT] for x in pop_males])
                        pos_new = self.generate_position(self.problem.lb, self.problem.ub)
                        pop_new[worst_male_idx] = [pos_new, None]

                        # Replace worst female
                        worst_female_idx = np.argmax([x[self.ID_TAR][self.ID_FIT] for x in pop_females])
                        pos_new = self.generate_position(self.problem.lb, self.problem.ub)
                        pop_new[nm + worst_female_idx] = [pos_new, None]

        # Boundary handling and update fitness
        for idx in range(len(pop_new)):
            pos_new = self.amend_position(pop_new[idx][self.ID_POS], self.problem.lb, self.problem.ub)
            pop_new[idx][self.ID_POS] = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx][self.ID_TAR] = self.get_target_wrapper(pos_new)

        # Update population
        self.pop = self.update_target_wrapper_population(pop_new)