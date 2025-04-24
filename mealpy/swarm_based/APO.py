import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAPO(Optimizer):
    """
    The original version of: Artificial Protozoa Optimizer (APO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pf_max (float): proportion fraction maximum, default=0.3
        + np (int): number of neighbor pairs, default=1
    """

    def __init__(self, epoch=10000, pop_size=100, pf_max=0.1, np_pairs=1, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pf_max (float): proportion fraction maximum, default = 0.3
            np_pairs (int): number of neighbor pairs, default = 1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pf_max = self.validator.check_float("pf_max", pf_max, (0, 1.0))
        self.np_pairs = self.validator.check_int("np_pairs", np_pairs, [1, pop_size // 2])
        self.set_parameters(["epoch", "pop_size", "pf_max", "np_pairs"])
        self.sort_flag = True

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

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
        # Get proportion fraction for current iteration
        pf = self.pf_max

        # Get random indexes for protozoa in dormancy or reproduction forms
        ri = np.random.permutation(self.pop_size)
        ri = ri[:int(np.ceil(self.pop_size * pf))]

        # Create new population
        pop_new = []
        for i in range(self.pop_size):
            # Clone current solution
            protozoa = self.pop[i].copy()
            pos_new = protozoa[self.ID_POS].copy()

            # Check if protozoa is in dormancy or reproduction form
            if i in ri:
                # Calculate probability of dormancy and reproduction
                pdr = 0.5 * (1 + np.cos((1 - i / self.pop_size) * np.pi))

                if np.random.random() < pdr:  # Dormancy form
                    # Generate random position
                    pos_new = self.generate_position(self.problem.lb, self.problem.ub)
                else:  # Reproduction form
                    # Randomly select + or -
                    flag = 1 if np.random.random() < 0.5 else -1

                    # Create mapping vector in reproduction
                    mr = np.zeros(self.problem.n_dims)
                    idx = np.random.permutation(self.problem.n_dims)
                    num_dims = int(np.ceil(np.random.random() * self.problem.n_dims * 0.5))
                    mr[idx[:num_dims]] = 1

                    # Update position
                    random_pos = self.generate_position(self.problem.lb, self.problem.ub)
                    pos_new = pos_new + flag * 0.5 * random_pos * mr
            else:  # Protozoa is in foraging form
                # Calculate foraging factor
                f = 0.5 * np.random.random() * (1 + np.cos(epoch / self.epoch * np.pi))

                # Create mapping vector in foraging
                mf = np.zeros(self.problem.n_dims)
                idx = np.random.permutation(self.problem.n_dims)
                num_dims = int(np.ceil(self.problem.n_dims * i / (self.pop_size * 2)))
                mf[idx[:num_dims]] = 1

                # Calculate probability of autotroph and heterotroph
                pah = 0.4 * (1 + np.cos(epoch / self.epoch * np.pi))

                if np.random.random() < pah:  # Protozoa is in autotroph form
                    # Randomly select another protozoa
                    j = np.random.randint(0, self.pop_size)

                    # Calculate effect of paired neighbors
                    epn = np.zeros((self.np_pairs, self.problem.n_dims))
                    for k in range(self.np_pairs):
                        # Select paired neighbors
                        if i == 0:
                            km = i
                            kp = i + np.random.randint(1, self.pop_size)
                        elif i == self.pop_size - 1:
                            km = np.random.randint(0, self.pop_size - 1)
                            kp = i
                        else:
                            km = np.random.randint(0, i)
                            kp = i + np.random.randint(1, self.pop_size - i)

                        # Calculate weight factor in autotroph form
                        denominator = self.pop[kp][self.ID_TAR][self.ID_FIT]
                        denominator = 1e-10 if abs(denominator) < 1e-10 else denominator
                        wa = 0.3 * np.exp(-abs(self.pop[km][self.ID_TAR][self.ID_FIT] / denominator))

                        # Calculate effect of paired neighbors
                        epn[k] = wa * (self.pop[km][self.ID_POS] - self.pop[kp][self.ID_POS])

                    # Update position
                    pos_new = pos_new + f * (self.pop[j][self.ID_POS] - pos_new + np.mean(epn, axis=0)) * mf

                else:  # Protozoa is in heterotroph form
                    # Calculate effect of paired neighbors
                    epn = np.zeros((self.np_pairs, self.problem.n_dims))
                    for k in range(self.np_pairs):
                        # Select paired neighbors
                        if i == 0:
                            imk = i
                            ipk = i + k
                        elif i == self.pop_size - 1:
                            imk = self.pop_size - 1 - k
                            ipk = i
                        else:
                            imk = i - k
                            ipk = i + k

                        # Ensure indices are within bounds
                        imk = max(0, imk)
                        ipk = min(self.pop_size - 1, ipk)

                        # Calculate weight factor in heterotroph form
                        denominator = self.pop[ipk][self.ID_TAR][self.ID_FIT]
                        denominator = 1e-10 if abs(denominator) < 1e-10 else denominator
                        wh = 0.3 * np.exp(-abs(self.pop[imk][self.ID_TAR][self.ID_FIT] / denominator))

                        # Calculate effect of paired neighbors
                        epn[k] = wh * (self.pop[imk][self.ID_POS] - self.pop[ipk][self.ID_POS])

                    # Randomly select + or -
                    flag = 1 if np.random.random() < 0.5 else -1

                    # Calculate X_near
                    coefficients = 1 + flag * 0.3 * np.random.random(self.problem.n_dims) * (1 - epoch / self.epoch)
                    x_near = coefficients * pos_new

                    # Update position
                    pos_new = pos_new + f * (x_near - pos_new + np.mean(epn, axis=0)) * mf

            # Amend position and evaluate fitness
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            protozoa[self.ID_POS] = pos_new

            # Add to new population
            pop_new.append(protozoa)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)

        # Update target wrappers for the entire population
        pop_new = self.update_target_wrapper_population(pop_new)

        # Update the pop and sort them based on fitness
        self.pop = self.greedy_selection_population(self.pop, pop_new)