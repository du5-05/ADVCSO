import numpy as np
from mealpy.optimizer import Optimizer

class JADE(Optimizer):
    """
    The original version of: JADE (Adaptive Differential Evolution with Optional External Archive)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c (float): control parameter (0.1 is recommended), default=0.02
        + p_best (float): top best percentage to consider (0.05-0.2 is recommended), default=0.05
        + archive_size (float): archive size based on pop_size (1-2.6 is recommended), default=2.6
    """

    ID_POS = 0
    ID_TAR = 1
    ID_MUT = 2  # Mutation vector
    ID_CR = 3  # Crossover rate
    ID_F = 4  # Scale factor

    def __init__(self, epoch=10000, pop_size=100, c=0.02, p_best=0.05, archive_size=2.6, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c (float): control parameter, default = 0.02
            p_best (float): percentage of top best solutions, default = 0.05
            archive_size (float): archive size based on pop_size, default = 2.6
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c = self.validator.check_float("c", c, (0, 1.0))
        self.p_best = self.validator.check_float("p_best", p_best, (0, 1.0))
        self.archive_size = self.validator.check_float("archive_size", archive_size, (0, 5.0))
        self.set_parameters(["epoch", "pop_size", "c", "p_best", "archive_size"])
        self.sort_flag = False

        # Initialize adaptive parameters
        self.u_cr = 0.5  # Mean of normal distribution for CR
        self.u_f = 0.5  # Mean of Cauchy distribution for F
        self.archive = []

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, mutation, cr, f]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        mutation = np.zeros_like(position)
        cr = np.random.normal(self.u_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        f = self.u_f + 0.1 * np.random.standard_cauchy()
        while f <= 0:
            f = self.u_f + 0.1 * np.random.standard_cauchy()
        f = min(f, 1.0)
        return [position, target, mutation, cr, f]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Memory for successful parameters
        scr = []  # Successful CR
        sf = []  # Successful F
        delta_f = []  # Fitness improvement values

        pop_new = []
        for idx in range(0, self.pop_size):
            # Generate CR and F values
            cr = np.random.normal(self.u_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            f = self.u_f + 0.1 * np.random.standard_cauchy()
            while f <= 0:
                f = self.u_f + 0.1 * np.random.standard_cauchy()
            f = min(f, 1.0)

            # DE/current-to-pbest/1 mutation
            # Select random p_best solution
            p_idx = np.random.choice(range(int(self.pop_size * self.p_best)))
            x_pbest = sorted(self.pop, key=lambda x: x[self.ID_TAR][self.ID_FIT])[p_idx][self.ID_POS]

            # Select random solution from population, different from current
            idxs = np.random.choice(np.delete(range(self.pop_size), idx), 1)[0]
            x_r1 = self.pop[idxs][self.ID_POS]

            # Select random solution from union of population and archive
            if len(self.archive) > 0:  # Only combine with archive if it's not empty
                archive_union = np.vstack(
                    (np.array([agent[self.ID_POS] for agent in self.pop]), np.array(self.archive)))
                x_r2_idx = np.random.randint(0, len(archive_union))
                x_r2 = archive_union[x_r2_idx]
            else:
                # If archive is empty, select from population only
                x_r2_idx = np.random.choice(np.delete(range(self.pop_size), [idx, idxs]), 1)[0]
                x_r2 = self.pop[x_r2_idx][self.ID_POS]

            # Mutation
            v = self.pop[idx][self.ID_POS] + f * (x_pbest - self.pop[idx][self.ID_POS]) + f * (x_r1 - x_r2)
            v = self.amend_position(v, self.problem.lb, self.problem.ub)

            # Crossover
            j_rand = np.random.randint(0, self.problem.n_dims)
            pos_new = np.where(np.random.random(self.problem.n_dims) < cr, v, self.pop[idx][self.ID_POS])
            pos_new[j_rand] = v[j_rand]  # Ensure at least one dimension is changed

            # Create new solution and evaluate
            agent = [pos_new, None, v, cr, f]  # Keep mutation vector for possible archive update
            if self.mode not in self.AVAILABLE_MODES:
                agent[self.ID_TAR] = self.get_target_wrapper(pos_new)
            pop_new.append(agent)

        # Update target for all solutions
        pop_new = self.update_target_wrapper_population(pop_new)

        # Selection and update archive
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                # Update archive with old solution
                self.archive.append(self.pop[idx][self.ID_POS])
                if len(self.archive) > int(self.archive_size * self.pop_size):
                    self.archive.pop(np.random.randint(0, len(self.archive)))
                # Store successful parameters
                scr.append(pop_new[idx][self.ID_CR])
                sf.append(pop_new[idx][self.ID_F])
                delta_f.append(abs(pop_new[idx][self.ID_TAR][self.ID_FIT] - self.pop[idx][self.ID_TAR][self.ID_FIT]))
                # Replace old solution
                self.pop[idx] = pop_new[idx]

        # Update u_cr and u_f if there were successful mutations
        if len(scr) > 0:
            # Lehmer mean for F
            weights = np.array(delta_f) / np.sum(delta_f)
            self.u_f = (1 - self.c) * self.u_f + self.c * np.sum(weights * np.array(sf) ** 2) / np.sum(
                weights * np.array(sf))
            # Arithmetic mean for CR
            self.u_cr = (1 - self.c) * self.u_cr + self.c * np.mean(scr)