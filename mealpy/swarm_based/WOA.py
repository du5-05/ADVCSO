import numpy as np
from mealpy.optimizer import Optimizer

class OriginalWOA(Optimizer):
    """
    鲸鱼优化算法（WOA）的原始版本
    原文链接： https://doi.org/10.1016/j.advengsoft.2016.01.008
    例如：
    >>> import numpy as np
    >>> from mealpy.swarm_based.WOA import OriginalWOA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = OriginalWOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        参数设置:
            epoch (int): 最大迭代次数, 默认值 = 10000
            pop_size (int): 种群数量, 默认值 = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): 当前迭代次数
        """
        a = 2 - 2 * epoch / (self.epoch - 1)  # 从2线性下降到0。控制搜索空间的缩小
        pop_new = []  # 初始化空列表，存储每一代更新后的种群

        for idx in range(0, self.pop_size):

            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = 0.5
            b = 1

            if np.random.uniform() < p:
                if np.abs(A) < 1:  # A的绝对值小于1
                    D = np.abs(C * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])  # D表示当前个体与全局最佳个体之间的距离
                    pos_new = self.g_best[self.ID_POS] - A * D  # 更新当前个体的位置
                else:  # A的绝对值大于1
                    # x_rand = pop[np.random.np.random.randint(self.pop_size)]         # 在种群中随机选择1个个体
                    x_rand = self.create_solution(self.problem.lb, self.problem.ub)
                    D = np.abs(C * x_rand[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = x_rand[self.ID_POS] - A * D
            else:
                D1 = np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] + np.exp(b * l) * np.cos(2 * np.pi * l) * D1

            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)  # 确保新的个体位置在界限内

            pop_new.append([pos_new, None])  # 将更新后的个体位置添加到pop_new中

            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)  # 计算新位置的目标值
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            print(len(pop_new))
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class HI_WOA(Optimizer):
    """
    混合改进鲸鱼优化算法：Hybrid Improved Whale Optimization Algorithm (HI-WOA)的原始版本

    Links: https://ieenp.explore.ieee.org/document/8900003

    超参数应在近似范围内微调，以更快地收敛到全局最优值:
        + feedback_max (int): 每次反馈的最大迭代次数, 默认值 = 10

    Examples
    >>> import numpy as np
    >>> from mealpy.swarm_based.WOA import HI_WOA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> feedback_max = 10
    >>> model = HI_WOA(epoch, pop_size, feedback_max)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, feedback_max=10, **kwargs):
        """
        参数设置:
            epoch (int): 最大迭代次数, default = 10000
            pop_size (int): 种群数量, default = 100
            feedback_max (int): 每次反馈的最大迭代次数, default = 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.feedback_max = self.validator.check_int("feedback_max", feedback_max, [2, 2+int(self.epoch/2)])
        # The maximum of times g_best doesn't change -> need to change half of population
        self.set_parameters(["epoch", "pop_size", "feedback_max"])
        self.sort_flag = True

    def initialize_variables(self):
        self.n_changes = int(self.pop_size / 2)
        self.dyn_feedback_count = 0

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): 当前迭代次数
        """
        a = 2 + 2 * np.cos(np.pi / 2 * (1 + epoch / self.epoch))  # Eq. 8
        pop_new = []
        for idx in range(0, self.pop_size):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = 0.5
            b = 1
            if np.random.uniform() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.g_best[self.ID_POS] - A * D
                else:
                    # x_rand = pop[np.random.np.random.randint(self.pop_size)]         # select random 1 position in pop
                    x_rand = self.create_solution(self.problem.lb, self.problem.ub)
                    D = np.abs(C * x_rand[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = x_rand[self.ID_POS] - A * D
            else:
                D1 = np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] + np.exp(b * l) * np.cos(2 * np.pi * l) * D1
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        ## 反馈机制
        _, current_best = self.get_global_best_solution(self.pop)
        if current_best[self.ID_TAR][self.ID_FIT] == self.g_best[self.ID_TAR][self.ID_FIT]:
            self.dyn_feedback_count += 1
        else:
            self.dyn_feedback_count = 0

        if self.dyn_feedback_count >= self.feedback_max:
            idx_list = np.random.choice(range(0, self.pop_size), self.n_changes, replace=False)
            pop_child = self.create_population(self.n_changes)
            for idx_counter, idx in enumerate(idx_list):
                self.pop[idx] = pop_child[idx_counter]
