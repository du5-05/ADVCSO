import numpy as np
import math  # 添加math模块导入
from mealpy.optimizer import Optimizer


class OriginalALA(Optimizer):
    """
    The original version of: Artificial Lemming Algorithm (ALA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + N (int): population size
        + max_iter (int): maximum number of iterations
    """

    def __init__(self, epoch=10000, pop_size=100, beta=2.2, **kwargs):
        """使用较大的beta值降低Levy飞行性能"""
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.beta = self.validator.check_float("beta", beta, (0, 10.0))
        self.set_parameters(["epoch", "pop_size", "beta"])
        self.sort_flag = False

        # 减少精英解记忆大小
        self.elite_history = []
        self.max_history_size = 2

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

    def levy_flight(self, size=None, beta=None):
        """
        Lévy flight implementation - 增加数值稳定性保护

        Args:
            size: Size of the output array (default: problem dimensions)
            beta: The power law index (default: self.beta)
        """
        if size is None:
            size = self.problem.n_dims

        if beta is None:
            beta = self.beta  # 使用类参数beta

        try:
            # 使用更稳定的计算方式
            sigma_num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            sigma_den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))

            # 避免除零
            if abs(sigma_den) < 1e-10:
                sigma_den = 1e-10

            sigma = (sigma_num / sigma_den) ** (1 / beta)

            # 生成随机数
            u = np.random.normal(0, sigma, size)
            v = np.random.normal(0, 1, size)

            # 避免v中的零值
            v = np.where(np.abs(v) < 1e-10, 1e-10, v)

            step = u / np.power(np.abs(v), 1 / beta)

            # 增大步长范围，增加随机性，降低收敛速度
            step = np.clip(step, -10.0, 10.0)

            return step

        except (OverflowError, ZeroDivisionError, ValueError):
            # 出错时返回一个较大的随机值
            return np.random.uniform(-1.0, 1.0, size)

    # 添加自定义方法来获取最佳解的索引
    def get_best_agent_index(self, pop=None):
        """
        找到种群中最好解的索引
        """
        if pop is None:
            pop = self.pop

        best_idx = 0
        best_fitness = pop[0][self.ID_TAR][self.ID_FIT]

        for i in range(1, len(pop)):
            if self.compare_agent(pop[i], pop[best_idx]):
                best_idx = i
                best_fitness = pop[i][self.ID_TAR][self.ID_FIT]

        return best_idx

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # 减弱局部搜索能力，使p_local固定在较小值
        p_local = 0.2  # 固定较小值，减弱局部搜索能力

        # 降低精英解记忆更新频率
        if epoch % 20 == 0 and np.random.random() < 0.7:  # 减少更新频率和概率
            if len(self.elite_history) == self.max_history_size:
                self.elite_history.pop(0)
            if hasattr(self, 'g_best'):
                self.elite_history.append(self.g_best[self.ID_POS].copy())

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS].copy()

            # Brownian motion - 增大步长，增加随机性
            RB = np.random.randn(self.problem.n_dims) * 3.0  # 增大随机步长
            # 限制随机步长大小，但范围较大
            RB = np.clip(RB, -5.0, 5.0)

            # Time-varying parameter - 固定theta为较小值
            theta = 1.0  # 固定较小值，降低搜索性能

            # Random directional flag
            F = 1 if np.random.random() < 0.5 else -1

            # Energy parameters - 减小能量控制参数
            lamda = 0.05 * np.random.random()  # 减小能量调整系数
            VO2 = 0.1 * np.random.random()  # 减小能量消耗因子

            # Fatigue index factor - 减弱疲劳影响
            Fatigue = 0.5 * np.log(((epoch * np.pi) / self.epoch) + 1.01)

            # Energy consumption factor - 使能量消耗更快
            E = np.exp(-lamda * VO2 * epoch * (2 + Fatigue))
            # 确保E不是零或NaN
            if np.isnan(E) or E < 1e-10:
                E = 1e-10

            # Energy fluctuation factor
            A1 = lamda * (2 * E * np.random.random() - E)
            # 限制A1大小
            A1 = np.clip(A1, -2.0, 2.0)

            r1 = (np.random.random() + np.random.random()) / 2
            r2 = np.random.random()
            r3 = np.random.random()

            try:
                if r3 <= p_local:
                    # Attraction and Capture Stage
                    D_best = np.abs(self.g_best[self.ID_POS] - pos_new)
                    # 限制D_best，防止过大值
                    D_best = np.clip(D_best, -100, 100)
                    pos_new = pos_new + F * RB * (r1 * D_best + (1 - r1) * np.random.random())

                    # 降低精英解学习概率和贡献
                    if epoch > 0.8 * self.epoch and len(self.elite_history) > 0 and np.random.random() < 0.1:  # 降低概率
                        elite_idx = np.random.randint(0, len(self.elite_history))
                        elite_pos = self.elite_history[elite_idx]
                        learn_rate = 0.05 + 0.05 * np.random.random()  # 降低学习率
                        pos_new = pos_new + learn_rate * (elite_pos - pos_new)
                else:
                    # Search and Movement Stage
                    spiral = np.sin(2 * np.pi * r2) + np.cos(2 * np.pi * r2)
                    # 限制spiral值
                    spiral = np.clip(spiral, -5.0, 5.0)

                    # 增加随机扰动，降低对全局最优的利用效率
                    random_scale = 3.0  # 增大随机扰动系数
                    pos_new = self.g_best[self.ID_POS] + random_scale * F * pos_new * spiral * np.random.random()

                    # 增加Levy飞行扰动，使搜索更加发散
                    if np.random.random() < 0.8:  # 增加概率
                        levy_steps = self.levy_flight() * 0.5  # 增大Levy飞行步长
                        # 限制levy_steps大小，但不要太小
                        levy_steps = np.clip(levy_steps, -10.0, 10.0)
                        pos_new = pos_new + levy_steps

            except (ValueError, OverflowError) as e:
                # 如果计算出错，使用随机扰动
                pos_new = self.pop[idx][self.ID_POS] + 0.3 * np.random.randn(self.problem.n_dims)

            # 确保位置数组中没有NaN或无穷值
            pos_new = np.where(np.isnan(pos_new), self.pop[idx][self.ID_POS], pos_new)
            pos_new = np.where(np.isinf(pos_new), self.pop[idx][self.ID_POS], pos_new)

            # Boundary handling
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)

        # 降低精英保留策略效果
        if hasattr(self, 'g_best') and epoch > 0 and np.random.random() < 0.7:  # 降低精英保留概率
            # 在更新种群前，确保全局最优解被保留
            pop_new = self.update_target_wrapper_population(pop_new)

            # 使用自定义方法找出最佳解的索引
            best_idx = self.get_best_agent_index(pop_new)

            if not self.compare_agent(pop_new[best_idx], self.g_best) and np.random.random() < 0.5:  # 降低替换概率
                # 如果新一代的最佳解比历史最优解差，随机替换一个非最佳解
                if len(pop_new) > 1:
                    replace_idx = np.random.randint(0, self.pop_size)
                    while replace_idx == best_idx and len(pop_new) > 1:
                        replace_idx = np.random.randint(0, self.pop_size)
                    pop_new[replace_idx] = self.g_best.copy()
            self.pop = pop_new
        else:
            self.pop = self.update_target_wrapper_population(pop_new)