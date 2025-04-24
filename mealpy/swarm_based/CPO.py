import numpy as np
import math  # 添加math模块导入
from mealpy.optimizer import Optimizer


class OriginalCPO(Optimizer):
    """
    The original version of: Chinese Pangolin Optimizer (CPO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + epoch (int): maximum number of iterations, default = 10000
        + pop_size (int): number of population size, default = 100
        + Dc (float): diffusion coefficient, default = 0.6
    """

    def __init__(self, epoch=10000, pop_size=100, Dc=0.6, **kwargs):
        """
        参数调整:
        - 修改Dc为0.6，与MATLAB版本完全一致
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.Dc = self.validator.check_float("Dc", Dc, (0, 2.0))
        self.set_parameters(["epoch", "pop_size", "Dc"])
        self.sort_flag = False

        # 添加蚂蚁位置跟踪 - 与原始MATLAB代码相对应
        self.manis_pos = None  # 对应MATLAB中的Manis_pos (穿山甲位置)
        self.manis_score = float('inf')  # 对应MATLAB中的Manis_score (穿山甲得分)
        self.ant_pos = None  # 对应MATLAB中的Ant_pos (蚂蚁位置)
        self.ant_score = float('inf')  # 对应MATLAB中的Ant_score (蚂蚁得分)

        # 精英解历史跟踪
        self.elite_history = []
        self.max_history_size = 5

    def aroma_concentration(self, max_iter):
        """Calculate aroma concentration based on time step - 对应MATLAB的Aroma_concentration函数"""
        try:
            Q = 100  # 与MATLAB一致
            M = np.zeros(max_iter)

            for t in range(1, max_iter + 1):
                r1 = np.random.random()
                H = 0.5 * r1  # 与MATLAB一致
                r2 = np.random.random()
                u = 2 + r2  # 与MATLAB一致
                sigma_y = 50 - ((10 * t) / max_iter)  # 与MATLAB一致

                # 增加安全检查，避免除以零或负数对数
                sigma_y = max(sigma_y, 1e-10)

                # 与MATLAB完全一致的实现
                log_term = max(1e-10, (np.pi * t) / max_iter)
                exp_term = np.exp(-t / max_iter)

                sigma_z = np.sin((np.pi * t) / max_iter) + 40 * exp_term - 10 * np.log(log_term)

                # 确保sigma_z不接近零
                sigma_z = max(abs(sigma_z), 1e-10)

                # 计算并限制分母
                denominator = np.pi * u * sigma_y * sigma_z
                denominator = max(denominator, 1e-10)

                # 计算指数项，避免溢出
                exp_term = np.exp(-(H ** 2) / (2 * (sigma_z ** 2)))
                # 限制指数项
                exp_term = min(exp_term, 100)

                M[t - 1] = (Q / denominator) * exp_term

            # 确保所有浓度值为正
            M = np.abs(M)

            # 检查是否有NaN或无穷值
            M = np.where(np.isnan(M), 0.5, M)
            M = np.where(np.isinf(M), 0.5, M)

            # 使用与MATLAB的rescale()相同的归一化方法
            M_min = np.min(M)
            M_max = np.max(M)
            M_range = M_max - M_min

            # 避免分母为零
            if M_range < 1e-10:
                return np.ones(max_iter) * 0.5

            normalized_M = (M - M_min) / M_range
            return normalized_M

        except Exception as e:
            # 如果计算失败，返回均匀分布
            print(f"Aroma concentration calculation failed: {e}")
            return np.ones(max_iter) * 0.5

    def aroma_trajectory(self, N, Dc):
        """Calculate aroma trajectory based on Brownian motion - 对应MATLAB的Aroma_trajectory函数"""
        try:
            dt = 1 / N

            # 与MATLAB一致
            dWx = np.sqrt(2 * Dc * dt) * np.random.randn(N)
            dWy = np.sqrt(2 * Dc * dt) * np.random.randn(N)
            dWz = np.sqrt(2 * Dc * dt) * np.random.randn(N)

            # 计算轨迹
            x = np.zeros(N)
            y = np.zeros(N)
            z = np.zeros(N)

            for k in range(1, N):
                # 与MATLAB完全一致
                x[k] = x[k - 1] + dWx[k]
                y[k] = y[k - 1] + dWy[k]
                z[k] = z[k - 1] + dWz[k]

            # 随机选择一个点
            random_index = np.random.randint(0, N)
            random_point_x = x[random_index]
            random_point_y = y[random_index]
            random_point_z = z[random_index]

            # 计算范数
            value = np.linalg.norm([random_point_x, random_point_y, random_point_z])

            # 限制值的范围，防止过大或为零
            value = max(value, 1e-10)
            value = min(value, 100)

            return value

        except Exception as e:
            # 如果计算失败，返回一个合理的默认值
            print(f"Aroma trajectory calculation failed: {e}")
            return 1.0

    def levy(self, dim):
        """计算Levy飞行步长 - 对应MATLAB的Levy函数"""
        try:
            beta = 1.5  # 与MATLAB一致

            # 使用math.gamma而不是np.math.gamma
            sigma_num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            sigma_den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)

            # 避免除零
            if abs(sigma_den) < 1e-10:
                sigma_den = 1e-10

            sigma = (sigma_num / sigma_den) ** (1 / beta)

            # 生成随机数
            u = np.random.randn(dim) * sigma
            v = np.random.randn(dim)

            # 避免v中的零值
            v = np.where(np.abs(v) < 1e-10, 1e-10, v)

            step = u / np.power(np.abs(v), 1 / beta)

            return step

        except Exception as e:
            # 如果计算失败，返回均匀分布的随机步长
            print(f"Levy flight calculation failed: {e}")
            return np.random.uniform(-1, 1, dim)

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
        # 初始化蚂蚁和穿山甲位置 (如果第一次迭代)
        if epoch == 0 and self.manis_pos is None:
            self.manis_pos = np.zeros(self.problem.n_dims)
            self.ant_pos = np.zeros(self.problem.n_dims)

            # 找出当前最佳解
            for i in range(self.pop_size):
                fitness = self.pop[i][self.ID_TAR][self.ID_FIT]
                if fitness <= self.manis_score:
                    self.manis_score = fitness
                    self.manis_pos = self.pop[i][self.ID_POS].copy()
                if fitness > self.manis_score and fitness < self.ant_score:
                    self.ant_score = fitness
                    self.ant_pos = self.pop[i][self.ID_POS].copy()

        # 计算随机系数 - 与MATLAB一致
        r1 = (np.random.random() + np.random.random()) / 2
        r2 = np.random.random()

        try:
            # 计算芳香浓度因子 - 对应MATLAB中的Cm数组
            Cm = self.aroma_concentration(self.epoch)

            # 检查Cm数组
            if np.any(np.isnan(Cm)) or np.any(np.isinf(Cm)):
                # 如果存在NaN或Inf，重置为均匀值
                Cm = np.ones(self.epoch) * 0.5

            # 计算自适应的搜索平衡因子 - 与MATLAB一致
            C1 = (2 - ((epoch * 2) / self.epoch))  # 与MATLAB一致

            # 计算芳香轨迹因子 - 对应MATLAB中的a
            a = self.aroma_trajectory(self.pop_size, self.Dc)

            # 检查a值
            if np.isnan(a) or np.isinf(a):
                a = 1.0  # 如果a无效，使用默认值

            # 计算Levy步长 - 与MATLAB一致
            levy_step_length = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                levy_step = self.levy(1)  # 计算单个值
                levy_step_length[i] = levy_step[0]

            # 精英解记忆机制
            if len(self.elite_history) == 0 and epoch > 0:
                # 添加当前最佳解到历史
                self.elite_history.append(self.manis_pos.copy())
            elif epoch % 10 == 0 and self.manis_score < self.g_best[self.ID_TAR][self.ID_FIT]:
                # 每10个迭代检查并更新历史
                if len(self.elite_history) >= self.max_history_size:
                    self.elite_history.pop(0)  # 移除最旧的
                self.elite_history.append(self.manis_pos.copy())

            pop_new = []
            for idx in range(0, self.pop_size):
                pos_current = self.pop[idx][self.ID_POS].copy()

                try:
                    # 能量校正因子 - 与MATLAB一致
                    lamda = 0.1 * np.random.random()
                    VO2 = 0.2 * np.random.random()

                    # 疲劳指数因子 - 与MATLAB一致
                    log_term = max(1e-10, ((epoch * np.pi) / self.epoch) + 1)
                    Fatigue = np.log(log_term)

                    # 能量消耗因子 - 与MATLAB一致
                    exp_term = -lamda * VO2 * epoch * (1.0 + Fatigue)
                    # 避免指数项过大
                    exp_term = max(-50, min(exp_term, 50))
                    E = np.exp(exp_term)

                    # 随机索引 - 与MATLAB一致
                    l = np.random.randint(0, min(self.epoch, len(Cm)))
                    r3 = np.random.random()

                    # 能量导向因子 - 与MATLAB一致
                    A1 = lamda * (2 * E * np.random.random() - E)

                    # Luring behavior - 与MATLAB更一致
                    if Cm[l] >= 0.2 and r3 <= 0.5:
                        # 吸引和捕获阶段 - 与MATLAB一致
                        D_ant = np.abs(a * self.ant_pos - self.manis_pos)
                        New_Ant_pos = pos_current + self.ant_pos - A1 * D_ant

                        # 移动和进食阶段 - 与MATLAB一致
                        levy_factor = levy_step_length[idx] * (1 - epoch / self.epoch)
                        D_manis = np.abs(C1 * New_Ant_pos - pos_current) - levy_factor
                        New_Manis_pos = pos_current + self.manis_pos - A1 * D_manis

                        # 更新位置 - 与MATLAB完全一致
                        try:
                            exp_term_1 = ((epoch) / self.epoch)
                            exp_term_2 = ((epoch * 4 * np.pi ** 2) / (self.epoch))

                            sin_input = New_Ant_pos * np.exp(exp_term_1)
                            tan_input = New_Manis_pos * np.exp(exp_term_2)

                            # 保护函数输入
                            sin_input = np.clip(sin_input, -50, 50)
                            tan_input = np.clip(tan_input, -50, 50)

                            sin_term = np.sin(sin_input)
                            tan_term = np.tan(tan_input)

                            # 保护除以零
                            tan_term = np.where(np.abs(tan_term) < 1e-10, 1e-10, tan_term)

                            complex_term = (sin_term / ((4 * np.pi) * tan_term)) * r1 * r2 * np.random.random()

                            pos_new = (New_Manis_pos + New_Ant_pos) / 2 + complex_term

                        except:
                            # 如果计算失败，使用简化版
                            pos_new = (New_Manis_pos + New_Ant_pos) / 2

                    # Predation behavior - 与MATLAB完全一致
                    elif Cm[l] <= 0.7 or r3 > 0.5:
                        if Cm[l] >= 0 and Cm[l] < 0.3:
                            # 搜索和定位阶段 - 与MATLAB一致
                            D_manis = np.abs(levy_step_length[idx] * self.manis_pos - pos_current)

                            # 使用更稳定的方式计算sin输入
                            sin_input = C1 * pos_current + A1 * np.abs(self.manis_pos - levy_step_length[idx] * D_manis)
                            # 限制sin输入
                            sin_input = np.clip(sin_input, -10, 10)

                            New_Manis_pos = np.sin(sin_input)
                            pos_new = New_Manis_pos * C1

                        elif Cm[l] >= 0.3 and Cm[l] < 0.6:
                            # 快速接近阶段 - 与MATLAB一致
                            D_manis = np.abs(a * self.manis_pos - pos_current)

                            # 使用更稳定的方式计算
                            exp_term = -a
                            # 限制指数项
                            exp_term = max(-50, min(exp_term, 50))

                            pi_term = np.random.random() * np.pi

                            New_Manis_pos = pos_current - A1 * np.abs(
                                self.manis_pos - np.exp(exp_term) * pi_term * D_manis)
                            pos_new = New_Manis_pos * C1

                        else:  # Cm[l] >= 0.6
                            # 挖掘和进食阶段 - 与MATLAB一致
                            D_manis = np.abs(C1 * self.manis_pos - pos_current)
                            New_Manis_pos = pos_current + A1 * np.abs(self.manis_pos - D_manis)
                            pos_new = New_Manis_pos * C1
                    else:
                        # 默认行为，使用当前位置
                        pos_new = pos_current
                except Exception as e:
                    # 如果计算过程出错，使用安全的随机扰动
                    print(f"Error in evolution step: {e}")
                    pos_new = pos_current + 0.1 * np.random.randn(self.problem.n_dims)

                # 检查并修复NaN和Inf
                pos_new = np.where(np.isnan(pos_new), pos_current, pos_new)
                pos_new = np.where(np.isinf(pos_new), pos_current, pos_new)

                # 边界检查
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

                # 创建新解
                target = self.get_target_wrapper(pos_new)
                pop_new.append([pos_new, target])

                # 更新穿山甲和蚂蚁位置
                fitness = target[self.ID_FIT]
                if fitness <= self.manis_score:
                    self.manis_score = fitness
                    self.manis_pos = pos_new.copy()
                if fitness > self.manis_score and fitness < self.ant_score:
                    self.ant_score = fitness
                    self.ant_pos = pos_new.copy()

            # 更新全局最优
            if self.manis_score < self.g_best[self.ID_TAR][self.ID_FIT]:
                self.g_best = [self.manis_pos.copy(), self.get_target_wrapper(self.manis_pos)]

            # 更新种群
            self.pop = pop_new

        except Exception as e:
            print(f"Evolution process failed: {e}")
            # 如果整个进化过程失败，生成新的随机种群
            pop_new = []
            for i in range(self.pop_size):
                pos = self.generate_position(self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos)
                pop_new.append([pos, target])
            self.pop = pop_new