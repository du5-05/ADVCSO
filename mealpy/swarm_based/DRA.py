import numpy as np
import math  # 添加math模块导入
from mealpy.optimizer import Optimizer


class OriginalDRA(Optimizer):
    """
    The original version of: Divine Religions Algorithm (DRA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + epoch (int): maximum number of iterations, default = 10000
        + pop_size (int): number of population size, default = 100
        + n_groups (int): number of groups, default = 5
        + bpsp (float): belief profile consideration rate, default = 0.5
        + mp (float): miracle rate, default = 0.5
        + pp (float): proselytism consideration rate, default = 0.9
        + rp (float): reward or penalty consideration rate, default = 0.2
    """

    def __init__(self, epoch=10000, pop_size=100, n_groups=5, bpsp=0.5, mp=0.5, pp=0.9, rp=0.2, **kwargs):
        """
        参数调整:
        - 恢复n_groups为5，与MATLAB版本一致
        - 恢复mp为0.5，与MATLAB版本一致
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_groups = self.validator.check_int("n_groups", n_groups, [2, 20])
        self.bpsp = self.validator.check_float("bpsp", bpsp, (0, 1.0))
        self.mp = self.validator.check_float("mp", mp, (0, 1.0))
        self.pp = self.validator.check_float("pp", pp, (0, 1.0))
        self.rp = self.validator.check_float("rp", rp, (0, 1.0))
        self.n_followers = pop_size - n_groups
        self.set_parameters(["epoch", "pop_size", "n_groups", "bpsp", "mp", "pp", "rp"])
        self.sort_flag = False

        # 精英解历史
        self.elite_history = []
        self.max_history_size = 3

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

    def get_levy_flight_step(self, beta=1.5, size=None):  # 使用最优beta值1.5
        """
        Get levy flight step size - 添加数值稳定性保护

        Args:
            beta: The power law index (default: 1.5) - 理论最优值
            size: Output size (default is self.problem.n_dims)
        """
        if size is None:
            size = self.problem.n_dims

        try:
            # Mantegna's algorithm for Levy flight
            sigma_num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            sigma_den = math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)

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

            # 限制步长，避免极端值，减小范围促进收敛
            step = np.clip(step, -2.0, 2.0)

            return step
        except (OverflowError, ZeroDivisionError, ValueError):
            # 出错时返回一个小随机值
            return np.random.uniform(-0.5, 0.5, size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        try:
            # 计算自适应奇迹率，与MATLAB代码完全一致
            # 原始MATLAB公式: MP=(1*rand)*(1-(it/MaxIteration*2))*(1*rand())
            mp_current = np.random.random() * (1 - (epoch / self.epoch * 2)) * np.random.random()

            # 精英解记忆更新 - 只在提升时更新
            if hasattr(self, 'g_best') and (len(self.elite_history) == 0 or
                                            (
                                            self.g_best[self.ID_TAR][self.ID_FIT] < self.elite_history[-1][self.ID_TAR][
                                                self.ID_FIT] if len(self.elite_history) > 0 else True)):
                if len(self.elite_history) >= self.max_history_size:
                    self.elite_history.pop(0)  # 移除最旧的记录
                # 存储全局最优解（完整解，包含位置和适应度）
                self.elite_history.append(self.g_best.copy())

            # 对种群进行排序并找到最佳解(Leader)
            BP = self.pop.copy()  # 创建种群副本
            BP = sorted(BP, key=lambda x: x[self.ID_TAR][self.ID_FIT])
            leader = BP[0].copy()

            # 创建新的追随者 (NewFollower) - 与MATLAB一致
            new_follower = self.create_solution(self.problem.lb, self.problem.ub)

            # 信仰概况考虑率 (BPSP) - 与MATLAB保持一致
            if np.random.random() <= self.bpsp:
                # 完全按照MATLAB逻辑实现
                random_dim = np.random.randint(0, self.problem.n_dims)
                random_agent_idx = np.random.randint(0, self.pop_size)
                random_belief_idx = np.random.randint(0, self.problem.n_dims)

                # 信仰概况交换
                new_follower[self.ID_POS][random_dim] = BP[random_agent_idx][self.ID_POS][random_belief_idx]
                new_follower[self.ID_TAR] = self.get_target_wrapper(new_follower[self.ID_POS])

            # 奇迹操作阶段：探索或利用
            if np.random.random() <= mp_current:  # 探索阶段 - 奇迹操作符 (Miracle Operator)
                # 遍历每个个体
                for i in range(self.pop_size):
                    pos_current = BP[i][self.ID_POS].copy()
                    old_pos = pos_current.copy()

                    # 与MATLAB完全一致的实现
                    if np.random.random() <= 0.5:
                        pos_current = pos_current * np.cos(np.pi / 2) * (
                                    np.random.random() - np.cos(np.random.random()))
                    else:
                        pos_current = pos_current + np.random.random() * (
                                    pos_current - np.round(1 ** np.random.random()) * pos_current)

                    # 检查是否有无穷值或NaN值，如果有则还原
                    pos_current = np.where(np.isnan(pos_current) | np.isinf(pos_current), old_pos, pos_current)

                    # 边界处理 - 修改为严格基于MATLAB的方式
                    pos_current = np.minimum(np.maximum(pos_current, self.problem.lb), self.problem.ub)

                    # 评估新解
                    target_current = self.get_target_wrapper(pos_current)

                    # 只有在更好时才更新，与MATLAB一致
                    if target_current[self.ID_FIT] < BP[i][self.ID_TAR][self.ID_FIT]:
                        BP[i] = [pos_current, target_current]
            else:  # 利用阶段 - 传教操作符 (proselytism Operator)
                # 设置新追随者的信念，与MATLAB一致
                new_follower[self.ID_POS] = leader[self.ID_POS] * (np.random.random() - np.sin(np.random.random()))
                new_follower[self.ID_POS] = np.minimum(np.maximum(new_follower[self.ID_POS], self.problem.lb),
                                                       self.problem.ub)
                new_follower[self.ID_TAR] = self.get_target_wrapper(new_follower[self.ID_POS])

                # 遍历每个个体
                for i in range(self.pop_size):
                    pos_current = BP[i][self.ID_POS].copy()
                    old_pos = pos_current.copy()

                    # 与MATLAB完全一致
                    if np.random.random() > (1 - mp_current):
                        mean_pos = np.mean(pos_current)
                        # 确保mean_pos不是NaN
                        if np.isnan(mean_pos):
                            mean_pos = 0.0

                        # 完全按照MATLAB逻辑实现
                        random_term = np.random.random() - 4 * np.sin(np.sin(3.14 * np.random.random()))
                        pos_current = (pos_current * 0.01) + (
                                    mean_pos * (1 - mp_current) + (1 - mean_pos) - random_term)
                    else:
                        # 与MATLAB完全一致
                        pos_current = leader[self.ID_POS] * (np.random.random() - np.cos(np.random.random()))

                    # 检查是否有无穷值或NaN值，如果有则还原
                    pos_current = np.where(np.isnan(pos_current) | np.isinf(pos_current), old_pos, pos_current)

                    # 边界处理 - 修改为严格基于MATLAB的方式
                    pos_current = np.minimum(np.maximum(pos_current, self.problem.lb), self.problem.ub)

                    # 评估新解
                    target_current = self.get_target_wrapper(pos_current)

                    # 只有在更好时才更新，与MATLAB一致
                    if target_current[self.ID_FIT] < BP[i][self.ID_TAR][self.ID_FIT]:
                        BP[i] = [pos_current, target_current]

            # 奖励或惩罚操作符 - 与MATLAB一致
            idx = np.random.randint(0, self.pop_size)
            pos_current = BP[idx][self.ID_POS].copy()
            old_pos = pos_current.copy()

            if np.random.random() >= self.rp:
                # 奖励操作符
                pos_current = pos_current * (1 - np.random.randn())
            else:
                # 惩罚操作符
                pos_current = pos_current * (1 + np.random.randn())

            # 检查是否有无穷值或NaN值，如果有则还原
            pos_current = np.where(np.isnan(pos_current) | np.isinf(pos_current), old_pos, pos_current)

            # 边界处理 - 修改为严格基于MATLAB的方式
            pos_current = np.minimum(np.maximum(pos_current, self.problem.lb), self.problem.ub)

            # 评估新解
            target_current = self.get_target_wrapper(pos_current)

            # 只有在更好时才更新，与MATLAB一致
            if target_current[self.ID_FIT] < BP[idx][self.ID_TAR][self.ID_FIT]:
                BP[idx] = [pos_current, target_current]

            # 重新排序BP
            BP = sorted(BP, key=lambda x: x[self.ID_TAR][self.ID_FIT])

            # 合并信仰概况和新追随者 - 与MATLAB一致
            if new_follower[self.ID_TAR][self.ID_FIT] < BP[-1][self.ID_TAR][self.ID_FIT]:
                BP[-1] = new_follower.copy()
                # 重新排序
                BP = sorted(BP, key=lambda x: x[self.ID_TAR][self.ID_FIT])

            # 更新全局最优解
            current_best = BP[0]
            if not hasattr(self, 'g_best') or current_best[self.ID_TAR][self.ID_FIT] < self.g_best[self.ID_TAR][
                self.ID_FIT]:
                self.g_best = current_best.copy()

            # 精英保留策略 - 确保最佳解不会丢失
            if hasattr(self, 'g_best'):
                # 如果最好解不是当前的全局最优，替换一个随机较差的解
                if BP[0][self.ID_TAR][self.ID_FIT] > self.g_best[self.ID_TAR][self.ID_FIT]:
                    idx = np.random.randint(int(self.pop_size / 2), self.pop_size)  # 选择后半部分的一个解
                    BP[idx] = self.g_best.copy()

            # 替换操作符 - 简化实现，本质上是交换操作
            if epoch % 5 == 0:  # 每隔几次迭代执行一次替换操作
                missionaries = BP[:self.n_groups]  # 前n_groups个解作为missionaries
                followers = BP[self.n_groups:]  # 剩余的解作为followers

                k = np.random.randint(0, min(self.n_groups, len(missionaries)))  # 随机选择一个组
                if len(followers) > 0:
                    f_idx = np.random.randint(0, len(followers))  # 随机选择一个follower
                    # 交换missionary和follower
                    temp = missionaries[k].copy()
                    missionaries[k] = followers[f_idx].copy()
                    followers[f_idx] = temp

                    # 更新BP
                    BP[:self.n_groups] = missionaries
                    BP[self.n_groups:] = followers

            # 更新种群
            self.pop = BP

        except Exception as e:
            print(f"DRA evolution failed: {e}")
            # 如果整个进化过程失败，生成新的随机种群，但保留全局最优解
            BP = []

            # 保留全局最优解
            if hasattr(self, 'g_best'):
                BP.append(self.g_best.copy())

            # 填充剩余位置
            while len(BP) < self.pop_size:
                pos = self.generate_position(self.problem.lb, self.problem.ub)
                pos = self.amend_position(pos, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos)
                BP.append([pos, target])

            # 确保BP是排序的
            BP = sorted(BP, key=lambda x: x[self.ID_TAR][self.ID_FIT])

            # 更新种群
            self.pop = BP