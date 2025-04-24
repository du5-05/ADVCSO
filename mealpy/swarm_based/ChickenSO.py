import numpy as np
from scipy.stats import cauchy
from mealpy.optimizer import Optimizer


# Base CSO class with shared functionality
class BaseCSO(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.G = self.validator.check_int("G", G, [1, 1000])
        self.jixiaoliang = self.validator.check_float("jixiaoliang", jixiaoliang, (0, 1e-5))
        self.set_parameters(["epoch", "pop_size", "G", "jixiaoliang"])
        self.sort_flag = False

        # 设置默认初始多样性值
        self.initial_diversity = 1e-10

    def generate_position(self, lb, ub):
        return np.random.uniform(lb, ub)

    def generate_good_point_set(self, pop_size, dim, lb, ub):

        # 选择满足 p ≥ 2*dim + 3 的最小素数
        def primes(n):
            sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
            for i in range(1, int(n ** 0.5) // 3 + 1):
                if sieve[i]:
                    k = 3 * i + 1 | 1
                    sieve[3 * i + 1::k] = False
                    sieve[k::k] = False
            primes = np.r_[2, 3, (3 * np.nonzero(sieve)[0][1:] * 2 + 1)]
            return primes

        prime_list = primes(100 * dim)
        valid_primes = prime_list[prime_list >= 2 * dim + 3]
        p = valid_primes[0]  # 选择第一个满足条件的素数

        # 生成佳点集矩阵
        k = np.arange(1, pop_size + 1).reshape(-1, 1)  # 种群索引 [M x 1]
        Ind = np.arange(1, dim + 1)  # 维度索引 [1 x N]
        tmp = 2 * np.cos((2 * np.pi * Ind) / p)  # 余弦调制 [1 x N]
        r = k * tmp  # 矩阵乘法 [M x N]
        Good_points = np.mod(r, 1)  # 取模归一化

        # 缩放到实际边界
        Positions = np.zeros((pop_size, dim))
        for j in range(dim):
            Positions[:, j] = lb[j] + (ub[j] - lb[j]) * Good_points[:, j]

        return Positions

    def update_global_best(self):
        """更新全局最优"""
        current_best = np.min(self.local_best_fit)
        if current_best < self.gbest:
            self.gbest = current_best
            self.gbest_pos = self.local_best_pos[np.argmin(self.local_best_fit)].copy()


# 改进1：仅使用佳点集初始化的CSO
class CSO_GPS(BaseCSO):
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(epoch, pop_size, G, jixiaoliang, **kwargs)
        self.name = "CSO with Good Point Set (GPS)"

    def generate_position(self, lb, ub):
        """使用佳点集初始化"""
        return self.generate_good_point_set(1, self.problem.n_dims, lb, ub)[0]

    def initialize_variables(self):
        """使用佳点集初始化种群"""
        super().initialize_variables()

        # 使用佳点集初始化种群
        gps_pop = self.generate_good_point_set(self.pop_size, self.problem.n_dims, self.problem.lb, self.problem.ub)
        self.pop = [self.create_solution(pos=gps_pop[i]) for i in range(self.pop_size)]

        # 初始化局部最优和全局最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]
        self.gbest = min(self.local_best_fit)
        best_idx = np.argmin(self.local_best_fit)
        self.gbest_pos = self.pop[best_idx][self.ID_POS].copy()

        # 对前10%精英个体施加扰动
        elite_num = max(1, int(0.1 * self.pop_size))
        sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
        for i in sorted_indices[:elite_num]:
            noise = 0.05 * (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
            self.pop[i][self.ID_POS] = np.clip(self.pop[i][self.ID_POS] + noise, self.problem.lb, self.problem.ub)

        # 更新局部最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]

        # 固定角色分配比例
        self.rNum = int(self.pop_size * 0.2)  # 公鸡比例20%
        self.hNum = int(self.pop_size * 0.3)  # 母鸡比例30%
        self.mNum = self.pop_size - self.rNum - self.hNum  # 小鸡比例50%

        # 初始化角色相关变量
        self.roosters = []
        self.hens = []
        self.chicks = []
        self.mate = None
        self.motherLib = None
        self.mother = None
        self.FL = None

        # 保存初始种群位置
        self.initial_positions = np.array([agent[self.ID_POS] for agent in self.pop])

        # 记录初始多样性
        self.initial_diversity = np.std(self.local_best_fit)
        if self.initial_diversity == 0:
            self.initial_diversity = 1e-10

    def evolve(self, epoch):
        """原始CSO进化过程"""
        # 固定周期分配角色
        if epoch % self.G == 0:
            # 按适应度排序
            sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])

            # 分配角色
            self.roosters = sorted_indices[:self.rNum].tolist()
            self.hens = sorted_indices[self.rNum:self.rNum + self.hNum].tolist()
            self.chicks = sorted_indices[self.rNum + self.hNum:].tolist()

            # 配偶和母鸡分配
            self.mate = np.random.randint(0, self.rNum, self.hNum)
            self.motherLib = np.random.choice(self.hens, size=self.mNum, replace=True)
            self.mother = np.random.choice(self.motherLib, size=self.mNum, replace=True)
            self.FL = 0.4 * np.random.rand(self.mNum) + 0.5

        # 更新公鸡位置
        for i in range(self.rNum):
            idx = self.roosters[i]
            another_idx = np.random.choice([r for r in self.roosters if r != idx])

            # 计算位置更新参数
            if self.local_best_fit[idx] <= self.local_best_fit[another_idx]:
                temp_sigma = 1.0
            else:
                temp_sigma = np.exp((self.local_best_fit[another_idx] - self.local_best_fit[idx]) /
                                    (abs(self.local_best_fit[idx]) + self.jixiaoliang))

            # 更新位置
            new_pos = self.local_best_pos[idx] * (1 + temp_sigma * np.random.randn(self.problem.n_dims))
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_pos.copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新母鸡位置
        for i in range(self.hNum):
            idx = self.hens[i]
            mate_idx = self.roosters[self.mate[i]]
            candidates = [k for k in self.roosters + self.hens if k != idx and k != mate_idx]
            other_idx = np.random.choice(candidates) if candidates else idx

            # 计算位置更新参数
            c1 = np.exp((self.local_best_fit[idx] - self.local_best_fit[mate_idx]) /
                        (abs(self.local_best_fit[idx]) + self.jixiaoliang))
            c2 = np.exp(self.local_best_fit[other_idx] - self.local_best_fit[idx])

            # 更新位置
            rand = np.random.rand(self.problem.n_dims)
            new_pos = self.local_best_pos[idx] + \
                      (self.local_best_pos[mate_idx] - self.local_best_pos[idx]) * c1 * rand + \
                      (self.local_best_pos[other_idx] - self.local_best_pos[idx]) * c2 * rand
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_pos.copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新小鸡位置
        for i in range(self.mNum):
            idx = self.chicks[i]
            mother_idx = self.mother[i]
            fl = self.FL[i]

            # 更新位置
            new_pos = self.local_best_pos[idx] + fl * (self.local_best_pos[mother_idx] - self.local_best_pos[idx])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_pos.copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新全局最优
        self.update_global_best()


# 改进2：仅使用动态角色分配的CSO
class CSO_DRA(BaseCSO):
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(epoch, pop_size, G, jixiaoliang, **kwargs)
        self.name = "CSO with Dynamic Role Assignment (DRA)"

    def generate_position(self, lb, ub):
        return np.random.uniform(lb, ub)

    def initialize_variables(self):
        super().initialize_variables()
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)  # 创建种群

        # 初始化局部最优和全局最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]
        self.gbest = min(self.local_best_fit)
        best_idx = np.argmin(self.local_best_fit)
        self.gbest_pos = self.pop[best_idx][self.ID_POS].copy()

        # 初始角色分配（动态调整）
        self.rNum = int(self.pop_size * 0.3)  # 初始公鸡比例30%
        self.hNum = int(self.pop_size * 0.3)  # 初始母鸡比例30%
        self.mNum = self.pop_size - self.rNum - self.hNum  # 小鸡比例40%

        # 初始化角色相关变量
        self.roosters = []
        self.hens = []
        self.chicks = []
        self.mate = None
        self.motherLib = None
        self.mother = None
        self.FL = None

        # 保存初始种群位置
        self.initial_positions = np.array([agent[self.ID_POS] for agent in self.pop])

        # 计算初始多样性
        self.initial_diversity = np.std(self.local_best_fit)
        if self.initial_diversity == 0:
            self.initial_diversity = 1e-10

    def evolve(self, epoch):
        """使用动态角色分配策略的CSO进化过程"""
        # 计算种群多样性
        current_diversity = np.std([agent[self.ID_TAR][0] for agent in self.pop])
        if current_diversity == 0 or np.isnan(current_diversity):
            current_diversity = 1e-10

        # 计算多样性比例
        diversity_ratio = np.nan_to_num(current_diversity / self.initial_diversity)

        # 自适应G参数
        decay = 0.5 * (1 + np.cos(np.pi * epoch / self.epoch))
        G_current = max(5, int(self.G * decay))

        # 动态角色分配
        if epoch % G_current == 0:
            # 动态调整角色比例
            self.rNum = max(1, min(int(self.pop_size * (0.3 + 0.2 * (current_diversity / self.initial_diversity))),
                                   self.pop_size - 2))
            self.hNum = max(1, min(int(self.pop_size * (0.3 - 0.1 * diversity_ratio)), self.pop_size - self.rNum - 1))
            self.mNum = max(0, self.pop_size - self.rNum - self.hNum)

            # 分配角色
            sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
            self.roosters = sorted_indices[:self.rNum].tolist()
            self.hens = sorted_indices[self.rNum:self.rNum + self.hNum].tolist()
            self.chicks = sorted_indices[self.rNum + self.hNum:].tolist()

            # 配偶和母鸡分配
            top_roosters = self.roosters[:max(1, int(0.5 * self.rNum))]
            self.mate = np.random.choice(top_roosters, size=self.hNum, replace=True)
            self.motherLib = np.random.choice(self.hens, size=self.mNum, replace=True)
            self.mother = np.random.choice(self.motherLib, size=self.mNum, replace=True)
            self.FL = 0.4 * np.random.rand(self.mNum) + 0.5

        # 更新公鸡位置
        for i in range(self.rNum):
            if i >= len(self.roosters):
                break
            idx = self.roosters[i]
            another_idx = np.random.choice([k for k in self.roosters if k != idx])

            # 自适应学习因子
            alpha = 0.5 * (1 - np.cos(np.pi * epoch / self.epoch))

            # 计算位置更新参数
            if self.local_best_fit[idx] <= self.local_best_fit[another_idx]:
                temp_sigma = 1.0
            else:
                temp_sigma = np.exp((self.local_best_fit[another_idx] - self.local_best_fit[idx]) /
                                    (abs(self.local_best_fit[idx]) + self.jixiaoliang))

            # 更新位置
            new_pos = self.local_best_pos[idx] * (1 + alpha * temp_sigma * np.random.randn(self.problem.n_dims))
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新母鸡位置
        for i in range(self.hNum):
            idx = self.hens[i]
            mate_idx = self.mate[i]
            candidates = [k for k in self.roosters + self.hens if k != idx and k != mate_idx]
            other_idx = np.random.choice(candidates) if candidates else idx

            # 排名自适应参数
            rank_factor = 1.0 - (np.where(np.argsort([agent[self.ID_TAR][0] for agent in self.pop]) == idx)[0][
                                     0] / self.pop_size)

            # 计算位置更新参数
            c1 = np.exp((self.local_best_fit[idx] - self.local_best_fit[mate_idx]) /
                        (abs(self.local_best_fit[idx]) + self.jixiaoliang))
            c2 = rank_factor * np.exp(self.local_best_fit[other_idx] - self.local_best_fit[idx])

            # 更新位置
            new_pos = self.local_best_pos[idx] + \
                      (self.local_best_pos[mate_idx] - self.local_best_pos[idx]) * c1 * np.random.rand(
                self.problem.n_dims) + \
                      (self.local_best_pos[other_idx] - self.local_best_pos[idx]) * c2 * np.random.rand(
                self.problem.n_dims)
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新小鸡位置
        for i in range(self.mNum):
            idx = self.chicks[i]
            mother_idx = self.mother[i]
            fl = self.FL[i]

            # 更新位置
            new_pos = self.local_best_pos[idx] + fl * (self.local_best_pos[mother_idx] - self.local_best_pos[idx])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新全局最优
        self.update_global_best()


# 改进3：仅使用混合变异策略的CSO
class CSO_MUT(BaseCSO):
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(epoch, pop_size, G, jixiaoliang, **kwargs)
        self.name = "CSO with Mutation Strategies (MUT)"
        self.mutation_switch = 0.3  # 30%迭代次数后切换为高斯变异

    def generate_position(self, lb, ub):
        return np.random.uniform(lb, ub)

    def initialize_variables(self):
        super().initialize_variables()
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)  # 创建种群

        # 初始化局部最优和全局最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]
        self.gbest = min(self.local_best_fit)
        best_idx = np.argmin(self.local_best_fit)
        self.gbest_pos = self.pop[best_idx][self.ID_POS].copy()

        # 固定角色分配比例
        self.rNum = int(self.pop_size * 0.2)  # 公鸡比例20%
        self.hNum = int(self.pop_size * 0.3)  # 母鸡比例30%
        self.mNum = self.pop_size - self.rNum - self.hNum  # 小鸡比例50%

        # 初始化角色相关变量
        self.roosters = []
        self.hens = []
        self.chicks = []
        self.mate = None
        self.motherLib = None
        self.mother = None
        self.FL = None

        # 保存初始种群位置
        self.initial_positions = np.array([agent[self.ID_POS] for agent in self.pop])

        # 计算初始多样性
        self.initial_diversity = np.std(self.local_best_fit)
        if self.initial_diversity == 0:
            self.initial_diversity = 1e-10

    def evolve(self, epoch):
        """使用混合变异策略的CSO进化过程"""
        # 计算种群多样性
        current_diversity = np.std([agent[self.ID_TAR][0] for agent in self.pop])
        if current_diversity == 0 or np.isnan(current_diversity):
            current_diversity = 1e-10

        # 计算多样性比例
        diversity_ratio = np.nan_to_num(current_diversity / self.initial_diversity)
        self.mutation_switch = 0.3 + 0.4 * (1 - diversity_ratio)  # 当多样性低时，延后切换

        # 固定周期分配角色
        if epoch % self.G == 0:
            # 按适应度排序
            sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])

            # 分配角色
            self.roosters = sorted_indices[:self.rNum].tolist()
            self.hens = sorted_indices[self.rNum:self.rNum + self.hNum].tolist()
            self.chicks = sorted_indices[self.rNum + self.hNum:].tolist()

            # 配偶和母鸡分配
            self.mate = np.random.randint(0, self.rNum, self.hNum)
            self.motherLib = np.random.choice(self.hens, size=self.mNum, replace=True)
            self.mother = np.random.choice(self.motherLib, size=self.mNum, replace=True)
            self.FL = 0.4 * np.random.rand(self.mNum) + 0.5

        # 更新公鸡位置
        for i in range(self.rNum):
            if i >= len(self.roosters):  # 边界保护
                break
            idx = self.roosters[i]
            another_idx = np.random.choice([k for k in self.roosters if k != idx])

            # 计算位置更新参数
            # 自适应学习因子（余弦衰减）
            alpha = 0.5 * (1 - np.cos(np.pi * epoch / self.epoch))  # 公鸡学习因子随迭代次数非线性衰减
            if self.local_best_fit[idx] > self.local_best_fit[another_idx]:
                temp_sigma = np.exp((self.local_best_fit[another_idx] - self.local_best_fit[idx]) /
                                    (abs(self.local_best_fit[idx]) + self.jixiaoliang))
            else:
                temp_sigma = 1.0

            # 更新位置
            new_pos = self.local_best_pos[idx] * (1 + alpha * temp_sigma * np.random.randn(self.problem.n_dims))
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 混合变异
            if np.random.rand() < 0.3:  # 30%概率触发变异
                if epoch < self.mutation_switch * self.epoch:
                    # 前期柯西变异
                    mutation = np.random.standard_cauchy(self.problem.n_dims)
                else:
                    # 后期高斯变异
                    mutation = 0.5 * np.random.randn(self.problem.n_dims)
                mutated_pos = new_agent[self.ID_POS] + 0.1 * (self.problem.ub - self.problem.lb) * mutation
                mutated_agent = self.create_solution(
                    pos=self.amend_position(mutated_pos, self.problem.lb, self.problem.ub))
                if mutated_agent[self.ID_TAR][0] < new_agent[self.ID_TAR][0]:
                    new_agent = mutated_agent

            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新母鸡位置
        for i in range(self.hNum):
            idx = self.hens[i]
            mate_idx = self.mate[i]
            candidates = [k for k in self.roosters + self.hens if k != idx and k != mate_idx]
            other_idx = np.random.choice(candidates) if candidates else idx

            # 排名自适应参数 越接近1，排名越靠前 优秀个体探索，较差个体跟随
            rank_factor = 1.0 - (np.where(np.argsort([agent[self.ID_TAR][0] for agent in self.pop]) == idx)[0][0]
                                 / self.pop_size)  # 母鸡跟随因子基于适应度排名动态调整
            c1 = np.exp((self.local_best_fit[idx] - self.local_best_fit[mate_idx]) / (
                    abs(self.local_best_fit[idx]) + self.jixiaoliang))
            c2 = rank_factor * np.exp(self.local_best_fit[other_idx] - self.local_best_fit[idx])

            # 更新位置
            new_pos = self.local_best_pos[idx] + \
                      (self.local_best_pos[mate_idx] - self.local_best_pos[idx]) * c1 * np.random.rand(
                self.problem.n_dims) + \
                      (self.local_best_pos[other_idx] - self.local_best_pos[idx]) * c2 * np.random.rand(
                self.problem.n_dims)
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 维度学习策略
            if self.gbest_pos is not None:  # 防止None访问
                for d in np.random.choice(self.problem.n_dims, size=int(0.2 * self.problem.n_dims),
                                          replace=False):  # 对每个维度独立进行精英解维度值学习
                    new_agent[self.ID_POS][d] = (self.gbest_pos[d] +
                                                 0.05 * (self.problem.ub[d] - self.problem.lb[d]) * np.random.randn())
                new_agent[self.ID_POS] = self.amend_position(new_agent[self.ID_POS], self.problem.lb, self.problem.ub)
                new_agent[self.ID_TAR] = self.get_target_wrapper(new_agent[self.ID_POS])

            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新小鸡位置
        for i in range(self.mNum):
            idx = self.chicks[i]
            mother_idx = self.mother[i]
            fl = self.FL[i]

            new_pos = self.local_best_pos[idx] + fl * (self.local_best_pos[mother_idx] - self.local_best_pos[idx])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 改进3：精英维度继承 随机选择两个维度，用全局最优解替换当前小鸡的位置
            if np.random.rand() < 0.2 and self.gbest_pos is not None:
                elite_dims = np.random.choice(self.problem.n_dims, size=2, replace=False)
                new_agent[self.ID_POS][elite_dims[0]] = self.gbest_pos[elite_dims[0]]
                new_agent[self.ID_POS][elite_dims[1]] = self.gbest_pos[elite_dims[1]]
                new_agent[self.ID_POS] = self.amend_position(new_agent[self.ID_POS], self.problem.lb, self.problem.ub)
                new_agent[self.ID_TAR] = self.get_target_wrapper(new_agent[self.ID_POS])

            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新全局最优
        self.update_global_best()


# 原始CSO算法
class OriginalCSO(BaseCSO):
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(epoch, pop_size, G, jixiaoliang, **kwargs)
        self.name = "Original CSO"

    def generate_position(self, lb, ub):
        return np.random.uniform(lb, ub)

    def initialize_variables(self):
        super().initialize_variables()
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)  # 创建种群

        # 初始化局部最优和全局最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]
        self.gbest = min(self.local_best_fit)
        best_idx = np.argmin(self.local_best_fit)
        self.gbest_pos = self.pop[best_idx][self.ID_POS].copy()

        # 固定角色分配比例
        self.rNum = int(self.pop_size * 0.2)  # 公鸡比例20%
        self.hNum = int(self.pop_size * 0.3)  # 母鸡比例30%
        self.mNum = self.pop_size - self.rNum - self.hNum  # 小鸡比例50%

        # 初始化角色相关变量
        self.roosters = []
        self.hens = []
        self.chicks = []
        self.mate = None
        self.motherLib = None
        self.mother = None
        self.FL = None

        # 保存初始种群位置
        self.initial_positions = np.array([agent[self.ID_POS] for agent in self.pop])

    def evolve(self, epoch):
        """原始CSO进化过程"""
        # 固定周期分配角色
        if epoch % self.G == 0:
            # 按适应度排序
            sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])

            # 分配角色
            self.roosters = sorted_indices[:self.rNum].tolist()
            self.hens = sorted_indices[self.rNum:self.rNum + self.hNum].tolist()
            self.chicks = sorted_indices[self.rNum + self.hNum:].tolist()

            # 配偶和母鸡分配
            self.mate = np.random.randint(0, self.rNum, self.hNum)
            self.motherLib = np.random.choice(self.hens, size=self.mNum, replace=True)
            self.mother = np.random.choice(self.motherLib, size=self.mNum, replace=True)
            self.FL = 0.4 * np.random.rand(self.mNum) + 0.5

        # 更新公鸡位置
        for i in range(self.rNum):
            idx = self.roosters[i]
            another_idx = np.random.choice([r for r in self.roosters if r != idx])

            # 计算位置更新参数
            if self.local_best_fit[idx] <= self.local_best_fit[another_idx]:
                temp_sigma = 1.0
            else:
                temp_sigma = np.exp((self.local_best_fit[another_idx] - self.local_best_fit[idx]) /
                                    (abs(self.local_best_fit[idx]) + self.jixiaoliang))

            # 更新位置
            new_pos = self.local_best_pos[idx] * (1 + temp_sigma * np.random.randn(self.problem.n_dims))
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_pos.copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新母鸡位置
        for i in range(self.hNum):
            idx = self.hens[i]
            mate_idx = self.roosters[self.mate[i]]
            candidates = [k for k in self.roosters + self.hens if k != idx and k != mate_idx]
            other_idx = np.random.choice(candidates) if candidates else idx

            # 计算位置更新参数
            c1 = np.exp((self.local_best_fit[idx] - self.local_best_fit[mate_idx]) /
                        (abs(self.local_best_fit[idx]) + self.jixiaoliang))
            c2 = np.exp(self.local_best_fit[other_idx] - self.local_best_fit[idx])

            # 更新位置
            rand = np.random.rand(self.problem.n_dims)
            new_pos = self.local_best_pos[idx] + \
                      (self.local_best_pos[mate_idx] - self.local_best_pos[idx]) * c1 * rand + \
                      (self.local_best_pos[other_idx] - self.local_best_pos[idx]) * c2 * rand
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_pos.copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新小鸡位置
        for i in range(self.mNum):
            idx = self.chicks[i]
            mother_idx = self.mother[i]
            fl = self.FL[i]

            # 更新位置
            new_pos = self.local_best_pos[idx] + fl * (self.local_best_pos[mother_idx] - self.local_best_pos[idx])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_pos.copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新全局最优
        self.update_global_best()


# 完整ADVCSO算法
class ADVCSO(BaseCSO):
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(epoch, pop_size, G, jixiaoliang, **kwargs)
        self.name = "Advanced CSO (ADVCSO)"
        self.mutation_switch = 0.3  # 30%迭代次数后切换为高斯变异

    def generate_position(self, lb, ub):
        """ADVCSO使用佳点集初始化"""
        return self.generate_good_point_set(1, self.problem.n_dims, lb, ub)[0]

    def initialize_variables(self):
        """ADVCSO初始化过程，使用佳点集和精英扰动"""
        super().initialize_variables()

        # 使用佳点集初始化种群
        gps_pop = self.generate_good_point_set(self.pop_size, self.problem.n_dims, self.problem.lb, self.problem.ub)
        self.pop = [self.create_solution(pos=gps_pop[i]) for i in range(self.pop_size)]

        # 初始化局部最优和全局最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]
        self.gbest = min(self.local_best_fit)
        best_idx = np.argmin(self.local_best_fit)
        self.gbest_pos = self.pop[best_idx][self.ID_POS].copy()

        # 对前10%精英个体施加扰动
        elite_num = max(1, int(0.1 * self.pop_size))
        sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
        for i in sorted_indices[:elite_num]:
            noise = 0.05 * (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
            self.pop[i][self.ID_POS] = np.clip(self.pop[i][self.ID_POS] + noise, self.problem.lb, self.problem.ub)

        # 更新局部最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]

        # 记录初始多样性
        self.initial_diversity = np.std(self.local_best_fit)
        if self.initial_diversity == 0:
            self.initial_diversity = 1e-10

        # 保存初始种群位置
        self.initial_positions = np.array([agent[self.ID_POS] for agent in self.pop])

        # 角色变量初始化
        self.roosters = []
        self.hens = []
        self.chicks = []
        self.mate = None
        self.motherLib = None
        self.mother = None
        self.FL = None

    def evolve(self, epoch):
        """ADVCSO进化过程，包含所有改进策略"""
        # 计算种群多样性
        current_diversity = np.std([agent[self.ID_TAR][0] for agent in self.pop])
        if current_diversity == 0 or np.isnan(current_diversity):
            current_diversity = 1e-10

        # 计算多样性比例
        diversity_ratio = np.nan_to_num(current_diversity / self.initial_diversity)
        self.mutation_switch = 0.3 + 0.4 * (1 - diversity_ratio)

        # 自适应G参数
        decay = 0.5 * (1 + np.cos(np.pi * epoch / self.epoch))
        G_current = max(5, int(self.G * decay))

        # 动态角色分配
        if epoch % G_current == 0:
            # 动态调整角色比例
            self.rNum = max(1, min(int(self.pop_size * (0.3 + 0.2 * (current_diversity / self.initial_diversity))),
                                   self.pop_size - 2))
            self.hNum = max(1, min(int(self.pop_size * (0.3 - 0.1 * diversity_ratio)), self.pop_size - self.rNum - 1))
            self.mNum = max(0, self.pop_size - self.rNum - self.hNum)

            # 分配角色
            sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
            self.roosters = sorted_indices[:self.rNum].tolist()
            self.hens = sorted_indices[self.rNum:self.rNum + self.hNum].tolist()
            self.chicks = sorted_indices[self.rNum + self.hNum:].tolist()

            # 配偶和母鸡分配
            top_roosters = self.roosters[:max(1, int(0.5 * self.rNum))]
            self.mate = np.random.choice(top_roosters, size=self.hNum, replace=True)
            self.motherLib = np.random.choice(self.hens, size=self.mNum, replace=True)
            self.mother = np.random.choice(self.motherLib, size=self.mNum, replace=True)
            self.FL = 0.4 * np.random.rand(self.mNum) + 0.5

        # 更新公鸡位置
        for i in range(self.rNum):
            if i >= len(self.roosters):
                break
            idx = self.roosters[i]
            another_idx = np.random.choice([k for k in self.roosters if k != idx])

            # 自适应学习因子
            alpha = 0.5 * (1 - np.cos(np.pi * epoch / self.epoch))

            # 计算位置更新参数
            if self.local_best_fit[idx] <= self.local_best_fit[another_idx]:
                temp_sigma = 1.0
            else:
                temp_sigma = np.exp((self.local_best_fit[another_idx] - self.local_best_fit[idx]) /
                                    (abs(self.local_best_fit[idx]) + self.jixiaoliang))

            # 更新位置
            new_pos = self.local_best_pos[idx] * (1 + alpha * temp_sigma * np.random.randn(self.problem.n_dims))
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 混合变异
            if np.random.rand() < 0.3:
                if epoch < self.mutation_switch * self.epoch:
                    # 前期柯西变异
                    mutation = np.random.standard_cauchy(self.problem.n_dims)
                else:
                    # 后期高斯变异
                    mutation = 0.5 * np.random.randn(self.problem.n_dims)
                mutated_pos = new_agent[self.ID_POS] + 0.1 * (self.problem.ub - self.problem.lb) * mutation
                mutated_agent = self.create_solution(
                    pos=self.amend_position(mutated_pos, self.problem.lb, self.problem.ub))
                if mutated_agent[self.ID_TAR][0] < new_agent[self.ID_TAR][0]:
                    new_agent = mutated_agent

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新母鸡位置
        for i in range(self.hNum):
            idx = self.hens[i]
            mate_idx = self.mate[i]
            candidates = [k for k in self.roosters + self.hens if k != idx and k != mate_idx]
            other_idx = np.random.choice(candidates) if candidates else idx

            # 排名自适应参数
            rank_factor = 1.0 - (np.where(np.argsort([agent[self.ID_TAR][0] for agent in self.pop]) == idx)[0][
                                     0] / self.pop_size)

            # 计算位置更新参数
            c1 = np.exp((self.local_best_fit[idx] - self.local_best_fit[mate_idx]) /
                        (abs(self.local_best_fit[idx]) + self.jixiaoliang))
            c2 = rank_factor * np.exp(self.local_best_fit[other_idx] - self.local_best_fit[idx])

            # 更新位置
            new_pos = self.local_best_pos[idx] + \
                      (self.local_best_pos[mate_idx] - self.local_best_pos[idx]) * c1 * np.random.rand(
                self.problem.n_dims) + \
                      (self.local_best_pos[other_idx] - self.local_best_pos[idx]) * c2 * np.random.rand(
                self.problem.n_dims)
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 维度学习策略
            if self.gbest_pos is not None:
                for d in np.random.choice(self.problem.n_dims, size=int(0.2 * self.problem.n_dims), replace=False):
                    new_agent[self.ID_POS][d] = (self.gbest_pos[d] + 0.05 * (
                                self.problem.ub[d] - self.problem.lb[d]) * np.random.randn())
                new_agent[self.ID_POS] = self.amend_position(new_agent[self.ID_POS], self.problem.lb, self.problem.ub)
                new_agent[self.ID_TAR] = self.get_target_wrapper(new_agent[self.ID_POS])

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新小鸡位置
        for i in range(self.mNum):
            idx = self.chicks[i]
            mother_idx = self.mother[i]
            fl = self.FL[i]

            # 更新位置
            new_pos = self.local_best_pos[idx] + fl * (self.local_best_pos[mother_idx] - self.local_best_pos[idx])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 精英维度继承
            if np.random.rand() < 0.2 and self.gbest_pos is not None:
                elite_dims = np.random.choice(self.problem.n_dims, size=2, replace=False)
                new_agent[self.ID_POS][elite_dims[0]] = self.gbest_pos[elite_dims[0]]
                new_agent[self.ID_POS][elite_dims[1]] = self.gbest_pos[elite_dims[1]]
                new_agent[self.ID_POS] = self.amend_position(new_agent[self.ID_POS], self.problem.lb, self.problem.ub)
                new_agent[self.ID_TAR] = self.get_target_wrapper(new_agent[self.ID_POS])

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新全局最优
        self.update_global_best()

class CSO_GPS_DRA(BaseCSO): # 组合改进1+2：佳点集初始化 + 动态角色分配
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(epoch, pop_size, G, jixiaoliang, **kwargs)
        self.name = "CSO with GPS + DRA"

    def generate_position(self, lb, ub):
        """使用佳点集初始化"""
        return self.generate_good_point_set(1, self.problem.n_dims, lb, ub)[0]

    def initialize_variables(self):
        """使用佳点集初始化种群，动态角色分配"""
        super().initialize_variables()

        # 使用佳点集初始化种群
        gps_pop = self.generate_good_point_set(self.pop_size, self.problem.n_dims, self.problem.lb, self.problem.ub)
        self.pop = [self.create_solution(pos=gps_pop[i]) for i in range(self.pop_size)]

        # 初始化局部最优和全局最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]
        self.gbest = min(self.local_best_fit)
        best_idx = np.argmin(self.local_best_fit)
        self.gbest_pos = self.pop[best_idx][self.ID_POS].copy()

        # 对前10%精英个体施加扰动
        elite_num = max(1, int(0.1 * self.pop_size))
        sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
        for i in sorted_indices[:elite_num]:
            noise = 0.05 * (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
            self.pop[i][self.ID_POS] = np.clip(self.pop[i][self.ID_POS] + noise, self.problem.lb, self.problem.ub)

        # 更新局部最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]

        # 初始角色分配（动态调整）
        self.rNum = int(self.pop_size * 0.3)  # 初始公鸡比例30%
        self.hNum = int(self.pop_size * 0.3)  # 初始母鸡比例30%
        self.mNum = self.pop_size - self.rNum - self.hNum  # 小鸡比例40%

        # 初始化角色相关变量
        self.roosters = []
        self.hens = []
        self.chicks = []
        self.mate = None
        self.motherLib = None
        self.mother = None
        self.FL = None

        # 保存初始种群位置
        self.initial_positions = np.array([agent[self.ID_POS] for agent in self.pop])

        # 计算初始多样性
        self.initial_diversity = np.std(self.local_best_fit)
        if self.initial_diversity == 0:
            self.initial_diversity = 1e-10

    def evolve(self, epoch):
        """使用动态角色分配策略的CSO进化过程"""
        # 计算种群多样性
        current_diversity = np.std([agent[self.ID_TAR][0] for agent in self.pop])
        if current_diversity == 0 or np.isnan(current_diversity):
            current_diversity = 1e-10

        # 计算多样性比例
        diversity_ratio = np.nan_to_num(current_diversity / self.initial_diversity)

        # 自适应G参数
        decay = 0.5 * (1 + np.cos(np.pi * epoch / self.epoch))
        G_current = max(5, int(self.G * decay))

        # 动态角色分配
        if epoch % G_current == 0:
            # 动态调整角色比例
            self.rNum = max(1, min(int(self.pop_size * (0.3 + 0.2 * (current_diversity / self.initial_diversity))),
                                   self.pop_size - 2))
            self.hNum = max(1, min(int(self.pop_size * (0.3 - 0.1 * diversity_ratio)), self.pop_size - self.rNum - 1))
            self.mNum = max(0, self.pop_size - self.rNum - self.hNum)

            # 分配角色
            sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
            self.roosters = sorted_indices[:self.rNum].tolist()
            self.hens = sorted_indices[self.rNum:self.rNum + self.hNum].tolist()
            self.chicks = sorted_indices[self.rNum + self.hNum:].tolist()

            # 配偶和母鸡分配
            top_roosters = self.roosters[:max(1, int(0.5 * self.rNum))]
            self.mate = np.random.choice(top_roosters, size=self.hNum, replace=True)
            self.motherLib = np.random.choice(self.hens, size=self.mNum, replace=True)
            self.mother = np.random.choice(self.motherLib, size=self.mNum, replace=True)
            self.FL = 0.4 * np.random.rand(self.mNum) + 0.5

        # 更新公鸡位置
        for i in range(self.rNum):
            if i >= len(self.roosters):
                break
            idx = self.roosters[i]
            another_idx = np.random.choice([k for k in self.roosters if k != idx])

            # 自适应学习因子
            alpha = 0.5 * (1 - np.cos(np.pi * epoch / self.epoch))

            # 计算位置更新参数
            if self.local_best_fit[idx] <= self.local_best_fit[another_idx]:
                temp_sigma = 1.0
            else:
                temp_sigma = np.exp((self.local_best_fit[another_idx] - self.local_best_fit[idx]) /
                                    (abs(self.local_best_fit[idx]) + self.jixiaoliang))

            # 更新位置
            new_pos = self.local_best_pos[idx] * (1 + alpha * temp_sigma * np.random.randn(self.problem.n_dims))
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新母鸡位置
        for i in range(self.hNum):
            idx = self.hens[i]
            mate_idx = self.mate[i]
            candidates = [k for k in self.roosters + self.hens if k != idx and k != mate_idx]
            other_idx = np.random.choice(candidates) if candidates else idx

            # 排名自适应参数
            rank_factor = 1.0 - (np.where(np.argsort([agent[self.ID_TAR][0] for agent in self.pop]) == idx)[0][
                                     0] / self.pop_size)

            # 计算位置更新参数
            c1 = np.exp((self.local_best_fit[idx] - self.local_best_fit[mate_idx]) /
                        (abs(self.local_best_fit[idx]) + self.jixiaoliang))
            c2 = rank_factor * np.exp(self.local_best_fit[other_idx] - self.local_best_fit[idx])

            # 更新位置
            new_pos = self.local_best_pos[idx] + \
                      (self.local_best_pos[mate_idx] - self.local_best_pos[idx]) * c1 * np.random.rand(
                self.problem.n_dims) + \
                      (self.local_best_pos[other_idx] - self.local_best_pos[idx]) * c2 * np.random.rand(
                self.problem.n_dims)
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新小鸡位置
        for i in range(self.mNum):
            idx = self.chicks[i]
            mother_idx = self.mother[i]
            fl = self.FL[i]

            # 更新位置
            new_pos = self.local_best_pos[idx] + fl * (self.local_best_pos[mother_idx] - self.local_best_pos[idx])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 更新局部最优
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新全局最优
        self.update_global_best()

class CSO_GPS_MUT(BaseCSO): # 组合改进1+3：佳点集初始化 + 混合变异策略
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(epoch, pop_size, G, jixiaoliang, **kwargs)
        self.name = "CSO with GPS + MUT"
        self.mutation_switch = 0.3  # 30%迭代次数后切换为高斯变异

    def generate_position(self, lb, ub):
        """使用佳点集初始化"""
        return self.generate_good_point_set(1, self.problem.n_dims, lb, ub)[0]

    def initialize_variables(self):
        """使用佳点集初始化种群"""
        super().initialize_variables()

        # 使用佳点集初始化种群
        gps_pop = self.generate_good_point_set(self.pop_size, self.problem.n_dims, self.problem.lb, self.problem.ub)
        self.pop = [self.create_solution(pos=gps_pop[i]) for i in range(self.pop_size)]

        # 初始化局部最优和全局最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]
        self.gbest = min(self.local_best_fit)
        best_idx = np.argmin(self.local_best_fit)
        self.gbest_pos = self.pop[best_idx][self.ID_POS].copy()

        # 对前10%精英个体施加扰动
        elite_num = max(1, int(0.1 * self.pop_size))
        sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
        for i in sorted_indices[:elite_num]:
            noise = 0.05 * (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
            self.pop[i][self.ID_POS] = np.clip(self.pop[i][self.ID_POS] + noise, self.problem.lb, self.problem.ub)

        # 更新局部最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]

        # 固定角色分配比例
        self.rNum = int(self.pop_size * 0.2)  # 公鸡比例20%
        self.hNum = int(self.pop_size * 0.3)  # 母鸡比例30%
        self.mNum = self.pop_size - self.rNum - self.hNum  # 小鸡比例50%

        # 初始化角色相关变量
        self.roosters = []
        self.hens = []
        self.chicks = []
        self.mate = None
        self.motherLib = None
        self.mother = None
        self.FL = None

        # 保存初始种群位置
        self.initial_positions = np.array([agent[self.ID_POS] for agent in self.pop])

        # 计算初始多样性
        self.initial_diversity = np.std(self.local_best_fit)
        if self.initial_diversity == 0:
            self.initial_diversity = 1e-10

    def evolve(self, epoch):
        """使用混合变异策略的CSO进化过程"""
        # 计算种群多样性
        current_diversity = np.std([agent[self.ID_TAR][0] for agent in self.pop])
        if current_diversity == 0 or np.isnan(current_diversity):
            current_diversity = 1e-10

        # 计算多样性比例
        diversity_ratio = np.nan_to_num(current_diversity / self.initial_diversity)
        self.mutation_switch = 0.3 + 0.4 * (1 - diversity_ratio)  # 当多样性低时，延后切换

        # 固定周期分配角色
        if epoch % self.G == 0:
            # 按适应度排序
            sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])

            # 分配角色
            self.roosters = sorted_indices[:self.rNum].tolist()
            self.hens = sorted_indices[self.rNum:self.rNum + self.hNum].tolist()
            self.chicks = sorted_indices[self.rNum + self.hNum:].tolist()

            # 配偶和母鸡分配
            self.mate = np.random.randint(0, self.rNum, self.hNum)
            self.motherLib = np.random.choice(self.hens, size=self.mNum, replace=True)
            self.mother = np.random.choice(self.motherLib, size=self.mNum, replace=True)
            self.FL = 0.4 * np.random.rand(self.mNum) + 0.5

            # 更新公鸡位置
            for i in range(self.rNum):
                if i >= len(self.roosters):  # 边界保护
                    break
                idx = self.roosters[i]
                another_idx = np.random.choice([k for k in self.roosters if k != idx])

                # 计算位置更新参数
                # 自适应学习因子（余弦衰减）
                alpha = 0.5 * (1 - np.cos(np.pi * epoch / self.epoch))  # 公鸡学习因子随迭代次数非线性衰减
                if self.local_best_fit[idx] > self.local_best_fit[another_idx]:
                    temp_sigma = np.exp((self.local_best_fit[another_idx] - self.local_best_fit[idx]) /
                                        (abs(self.local_best_fit[idx]) + self.jixiaoliang))
                else:
                    temp_sigma = 1.0

                # 更新位置
                new_pos = self.local_best_pos[idx] * (1 + alpha * temp_sigma * np.random.randn(self.problem.n_dims))
                new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
                new_agent = self.create_solution(pos=new_pos)

                # 混合变异
                if np.random.rand() < 0.3:  # 30%概率触发变异
                    if epoch < self.mutation_switch * self.epoch:
                        # 前期柯西变异
                        mutation = np.random.standard_cauchy(self.problem.n_dims)
                    else:
                        # 后期高斯变异
                        mutation = 0.5 * np.random.randn(self.problem.n_dims)
                    mutated_pos = new_agent[self.ID_POS] + 0.1 * (self.problem.ub - self.problem.lb) * mutation
                    mutated_agent = self.create_solution(
                        pos=self.amend_position(mutated_pos, self.problem.lb, self.problem.ub))
                    if mutated_agent[self.ID_TAR][0] < new_agent[self.ID_TAR][0]:
                        new_agent = mutated_agent

                if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                    self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                    self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
                self.pop[idx] = new_agent

            # 更新母鸡位置
            for i in range(self.hNum):
                idx = self.hens[i]
                mate_idx = self.mate[i]
                candidates = [k for k in self.roosters + self.hens if k != idx and k != mate_idx]
                other_idx = np.random.choice(candidates) if candidates else idx

                # 排名自适应参数 越接近1，排名越靠前 优秀个体探索，较差个体跟随
                rank_factor = 1.0 - (np.where(np.argsort([agent[self.ID_TAR][0] for agent in self.pop]) == idx)[0][0]
                                     / self.pop_size)  # 母鸡跟随因子基于适应度排名动态调整
                c1 = np.exp((self.local_best_fit[idx] - self.local_best_fit[mate_idx]) / (
                        abs(self.local_best_fit[idx]) + self.jixiaoliang))
                c2 = rank_factor * np.exp(self.local_best_fit[other_idx] - self.local_best_fit[idx])

                # 更新位置
                new_pos = self.local_best_pos[idx] + \
                          (self.local_best_pos[mate_idx] - self.local_best_pos[idx]) * c1 * np.random.rand(
                    self.problem.n_dims) + \
                          (self.local_best_pos[other_idx] - self.local_best_pos[idx]) * c2 * np.random.rand(
                    self.problem.n_dims)
                new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
                new_agent = self.create_solution(pos=new_pos)

                # 维度学习策略
                if self.gbest_pos is not None:  # 防止None访问
                    for d in np.random.choice(self.problem.n_dims, size=int(0.2 * self.problem.n_dims),
                                              replace=False):  # 对每个维度独立进行精英解维度值学习
                        new_agent[self.ID_POS][d] = (self.gbest_pos[d] +
                                                     0.05 * (self.problem.ub[d] - self.problem.lb[
                                    d]) * np.random.randn())
                    new_agent[self.ID_POS] = self.amend_position(new_agent[self.ID_POS], self.problem.lb,
                                                                 self.problem.ub)
                    new_agent[self.ID_TAR] = self.get_target_wrapper(new_agent[self.ID_POS])

                if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                    self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                    self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
                self.pop[idx] = new_agent

            # 更新小鸡位置
            for i in range(self.mNum):
                idx = self.chicks[i]
                mother_idx = self.mother[i]
                fl = self.FL[i]

                new_pos = self.local_best_pos[idx] + fl * (self.local_best_pos[mother_idx] - self.local_best_pos[idx])
                new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
                new_agent = self.create_solution(pos=new_pos)

                # 改进3：精英维度继承 随机选择两个维度，用全局最优解替换当前小鸡的位置
                if np.random.rand() < 0.2 and self.gbest_pos is not None:
                    elite_dims = np.random.choice(self.problem.n_dims, size=2, replace=False)
                    new_agent[self.ID_POS][elite_dims[0]] = self.gbest_pos[elite_dims[0]]
                    new_agent[self.ID_POS][elite_dims[1]] = self.gbest_pos[elite_dims[1]]
                    new_agent[self.ID_POS] = self.amend_position(new_agent[self.ID_POS], self.problem.lb,
                                                                 self.problem.ub)
                    new_agent[self.ID_TAR] = self.get_target_wrapper(new_agent[self.ID_POS])

                if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                    self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                    self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
                self.pop[idx] = new_agent

            # 更新全局最优
            self.update_global_best()

class CSO_DRA_MUT(BaseCSO): # 组合改进2+3：动态角色分配 + 混合变异策略
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(epoch, pop_size, G, jixiaoliang, **kwargs)
        self.name = "CSO with DRA + MUT"
        self.mutation_switch = 0.3  # 30%迭代次数后切换为高斯变异

    def generate_position(self, lb, ub):
        return np.random.uniform(lb, ub)

    def initialize_variables(self):
        super().initialize_variables()
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)  # 创建种群

        # 初始化局部最优和全局最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]
        self.gbest = min(self.local_best_fit)
        best_idx = np.argmin(self.local_best_fit)
        self.gbest_pos = self.pop[best_idx][self.ID_POS].copy()

        # 初始角色分配（动态调整）
        self.rNum = int(self.pop_size * 0.3)  # 初始公鸡比例30%
        self.hNum = int(self.pop_size * 0.3)  # 初始母鸡比例30%
        self.mNum = self.pop_size - self.rNum - self.hNum  # 小鸡比例40%

        # 初始化角色相关变量
        self.roosters = []
        self.hens = []
        self.chicks = []
        self.mate = None
        self.motherLib = None
        self.mother = None
        self.FL = None

        # 保存初始种群位置
        self.initial_positions = np.array([agent[self.ID_POS] for agent in self.pop])

        # 计算初始多样性
        self.initial_diversity = np.std(self.local_best_fit)
        if self.initial_diversity == 0:
            self.initial_diversity = 1e-10

    def evolve(self, epoch):
        """结合动态角色分配和混合变异策略的CSO进化过程"""
        # 计算种群多样性
        current_diversity = np.std([agent[self.ID_TAR][0] for agent in self.pop])
        if current_diversity == 0 or np.isnan(current_diversity):
            current_diversity = 1e-10

        # 计算多样性比例
        diversity_ratio = np.nan_to_num(current_diversity / self.initial_diversity)
        self.mutation_switch = 0.3 + 0.4 * (1 - diversity_ratio)  # 当多样性低时，延后切换

        # 自适应G参数
        decay = 0.5 * (1 + np.cos(np.pi * epoch / self.epoch))
        G_current = max(5, int(self.G * decay))

        # 动态角色分配
        if epoch % G_current == 0:
            # 动态调整角色比例
            self.rNum = max(1, min(int(self.pop_size * (0.3 + 0.2 * (current_diversity / self.initial_diversity))),
                                   self.pop_size - 2))
            self.hNum = max(1, min(int(self.pop_size * (0.3 - 0.1 * diversity_ratio)), self.pop_size - self.rNum - 1))
            self.mNum = max(0, self.pop_size - self.rNum - self.hNum)

            # 分配角色
            sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
            self.roosters = sorted_indices[:self.rNum].tolist()
            self.hens = sorted_indices[self.rNum:self.rNum + self.hNum].tolist()
            self.chicks = sorted_indices[self.rNum + self.hNum:].tolist()

            # 配偶和母鸡分配
            top_roosters = self.roosters[:max(1, int(0.5 * self.rNum))]
            self.mate = np.random.choice(top_roosters, size=self.hNum, replace=True)
            self.motherLib = np.random.choice(self.hens, size=self.mNum, replace=True)
            self.mother = np.random.choice(self.motherLib, size=self.mNum, replace=True)
            self.FL = 0.4 * np.random.rand(self.mNum) + 0.5

            # 更新公鸡位置
            for i in range(self.rNum):
                if i >= len(self.roosters):  # 边界保护
                    break
                idx = self.roosters[i]
                another_idx = np.random.choice([k for k in self.roosters if k != idx])

                # 计算位置更新参数
                # 自适应学习因子（余弦衰减）
                alpha = 0.5 * (1 - np.cos(np.pi * epoch / self.epoch))  # 公鸡学习因子随迭代次数非线性衰减
                if self.local_best_fit[idx] > self.local_best_fit[another_idx]:
                    temp_sigma = np.exp((self.local_best_fit[another_idx] - self.local_best_fit[idx]) /
                                        (abs(self.local_best_fit[idx]) + self.jixiaoliang))
                else:
                    temp_sigma = 1.0

                # 更新位置
                new_pos = self.local_best_pos[idx] * (1 + alpha * temp_sigma * np.random.randn(self.problem.n_dims))
                new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
                new_agent = self.create_solution(pos=new_pos)

                # 混合变异
                if np.random.rand() < 0.3:  # 30%概率触发变异
                    if epoch < self.mutation_switch * self.epoch:
                        # 前期柯西变异
                        mutation = np.random.standard_cauchy(self.problem.n_dims)
                    else:
                        # 后期高斯变异
                        mutation = 0.5 * np.random.randn(self.problem.n_dims)
                    mutated_pos = new_agent[self.ID_POS] + 0.1 * (self.problem.ub - self.problem.lb) * mutation
                    mutated_agent = self.create_solution(
                        pos=self.amend_position(mutated_pos, self.problem.lb, self.problem.ub))
                    if mutated_agent[self.ID_TAR][0] < new_agent[self.ID_TAR][0]:
                        new_agent = mutated_agent

                if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                    self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                    self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
                self.pop[idx] = new_agent

            # 更新母鸡位置
            for i in range(self.hNum):
                idx = self.hens[i]
                mate_idx = self.mate[i]
                candidates = [k for k in self.roosters + self.hens if k != idx and k != mate_idx]
                other_idx = np.random.choice(candidates) if candidates else idx

                # 排名自适应参数 越接近1，排名越靠前 优秀个体探索，较差个体跟随
                rank_factor = 1.0 - (np.where(np.argsort([agent[self.ID_TAR][0] for agent in self.pop]) == idx)[0][0]
                                     / self.pop_size)  # 母鸡跟随因子基于适应度排名动态调整
                c1 = np.exp((self.local_best_fit[idx] - self.local_best_fit[mate_idx]) / (
                        abs(self.local_best_fit[idx]) + self.jixiaoliang))
                c2 = rank_factor * np.exp(self.local_best_fit[other_idx] - self.local_best_fit[idx])

                # 更新位置
                new_pos = self.local_best_pos[idx] + \
                          (self.local_best_pos[mate_idx] - self.local_best_pos[idx]) * c1 * np.random.rand(
                    self.problem.n_dims) + \
                          (self.local_best_pos[other_idx] - self.local_best_pos[idx]) * c2 * np.random.rand(
                    self.problem.n_dims)
                new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
                new_agent = self.create_solution(pos=new_pos)

                # 维度学习策略
                if self.gbest_pos is not None:  # 防止None访问
                    for d in np.random.choice(self.problem.n_dims, size=int(0.2 * self.problem.n_dims),
                                              replace=False):  # 对每个维度独立进行精英解维度值学习
                        new_agent[self.ID_POS][d] = (self.gbest_pos[d] +
                                                     0.05 * (self.problem.ub[d] - self.problem.lb[
                                    d]) * np.random.randn())
                    new_agent[self.ID_POS] = self.amend_position(new_agent[self.ID_POS], self.problem.lb,
                                                                 self.problem.ub)
                    new_agent[self.ID_TAR] = self.get_target_wrapper(new_agent[self.ID_POS])

                if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                    self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                    self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
                self.pop[idx] = new_agent

            # 更新小鸡位置
            for i in range(self.mNum):
                idx = self.chicks[i]
                mother_idx = self.mother[i]
                fl = self.FL[i]

                new_pos = self.local_best_pos[idx] + fl * (self.local_best_pos[mother_idx] - self.local_best_pos[idx])
                new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
                new_agent = self.create_solution(pos=new_pos)

                # 改进3：精英维度继承 随机选择两个维度，用全局最优解替换当前小鸡的位置
                if np.random.rand() < 0.2 and self.gbest_pos is not None:
                    elite_dims = np.random.choice(self.problem.n_dims, size=2, replace=False)
                    new_agent[self.ID_POS][elite_dims[0]] = self.gbest_pos[elite_dims[0]]
                    new_agent[self.ID_POS][elite_dims[1]] = self.gbest_pos[elite_dims[1]]
                    new_agent[self.ID_POS] = self.amend_position(new_agent[self.ID_POS], self.problem.lb,
                                                                 self.problem.ub)
                    new_agent[self.ID_TAR] = self.get_target_wrapper(new_agent[self.ID_POS])

                if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                    self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                    self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
                self.pop[idx] = new_agent

            # 更新全局最优
            self.update_global_best()