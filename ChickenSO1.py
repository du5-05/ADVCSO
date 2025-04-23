import numpy as np
from scipy.stats import cauchy
from mealpy.optimizer import Optimizer

class OriginalCSO(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化函数，继承自Optimzer类

        # 初始化时验证并设置各种算法参数
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])  # 迭代次数，要求在1到100000之间
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])  # 种群大小，要求在10到10000之间
        self.G = self.validator.check_int("G", G, [1, 1000])  # 更新周期（每隔G代进行一次更新）
        self.jixiaoliang = self.validator.check_float("jixiaoliang", jixiaoliang, (0, 1e-5))  # 精度控制参数

        # 设置算法所需的参数列表，便于后续使用
        self.set_parameters(["epoch", "pop_size", "G", "jixiaoliang"])
        self.sort_flag = False  # 是否需要对种群进行排序，初始为False

    def generate_position(self, lb, ub):  # 使用父类的随机生成方法
        return np.random.uniform(lb, ub)

    def initialize_variables(self):
        # 改进建议：可以考虑增加参数的可配置性，比如增加初始化方式的选择（例如：不同种群的初始化方法等）
        # 初始化种群和相关变量
        super().initialize_variables()  # 调用父类的初始化方法
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)  # 创建种群  在optimizer文件中找到这个函数来修改佳点集

        # 改进建议：种群划分方式可以根据实际问题场景进行调整，是否可以增加不同的划分策略
        # 根据种群大小确定不同类别的个体数
        self.rNum = int(self.pop_size * 0.2)  # 公鸡数量，占种群的20%
        self.hNum = int(self.pop_size * 0.3)  # 母鸡数量，占种群的30%
        self.mNum = self.pop_size - self.rNum - self.hNum  # 小鸡数量，占剩余的50%

        # 初始化每个个体的局部最优解的位置和适应度
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]  # 存储每个个体的局部最优位置
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]  # 存储每个个体的局部最优适应度

        self.gbest = float("inf")  # 初始化全局最优适应度为无穷大
        self.gbest_pos = None  # 初始化全局最优位置为空

        # 初始化其他相关的变量
        self.roosters = []  # 存储公鸡的列表
        self.hens = []  # 存储母鸡的列表
        self.chicks = []  # 存储小鸡的列表
        self.mate = None  # 配偶选择
        self.motherLib = None  # 母鸡库
        self.mother = None  # 母鸡
        self.FL = None  # 位置更新因子（fl）

        # 保存初始种群位置（随机初始化）
        self.initial_positions = np.array([agent[self.ID_POS] for agent in self.pop])

    def create_solution(self, lb=None, ub=None, pos=None):
        # 创建一个新的解，并计算其适应度
        if lb is None:
            lb = self.problem.lb  # 获取问题的下界
        if ub is None:
            ub = self.problem.ub  # 获取问题的上界

        if pos is None:
            pos = self.generate_position(lb, ub)  # 如果没有提供位置，则生成一个随机位置
        pos = self.amend_position(pos, lb, ub)  # 修正位置，确保其在上下界范围内
        target = self.get_target_wrapper(pos)  # 计算该位置的目标函数值（即适应度）
        fitness = target[0] if isinstance(target, (list, tuple, np.ndarray)) else target  # 如果目标函数值为列表/元组/数组，则取第一个值作为适应度
        return [pos, [fitness]]  # 返回位置和适应度

    def evolve(self, epoch):
        # 进化过程，在每G代时更新种群  G小，搜索慢但精细化，G大，搜索快但容易陷入局部最优
        # 改进建议：改成自适应参数，例如通过一个非线性公式，使G从大变小
        if epoch % self.G == 0:
            # 对种群进行排序，按适应度从低到高排序（最小适应度为最好）
            sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
            # 获取公鸡、母鸡、小鸡的索引
            self.roosters = sorted_indices[:self.rNum].tolist()  # 公鸡
            self.hens = sorted_indices[self.rNum:self.rNum + self.hNum].tolist()  # 母鸡
            self.chicks = sorted_indices[self.rNum + self.hNum:].tolist()  # 小鸡

            # 随机选择配偶、母鸡库、母鸡和FL因子（位置更新因子）
            self.mate = np.random.randint(0, self.rNum, self.hNum)
            self.motherLib = np.random.choice(self.hens, size=self.mNum, replace=True)
            self.mother = np.random.choice(self.motherLib, size=self.mNum, replace=True)
            self.FL = 0.4 * np.random.rand(self.mNum) + 0.5  # FL因子随机生成，范围在[0.5, 0.9]之间  影响小鸡位置

        # 更新公鸡位置
        for i in range(self.rNum):
            idx = self.roosters[i]
            another_idx = self.roosters[np.random.choice([k for k in range(self.rNum) if k != i])]  # 随机选择另一个公鸡进行比较

            # 适应度较差的公鸡在更新时，会有较大的变化范围
            if self.local_best_fit[idx] <= self.local_best_fit[another_idx]:
                temp_sigma = 1  # 如果当前个体的适应度较好，则没有变化
            else:
                # 改进建议：temp_sigma的计算方式依赖于指数函数，可能会引入过大的变化，是否可以根据问题特性调节？
                # 如果当前个体适应度较差，计算temp_sigma，控制位置更新的幅度
                temp_sigma = np.exp((self.local_best_fit[another_idx] - self.local_best_fit[idx]) /
                                    (abs(self.local_best_fit[idx]) + self.jixiaoliang))

            # 更新公鸡的位置
            new_pos = self.local_best_pos[idx] * (1 + temp_sigma * np.random.randn(self.problem.n_dims))  # 使用正态分布进行位置变化
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)  # 修正位置，确保在问题的上下界内
            new_agent = self.create_solution(
                pos=new_pos,
                lb=self.problem.lb,
                ub=self.problem.ub)

            # 如果更新后适应度变好，则更新局部最优解
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_pos.copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新母鸡位置
        for i in range(self.hNum):
            idx = self.hens[i]
            mate_idx = self.roosters[self.mate[i]]  # 获取配偶公鸡的索引
            candidates = [k for k in self.roosters + self.hens if k != idx and k != mate_idx]  # 获取所有非当前个体和配偶的候选者
            other_idx = np.random.choice(candidates) if candidates else idx  # 随机选择一个候选者

            # 指数函数会趋于某一个值，在迭代后期变化幅度就会很小，适合精细化搜索，但变化太快就会趋近无穷小，基本上就不变化
            # 改进建议：可尝试修改迭代一定次数来使用c1，c2更新位置
            current_fit = self.local_best_fit[idx]  # 当前母鸡的适应度
            c1 = np.exp((current_fit - self.local_best_fit[mate_idx]) / (abs(current_fit) + self.jixiaoliang))  # 计算因子c1
            c2 = np.exp(self.local_best_fit[other_idx] - current_fit)  # 计算因子c2

            # 更新母鸡的位置
            rand = np.random.rand(self.problem.n_dims)
            new_pos = (self.local_best_pos[idx] +
                       (self.local_best_pos[mate_idx] - self.local_best_pos[idx]) * c1 * rand +
                       (self.local_best_pos[other_idx] - self.local_best_pos[idx]) * c2 * rand)
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)  # 修正位置
            new_agent = self.create_solution(
                pos=new_pos,
                lb=self.problem.lb,
                ub=self.problem.ub)

            # 更新局部最优解
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_pos.copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新小鸡位置
        for i in range(self.mNum):
            idx = self.chicks[i]
            mother_idx = self.mother[i]  # 获取母鸡的索引
            fl = self.FL[i]  # 获取位置更新因子

            # 更新小鸡的位置
            new_pos = self.local_best_pos[idx] + fl * (
                        self.local_best_pos[mother_idx] - self.local_best_pos[idx])  # 根据母鸡的位置更新小鸡
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)  # 修正位置
            new_agent = self.create_solution(
                pos=new_pos,
                lb=self.problem.lb,
                ub=self.problem.ub)

            # 更新局部最优解
            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_pos.copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # 更新全局最优解
        current_best = np.min(self.local_best_fit)
        if current_best < self.gbest:
            self.gbest = current_best  # 更新全局最优适应度
            self.gbest_pos = self.local_best_pos[np.argmin(self.local_best_fit)].copy()  # 更新全局最优位置

class ADVCSO(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, G=5, jixiaoliang=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.G = self.validator.check_int("G", G, [1, 1000])
        self.jixiaoliang = self.validator.check_float("jixiaoliang", jixiaoliang, (0, 1e-5))
        self.set_parameters(["epoch", "pop_size", "G", "jixiaoliang"])
        self.sort_flag = False

        # 新增参数：柯西/高斯变异切换阈值
        self.mutation_switch = 0.3  # 30%迭代次数后切换为高斯变异
        self.role_history = []  # 新增：记录角色分配历史

    def generate_good_point_set(self, pop_size, dim, lb, ub):
        """佳点集初始化"""
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

    def generate_position(self, lb, ub):  # 覆盖父类方法：生成佳点集位置
        return self.generate_good_point_set(1, self.problem.n_dims, lb, ub)[0]

    def initialize_variables(self):
        """改进1：混合初始化策略"""
        super().initialize_variables()

        # 生成佳点集初始种群
        gps_pop = self.generate_good_point_set(self.pop_size, self.problem.n_dims, self.problem.lb, self.problem.ub)
        self.pop = [self.create_solution(pos=gps_pop[i]) for i in range(self.pop_size)]

        # 初始化局部最优
        self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
        self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]
        self.gbest = min(self.local_best_fit)
        best_idx = np.argmin(self.local_best_fit)
        self.gbest_pos = self.pop[best_idx][self.ID_POS].copy()

        # 对前10%精英个体施加扰动
        elite_num = max(1, int(0.1 * self.pop_size))
        sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
        for i in sorted_indices[:elite_num]:
            noise = 0.05 * (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)  # 施加高斯噪声，避免早熟收敛
            self.pop[i][self.ID_POS] = np.clip(self.pop[i][self.ID_POS] + noise, self.problem.lb, self.problem.ub)

            # 更新局部最优（避免覆盖gbest_pos）
            self.local_best_pos = [agent[self.ID_POS].copy() for agent in self.pop]
            self.local_best_fit = [agent[self.ID_TAR][0] for agent in self.pop]

            # 记录初始多样性并防止为零
            self.initial_diversity = np.std(self.local_best_fit)
            if self.initial_diversity == 0:
                self.initial_diversity = 1e-10  # 防止初始多样性为零

        # 保存初始种群位置（佳点集初始化）
        self.initial_positions = np.array([agent[self.ID_POS] for agent in self.pop])

    def evolve(self, epoch):
        """改进2：动态角色分配与自适应参数"""
        # 计算种群多样性
        current_diversity = np.std([agent[self.ID_TAR][0] for agent in self.pop])
        if current_diversity == 0 or np.isnan(current_diversity):
            current_diversity = 1e-10  # 防止当前种群多样性为零或NaN

        # 处理初始多样性为零的情况
        if self.initial_diversity == 0:
            self.initial_diversity = 1e-10

        # 计算多样性比例并处理NaN
        diversity_ratio = np.nan_to_num(current_diversity / self.initial_diversity)
        self.mutation_switch = 0.3 + 0.4 * (1 - diversity_ratio)  # 当多样性低时，延后切换

        # 自适应G参数（余弦退火）
        decay = 0.5 * (1 + np.cos(np.pi * epoch / self.epoch))  # 通过余弦函数动态衰减参数，初期保持高探索性，后期增强开发能力
        G_current = max(5, int(self.G * decay))  # G从初始值衰减到5

        # 动态调整角色数量和分配索引
        if epoch % G_current == 0:
            # 动态调整角色比例（添加范围限制）
            self.rNum = max(1, min(int(self.pop_size * (0.3 + 0.2 * (current_diversity / self.initial_diversity))), self.pop_size - 2))  # 至少1个公鸡 种群多样性下降时增加公鸡比例
            self.hNum = max(1, min(int(self.pop_size * (0.3 - 0.1 * diversity_ratio)), self.pop_size - self.rNum - 1))  # 至少1个母鸡
            self.mNum = max(0, self.pop_size - self.rNum - self.hNum)

            # 排序并分配角色索引
            sorted_indices = np.argsort([agent[self.ID_TAR][0] for agent in self.pop])
            self.roosters = sorted_indices[:self.rNum].tolist()
            self.hens = sorted_indices[self.rNum:self.rNum + self.hNum].tolist()
            self.chicks = sorted_indices[self.rNum + self.hNum:].tolist()

            # 更新配偶、母鸡库等
            top_roosters = self.roosters[:max(1, int(0.5 * self.rNum))]
            self.mate = np.random.choice(top_roosters, size=self.hNum, replace=True)
            self.motherLib = np.random.choice(self.hens, size=self.mNum, replace=True)
            self.mother = np.random.choice(self.motherLib, size=self.mNum, replace=True)
            self.FL = 0.4 * np.random.rand(self.mNum) + 0.5

            # 记录当前角色比例（转换为百分比）
            self.role_history.append({
                "epoch": epoch,
                "rooster_ratio": self.rNum / self.pop_size,
                "hen_ratio": self.hNum / self.pop_size,
                "chick_ratio": self.mNum / self.pop_size
            })

        # ============= 公鸡更新 =============
        for i in range(self.rNum):
            if i >= len(self.roosters):  # 边界保护
                break
            idx = self.roosters[i]
            another_idx = np.random.choice([k for k in self.roosters if k != idx])

            # 自适应学习因子（余弦衰减）
            alpha = 0.5 * (1 - np.cos(np.pi * epoch / self.epoch))  # 公鸡学习因子随迭代次数非线性衰减
            if self.local_best_fit[idx] > self.local_best_fit[another_idx]:
                temp_sigma = np.exp((self.local_best_fit[another_idx] - self.local_best_fit[idx]) /
                                    (abs(self.local_best_fit[idx]) + self.jixiaoliang))
            else:
                temp_sigma = 1.0

            new_pos = self.local_best_pos[idx] * (1 + alpha * temp_sigma * np.random.randn(self.problem.n_dims))
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 改进3：公鸡混合变异
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

        # ============= 母鸡更新 =============
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

            new_pos = self.local_best_pos[idx] + \
                      (self.local_best_pos[mate_idx] - self.local_best_pos[idx]) * c1 * np.random.rand(
                self.problem.n_dims) + \
                      (self.local_best_pos[other_idx] - self.local_best_pos[idx]) * c2 * np.random.rand(
                self.problem.n_dims)
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_agent = self.create_solution(pos=new_pos)

            # 改进3：维度学习策略 将部分维度替换为全局最优解
            if self.gbest_pos is not None:  # 防止None访问
                for d in np.random.choice(self.problem.n_dims, size=int(0.2 * self.problem.n_dims), replace=False):  # 对每个维度独立进行精英解维度值学习
                    new_agent[self.ID_POS][d] = (self.gbest_pos[d] +
                                                 0.05 * (self.problem.ub[d] - self.problem.lb[d]) * np.random.randn())
                new_agent[self.ID_POS] = self.amend_position(new_agent[self.ID_POS], self.problem.lb, self.problem.ub)
                new_agent[self.ID_TAR] = self.get_target_wrapper(new_agent[self.ID_POS])

            if new_agent[self.ID_TAR][0] < self.local_best_fit[idx]:
                self.local_best_pos[idx] = new_agent[self.ID_POS].copy()
                self.local_best_fit[idx] = new_agent[self.ID_TAR][0]
            self.pop[idx] = new_agent

        # ============= 小鸡更新 =============
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
        current_best = np.min(self.local_best_fit)
        if current_best < self.gbest:
            self.gbest = current_best
            self.gbest_pos = self.local_best_pos[np.argmin(self.local_best_fit)].copy()