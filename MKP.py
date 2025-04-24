import numpy as np
import matplotlib.pyplot as plt
from mealpy import Problem
from ADVCSO import ADVCSO
from CSO import OriginalCSO
from GTO import OriginalGTO
from AVOA import OriginalAVOA
from HGS import OriginalHGS

class MKPModel(Problem):
    def __init__(self, weights, values, capacities):
        self.weights = weights  # 物品重量列表 (n,)
        self.values = values    # 物品价值列表 (n,)
        self.capacities = capacities  # 背包容量列表 (m,)
        self.n_items = len(weights)
        self.n_knapsacks = len(capacities)

        lb = [0] * self.n_items  # 每个物品的下界为 0（不放入任何背包）
        ub = [self.n_knapsacks] * self.n_items  # 每个物品的上界为背包数量
        super().__init__(lb=lb, ub=ub, minmax="max", dim=self.n_items)

    def repair(self, solution):
        solution = np.round(solution).astype(int)

        # 第一阶段：处理超容
        for k in range(self.n_knapsacks):
            items = np.where(solution == k + 1)[0]
            total_weight = np.sum(self.weights[items])

            # 按价值密度排序（价值/重量）
            sorted_items = sorted(items,
                                  key=lambda x: self.values[x] / self.weights[x],
                                  reverse=True)

            # 移除低价值密度物品直到满足容量
            while total_weight > self.capacities[k] and len(sorted_items) > 0:
                removed = sorted_items.pop()  # 移除价值密度最低的
                solution[removed] = 0
                total_weight -= self.weights[removed]

        # 第二阶段：填充剩余容量
        unassigned = np.where(solution == 0)[0]
        for item in unassigned:
            best_knapsack = -1
            best_value = 0
            for k in range(self.n_knapsacks):
                current_items = np.where(solution == k + 1)[0]
                current_weight = np.sum(self.weights[current_items])
                if (current_weight + self.weights[item] <= self.capacities[k]
                        and self.values[item] > best_value):
                    best_knapsack = k
                    best_value = self.values[item]
            if best_knapsack != -1:
                solution[item] = best_knapsack + 1

        return solution

    def fit_func(self, solution):
        repaired = self.repair(solution)
        total_value = 0
        penalty = 0

        for k in range(self.n_knapsacks):
            items = np.where(repaired == k + 1)[0]
            total_weight = np.sum(self.weights[items])
            total_value += np.sum(self.values[items])

            # 增加超容惩罚
            if total_weight > self.capacities[k]:
                penalty += (total_weight - self.capacities[k]) * 1000

        return total_value - penalty

def run_mkp_example():
    # 示例数据（10个物品，3个背包）
    weights = np.array([1.2, 3.4, 2.5, 1.6, 1.9, 4.3, 5.1, 2.8, 3.5, 4.2])
    values = np.array([2.3, 1.5, 3.4, 1.6, 5.2, 4.3, 2.8, 3.9, 4.1, 2.5])
    capacities = [5.3, 4.5, 6.2]
    model = MKPModel(weights, values, capacities)

    # 实验参数
    algorithms = ["ADVCSO", "CSO", "GTO", "AVOA", "HGS"]
    n_runs = 20
    epoch = 30
    pop_size = 50

    # 优化器字典
    optimizers = {
        "ADVCSO": ADVCSO,
        "CSO": OriginalCSO,
        "GTO": OriginalGTO,
        "AVOA": OriginalAVOA,
        "HGS": OriginalHGS,
    }

    # 结果存储结构
    results = {algo: {"fitness": [], "histories": [], "solutions": []} for algo in algorithms}

    # 运行实验
    for algo_name in algorithms:
        print(f"\n正在运行 {algo_name} 算法...")
        for run in range(n_runs):
            print(f"进度: {run + 1}/{n_runs}")
            # 初始化优化器
            optimizer = optimizers[algo_name](epoch=epoch, pop_size=pop_size)

            # 运行优化
            best_solution, best_fitness = optimizer.solve(model)

            # 存储结果
            results[algo_name]["fitness"].append(best_fitness)
            results[algo_name]["histories"].append(optimizer.history.list_global_best_fit)
            results[algo_name]["solutions"].append(best_solution)

    # 计算统计指标
    stats = {}
    for algo_name in algorithms:
        fitness = np.array(results[algo_name]["fitness"])
        stats[algo_name] = {
            "best": np.max(fitness),
            "worst": np.min(fitness),
            "mean": np.mean(fitness),
            "std": np.std(fitness),
            "optimal_count": np.sum(fitness == np.max(fitness))
        }

    # 打印统计结果
    print("\n统计结果:")
    for algo_name in algorithms:
        print(f"\n{algo_name}:")
        print(f"最佳适应度: {stats[algo_name]['best']:.2f}")
        print(f"最差适应度: {stats[algo_name]['worst']:.2f}")
        print(f"平均适应度: {stats[algo_name]['mean']:.2f}")
        print(f"标准差: {stats[algo_name]['std']:.2f}")
        print(f"达到最优次数: {stats[algo_name]['optimal_count']}")

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    colors = {"ADVCSO": 'b', "CSO": 'm', "GTO": 'c', "AVOA": 'k', "HGS": 'r'}
    markers = {"ADVCSO": 'o', "CSO": '^', "GTO": 's', "AVOA": '*', "HGS": 'D'}

    for algo_name in algorithms:
        # 计算平均历史适应度
        histories = np.array(results[algo_name]["histories"])
        avg_history = np.mean(histories, axis=0)

        # 绘制曲线
        plt.plot(avg_history,
                    label=algo_name,
                    color=colors[algo_name],
                    linewidth=2,
                    marker=markers[algo_name],
                    markevery=5,
                    markersize=6)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Algorithm Comparison", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_mkp_example()