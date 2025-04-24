import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl.utils import get_column_letter
from mealpy import Problem
from ADVCSO import ADVCSO
from CSO import OriginalCSO
from GTO import OriginalGTO
from AVOA import OriginalAVOA
from HGS import OriginalHGS


class MTSPModel(Problem):
    def __init__(self, cities, num_salesmen, depot_idx, mode="minmax"):
        self.cities = cities
        self.num_salesmen = num_salesmen
        self.depot_idx = depot_idx  # 固定配送中心
        self.mode = mode
        self.n_cities = cities.shape[0]

        # 编码维度：每个城市分配一个概率（选择哪个旅行商）
        dim = self.n_cities * self.num_salesmen
        lb = [0.0] * dim
        ub = [1.0] * dim

        super().__init__(lb=lb, ub=ub, minmax="min", name="MTSP")
        self.log_to = "console"

    def decode(self, solution):
        # 将解转换为旅行商分配概率矩阵
        prob_matrix = np.reshape(solution, (self.n_cities, self.num_salesmen))
        paths = [[] for _ in range(self.num_salesmen)]

        # 每个城市分配给概率最高的旅行商
        for city in range(self.n_cities):
            if city == self.depot_idx:
                continue  # 配送中心不参与分配
            salesman = np.argmax(prob_matrix[city])
            paths[salesman].append(city)

        # 强制每个旅行商路径包含配送中心并去重
        for i in range(self.num_salesmen):
            path = paths[i]
            if self.depot_idx not in path:
                path = [self.depot_idx] + path + [self.depot_idx]
            else:
                path = [self.depot_idx] + [x for x in path if x != self.depot_idx] + [self.depot_idx]
            paths[i] = list(dict.fromkeys(path))  # 保留顺序去重

        return paths

    def calculate_distance(self, path):  # 计算单条路径的总距离
        total = 0.0
        for i in range(len(path) - 1):
            total += np.linalg.norm(self.cities[path[i]] - self.cities[path[i + 1]])
        return total

    def fit_func(self, solution):
        paths = self.decode(solution)
        distances = [self.calculate_distance(p) for p in paths if len(p) > 2]  # 过滤空路径

        if len(distances) == 0:
            return 1e10  # 无效解惩罚

        max_dist = np.max(distances)
        total_dist = np.sum(distances)
        balance_penalty = np.std(distances) * 0.1  # 平衡性惩罚

        # 最终适应度（优先优化核心目标）
        if self.mode == "minmax":
            return max_dist + balance_penalty
        else:
            return total_dist + balance_penalty


def visualize_results(model, best_solution, history, algo_name=""):
    paths = model.decode(best_solution)

    # 绘制路径图
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    colors = plt.cm.tab20(np.linspace(0, 1, len(paths)))

    # 绘制各旅行商路径
    for i, path in enumerate(paths):
        coords = model.cities[path]
        # 绘制路径线
        line, = ax.plot(coords[:, 0], coords[:, 1],
                        color=colors[i], linewidth=2.5,
                        marker='o', markersize=8,
                        alpha=0.8, label=f'Salesman {i + 1}')

        # 添加方向箭头
        arrow_interval = max(1, len(coords) // 5)
        for j in range(len(coords) - 1):
            if j % arrow_interval == 0 or j == len(coords) - 2:
                dx = coords[j + 1, 0] - coords[j, 0]
                dy = coords[j + 1, 1] - coords[j, 1]
                ax.arrow(coords[j, 0], coords[j, 1],
                         dx * 0.7, dy * 0.7,
                         head_width=2.0, head_length=3.0,
                         fc=colors[i], ec='k', lw=0.8,
                         length_includes_head=True)

    # 绘制配送中心
    depot_coord = model.cities[model.depot_idx]
    ax.scatter(depot_coord[0], depot_coord[1],
               c='red', s=300, marker='s', edgecolors='k',
               zorder=10, label='Depot')

    plt.title(f'Optimized Routes ({algo_name})', fontsize=20)
    plt.xlabel('X Coordinate', fontsize=16)
    plt.ylabel('Y Coordinate', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tight_layout()

    # 保存图像
    save_dir = os.path.join(r"C:\Users\wukunwei555\Desktop\EI\CSO\MTSP\image", algo_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'result_{time.strftime("%Y%m%d_%H%M%S")}.png'), dpi=300)
    plt.show()
    plt.close()


def run_mtsp_example():
    cities = np.random.rand(50, 2) * 100  # 生成50个随机城市坐标
    depot_idx = 0  # 固定配送中心为第一个城市
    num_salesmen = 5
    n_runs = 20

    model = MTSPModel(
        cities=cities,
        num_salesmen=num_salesmen,
        depot_idx=depot_idx,
        mode="minmax"
    )

    # 配置算法
    algorithms = {
        "ADVCSO": ADVCSO(epoch=1000, pop_size=50),
        "CSO": OriginalCSO(epoch=1000, pop_size=50),
        "GTO": OriginalGTO(epoch=1000, pop_size=50),
        "AVOA": OriginalAVOA(epoch=1000, pop_size=50),
        "HGS": OriginalHGS(epoch=1000, pop_size=50),
    }

    # 存储统计结果
    stats = {
        algo_name: {
            "max_dists": [],
            "total_dists": [],
            "fitnesses": []
        } for algo_name in algorithms.keys()
    }

    # 运行所有算法
    for run in range(n_runs):
        for algo_name, optimizer in algorithms.items():
            # 运行算法
            best_solution, best_fitness = optimizer.solve(model)
            paths = model.decode(best_solution)

            # 计算指标
            distances = [model.calculate_distance(p) for p in paths if len(p) > 2]
            valid_distances = distances if distances else [np.nan]

            # 存储结果
            stats[algo_name]["max_dists"].append(np.max(valid_distances))
            stats[algo_name]["total_dists"].append(np.sum(valid_distances))
            stats[algo_name]["fitnesses"].append(best_fitness)

            # 只在最后一次运行生成可视化结果
            if run == n_runs - 1:
                visualize_results(model, best_solution,
                                  optimizer.history.list_global_best.copy(),
                                  f"{algo_name}_FinalRun")

    # 计算统计指标
    results = []
    for algo_name, data in stats.items():
        # 转换为numpy数组处理缺失值
        max_dists = np.array(data["max_dists"])
        total_dists = np.array(data["total_dists"])
        fitnesses = np.array(data["fitnesses"])

        # 过滤无效值
        max_dists = max_dists[~np.isnan(max_dists)]
        total_dists = total_dists[~np.isnan(total_dists)]
        fitnesses = fitnesses[~np.isnan(fitnesses)]

        # 计算统计量
        results.append({
            "Algorithm": algo_name,
            "Avg Max Distance": np.mean(max_dists) if len(max_dists) > 0 else np.nan,
            "Std Max Distance": np.std(max_dists) if len(max_dists) > 0 else np.nan,
            "Avg Total Distance": np.mean(total_dists) if len(total_dists) > 0 else np.nan,
            "Std Total Distance": np.std(total_dists) if len(total_dists) > 0 else np.nan,
            "Avg Fitness": np.mean(fitnesses) if len(fitnesses) > 0 else np.nan,
            "Std Fitness": np.std(fitnesses) if len(fitnesses) > 0 else np.nan,
            "Success Runs": len(fitnesses)  # 有效运行次数
        })

    # 生成结果表格
    df = pd.DataFrame(results)

    # 保存到Excel（带格式优化）
    data_dir = r"C:\Users\wukunwei555\Desktop\EI\CSO\MTSP\data"
    os.makedirs(data_dir, exist_ok=True)
    excel_path = os.path.join(data_dir, f'statistics_{time.strftime("%Y%m%d_%H%M%S")}.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
        worksheet = writer.sheets['Sheet1']

        # 获取列名与Excel列字母的映射
        col_letters = {col: get_column_letter(idx + 1) for idx, col in enumerate(df.columns)}

        # 设置列宽
        col_widths = {
            'Algorithm': 15,
            'Avg Max Distance': 18,
            'Std Max Distance': 18,
            'Avg Total Distance': 20,
            'Std Total Distance': 20,
            'Avg Fitness': 15,
            'Std Fitness': 15,
            'Success Runs': 15
        }

        for col, width in col_widths.items():
            letter = col_letters[col]  # 转换列名为字母（如 'Algorithm' -> 'A'）
            worksheet.column_dimensions[letter].width = width

    print(f"\n结果已保存至：{excel_path}")

if __name__ == "__main__":
    run_mtsp_example()