import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    num_salesmen = 10

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

    # 存储各算法结果
    results = []

    # 运行所有算法
    for algo_name, optimizer in algorithms.items():
        best_solution, best_fitness = optimizer.solve(model)

        paths = model.decode(best_solution)
        distances = [model.calculate_distance(p) for p in paths if len(p) > 2]

        # 收集结果数据
        results.append({
            "Algorithm": algo_name,
            "Max Distance": np.max(distances) if distances else np.nan,
            "Total Distance": np.sum(distances) if distances else np.nan,
            "Best Fitness": best_fitness,
            "History": optimizer.history.list_global_best_fit.copy(),
            "Depot Node": model.depot_idx,
        })

        visualize_results(model, best_solution, optimizer.history.list_global_best_fit.copy(), algo_name)

    # 生成结果表格
    df = pd.DataFrame(results)

    # 保存到Excel
    data_dir = r"C:\Users\wukunwei555\Desktop\EI\CSO\MTSP\data"
    os.makedirs(data_dir, exist_ok=True)
    excel_path = os.path.join(data_dir, f'results_{time.strftime("%Y%m%d_%H%M%S")}.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
        worksheet = writer.sheets['Sheet1']

        # 自动调整列宽
        for column in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in column)
            adjusted_width = max_length + 2
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

    print(f"\n结果已保存至：{excel_path}")

    # 绘制对比曲线
    plt.figure(figsize=(12, 8))
    for data in results:
        algo_name = data["Algorithm"]
        history = data.get("History")
        if history is not None and len(history) > 0:
            plt.plot(history, label=algo_name, linewidth=2)

    plt.title(f'Algorithm Comparison\n(Salesmen: {num_salesmen})', fontsize=20)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Best Fitness', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(r"C:\Users\wukunwei555\Desktop\EI\CSO\MTSP\image", f'comparison_{time.strftime("%Y%m%d_%H%M%S")}.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    run_mtsp_example()