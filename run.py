import sys
import time
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import opfunu
import mealpy
from mealpy.swarm_based import ALA, APO, SFOA, JADE, CPO, ChickenSO, DRA, RBMO, LSHADE
import multiprocessing
from functools import partial

'''
跑重复试验
'''

'''
适应度函数及维度dim的选择
cec函数名字格式：函数名+年份，比如要选择2022的F1函数，func_num = 'F1'+'2022'
cec2005：F1-F25, 可选 dim = 10, 30, 50
cec2008：F1-F7,  可选 2 <= dim <= 1000
cec2010：F1-F20, 可选 100 <= dim <= 1000
cec2013：F1-F28, 可选 dim = 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
cec2014：F1-F30, 可选 dim = 10, 20, 30, 50, 100
cec2015：F1-F15, 可选 dim = 10, 30
cec2017：F1-F29, 可选 dim = 2, 10, 20, 30, 50, 100
cec2019：F1-F10, 可选 dim: F1=9,F2=16,F3=18,其他=10
cec2020：F1-F10, 可选 dim = 2, 5, 10, 15, 20, 30, 50, 100
cec2021：F1-F10, 可选 dim = 2, 10, 20
cec2022：F1-F12, 可选 dim = 2, 10, 20
'''

# 将目标函数定义为模块级函数（关键修改）
def cec_fun(x, func_num, dim):
    """CEC测试函数包装器"""
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=dim)
    return func.evaluate(x)

def run_single_trial(args):
    """多进程运行的单个试验"""
    problem_dict, epoch, pop_size = args

    # 初始化所有模型
    models = {
        'ADVCSO': ChickenSO.ADVCSO(epoch, pop_size),
        'ALA': ALA.OriginalALA(epoch, pop_size),
        # 'APO': APO.OriginalAPO(epoch, pop_size),
        # 'SFOA': SFOA.OriginalSFOA(epoch, pop_size),
        'CPO': CPO.OriginalCPO(epoch, pop_size),
        # 'DRA': DRA.OriginalDRA(epoch, pop_size),
        # 'RBMO': RBMO.OriginalRBMO(epoch, pop_size),
        'JADE': JADE.JADE(epoch, pop_size)
        # 'LSHADE': LSHADE.OriginalLSHADE(epoch, pop_size)
    }

    results = {}
    for name, model in models.items():
        start_time = time.time()
        _, best_f = model.solve(problem_dict)
        results[name] = (best_f, time.time() - start_time)
    return results


if __name__ == '__main__':
    # 实验配置
    year = '2017'
    dim = 30
    epoch = 1000
    pop_size = 50
    num_runs = 50  # 多次运行次数

    for fun_num in range(1, 30):  # 遍历CEC2017测试函数
        fun_name = f'F{fun_num}'
        func_num = f"{fun_name}{year}"
        print(f"\n正在处理函数: {func_num}")

        # 使用partial绑定当前函数的参数（关键修改）
        current_func = partial(cec_fun, func_num=func_num, dim=dim)

        # 构造问题字典
        func_class = opfunu.get_functions_by_classname(func_num)[0](ndim=dim)
        problem_dict = {
            "fit_func": current_func,
            "lb": func_class.lb.tolist(),
            "ub": func_class.ub.tolist(),
            "minmax": "min",
        }

        # ========== 单次运行可视化 ==========
        # 初始化模型
        # models = {
        #     'ADVCSO': ChickenSO.ADVCSO(epoch, pop_size),
        #     'CSO': ChickenSO.OriginalCSO(epoch, pop_size),
        #     'GTO': GTO.OriginalGTO(epoch, pop_size),
        #     'COA': COA.OriginalCOA(epoch, pop_size),
        #     'HGS': HGS.OriginalHGS(epoch, pop_size),
        #     'SLO': SLO.OriginalSLO(epoch, pop_size)
        # }

        # # 运行并收集结果
        # convergence_data = {}
        # single_run_results = {
        #     "Function": func_num,
        #     "Year": year,
        #     "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        # }
        #
        # for name, model in models.items():
        #     _, best_f = model.solve(problem_dict)
        #     convergence_data[name] = model.history.list_global_best_fit
        #     single_run_results[name] = best_f  # 记录最佳适应度值
        #
        # # 保存单次运行结果到CSV
        # data_dir = r'C:\Users\wukunwei555\Desktop\EI\data'
        # os.makedirs(data_dir, exist_ok=True)
        # csv_path = os.path.join(data_dir, "single_run_results.csv")
        #
        # # 创建或追加数据
        # if os.path.exists(csv_path):
        #     pd.DataFrame([single_run_results]).to_csv(csv_path, mode='a', header=False, index=False)
        # else:
        #     pd.DataFrame([single_run_results]).to_csv(csv_path, index=False)
        #
        # # 绘制收敛曲线
        # plt.figure(figsize=(10, 6))
        # markers = ['o', '^', 's', '*', 'D', 'x']
        # colors = ['b', 'm', 'c', 'k', 'r', 'g']
        # for (name, data), marker, color in zip(convergence_data.items(), markers, colors):
        #     plt.plot(data, linestyle='-', marker=marker, markevery=40,
        #              color=color, linewidth=2, markersize=6, label=name)
        #
        # plt.title(f'Convergence Curve: {func_num}', fontsize=12)
        # plt.xlabel('Iteration', fontsize=10)
        # plt.ylabel('Fitness', fontsize=10)
        # plt.yscale('log')
        # plt.grid(True, which='both', linestyle='--', alpha=0.5)
        # plt.legend()
        #
        # # 保存图片
        # save_dir = r'C:\Users\wukunwei555\Desktop\EI\images'
        # os.makedirs(save_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_dir, f"{func_num}_convergence.png"),
        #             dpi=300, bbox_inches='tight')
        # plt.close()

        # 绘制并保存三维函数图
        # plt.figure(figsize=(10, 6))
        # try:
        #     func_3d = opfunu.get_functions_by_classname(func_num)[0](ndim=2)
        #     opfunu.plot_3d(func_3d, n_space=500, show=False)
        #     plt.title(f'3D Function Plot: cec{year}-{fun_name}', fontsize=12)
        #     plt.tight_layout()
        #
        #     plot_3d_dir = r'C:\Users\wukunwei555\Desktop\EI\3d_plots'
        #     os.makedirs(plot_3d_dir, exist_ok=True)
        #
        #     timestamp = time.strftime('%Y%m%d_%H%M%S')
        #     plot_filename = os.path.join(plot_3d_dir, f"{func_num}_3d_{timestamp}.png")
        #     plt.savefig(plot_filename, dpi=250, bbox_inches='tight')
        #     print(f"3D图已保存至：{plot_filename}")
        # except Exception as e:
        #     print(f"生成3D图时出错：{str(e)}")
        # finally:
        #     plt.close()

        # ========== 多次运行统计 ==========
        print(f"开始多次运行实验：{func_num}...")
        start_time = time.time()

        # 准备多进程参数（包含完整problem_dict）
        task_args = [(problem_dict, epoch, pop_size)] * num_runs

        # 使用进程池并行执行
        with multiprocessing.Pool() as pool:
            trials_results = pool.map(run_single_trial, task_args)

        # # 准备箱线图数据
        # algorithms = ['ADVCSO', 'CSO', 'GTO', 'COA', 'HGS', 'SLO']
        # box_data = []
        # for algo in algorithms:
        #     algo_data = [res[algo][0] for res in trials_results]
        #     box_data.append(algo_data)
        #
        # # 创建箱线图
        # plt.figure(figsize=(12, 6))
        # boxes = plt.boxplot(
        #     box_data,
        #     labels=algorithms,
        #     patch_artist=True,
        #     showmeans=True,
        #     widths=0.6,
        #     meanline=False
        # )
        #
        # # 样式设置
        # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        # for box, color in zip(boxes['boxes'], colors):
        #     box.set(facecolor=color, alpha=0.6, linewidth=1.5)
        # for whisker in boxes['whiskers']:
        #     whisker.set(color='gray', linestyle='--')
        # for cap in boxes['caps']:
        #     cap.set(color='gray', linewidth=1.5)
        # for median in boxes['medians']:
        #     median.set(color='red', linewidth=2)
        # for mean in boxes['means']:
        #     mean.set(marker='D', markeredgecolor='black', markerfacecolor='gold')
        #
        # plt.title(f'Algorithm Comparison: {func_num}\n({num_runs} Independent Runs)', fontsize=12)
        # plt.ylabel('Fitness Value (log scale)')
        # plt.yscale('log')
        # plt.grid(axis='y', alpha=0.4)
        # plt.xticks(fontsize=9)
        #
        # # 保存箱线图
        # boxplot_dir = r'C:\Users\wukunwei555\Desktop\EI\boxplots'
        # os.makedirs(boxplot_dir, exist_ok=True)
        # plt.savefig(os.path.join(boxplot_dir, f"{func_num}_boxplot.png"),
        #             dpi=300, bbox_inches='tight')
        # plt.close()

        # 结果处理
        metrics = {}
        for algo in ['ADVCSO', 'ALA', 'CPO', 'JADE']:
            values = np.array([res[algo][0] for res in trials_results])
            times = np.array([res[algo][1] for res in trials_results])

            metrics[algo] = {
                'best': np.min(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'avg_time': np.mean(times)
            }

        # 保存统计结果
        stats_df = pd.DataFrame({
            'Function': [func_num] * 4,
            'Algorithm': list(metrics.keys()),
            'Best': [v['best'] for v in metrics.values()],
            'Mean': [v['mean'] for v in metrics.values()],
            'Std': [v['std'] for v in metrics.values()],
            'AvgTime': [v['avg_time'] for v in metrics.values()]
        })

        csv_path = r'C:\Users\wukunwei555\Desktop\EI\data\experiment_results.csv'
        if os.path.exists(csv_path):
            stats_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            stats_df.to_csv(csv_path, index=False)

        print(f"完成{func_num}处理，耗时：{time.time() - start_time:.2f}s")

    print("所有实验完成！")