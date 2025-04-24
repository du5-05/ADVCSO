import sys
import time
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import opfunu
import mealpy
from mealpy import Problem  # 新增导入Problem类
from mealpy.swarm_based import ALA, APO, SFOA, JADE, CPO, ChickenSO1, DRA, RBMO, LSHADE
from functools import partial

'''
跑单次试验
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

# 将目标函数定义为模块级函数
def cec_fun(x, func_num, dim):  # CEC测试函数包装器
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=dim)
    return func.evaluate(x)

def run_single_trial(args):  # 多进程运行的单个试验
    problem_dict, epoch, pop_size = args

    # 初始化所有模型
    models = {
        'ADVCSO': ChickenSO1.ADVCSO(epoch, pop_size),
        # 'OriginalCSO': ChickenSO.OriginalCSO(epoch, pop_size),
        # 'CSO_GPS': ChickenSO.CSO_GPS(epoch, pop_size),
        # 'CSO_DRA': ChickenSO.CSO_DRA(epoch, pop_size),
        # 'CSO_MUT': ChickenSO.CSO_MUT(epoch, pop_size),
        # 'CSO_GPS_DRA': ChickenSO.CSO_GPS_DRA(epoch, pop_size),
        # 'CSO_GPS_MUT': ChickenSO.CSO_GPS_MUT(epoch, pop_size),
        # 'CSO_DRA_MUT': ChickenSO.CSO_DRA_MUT(epoch, pop_size)
    }

    results = {}

    for name, model in models.items():
        start_time = time.time()
        _, best_f = model.solve(problem_dict)
        results[name] = (best_f, time.time() - start_time)
    return results


# def plot_initial_positions(positions, algo_name, func_name, dim, save_dir):  # 绘制二维/三维初始种群分布
#     plt.figure(figsize=(8, 6))
#
#     # 根据算法名称调整标题
#     if algo_name == 'ADVCSO':
#         title = 'Good Points Set Initialization'
#     elif algo_name == 'CSO':
#         title = 'Random Initialization'
#     else:
#         title = f'{algo_name} Initial Population'
#
#     if dim == 2:
#         plt.scatter(positions[:, 0], positions[:, 1], s=50, alpha=0.6, edgecolors='w')
#         plt.xlabel('Dimension 1')
#         plt.ylabel('Dimension 2')
#     elif dim == 3:
#         ax = plt.subplot(projection='3d')
#         ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=50, alpha=0.6)
#         ax.set_xlabel('X1')
#         ax.set_ylabel('X2')
#         ax.set_zlabel('X3')
#     else:
#         print(f"{dim}D数据无法可视化，使用前两维投影")
#         plt.scatter(positions[:, 0], positions[:, 1], s=50, alpha=0.6, edgecolors='w')
#         plt.xlabel('X1 (Projection)')
#         plt.ylabel('X2 (Projection)')
#
#     plt.title(title)
#     plt.grid(True, alpha=0.3)
#
#     # 保存图片
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, f"{algo_name}_init_{dim}D.png"),
#                 dpi=1200, bbox_inches='tight')
#     plt.close()

if __name__ == '__main__':
    # 实验配置
    year = '2017'
    dim = 30
    epoch = 1000
    pop_size = 50
    num_runs = 50  # 多次运行次数

    for fun_num in range(24, 30):  # 遍历CEC2017测试函数
        fun_name = f'F{fun_num}'
        func_num = f"{fun_name}{year}"
        print(f"\n正在处理函数: {func_num}")

        # 使用partial绑定当前函数的参数
        current_func = partial(cec_fun, func_num=func_num, dim=dim)

        # 构造问题字典
        func_class = opfunu.get_functions_by_classname(func_num)[0](ndim=dim)

        problem = Problem(
            lb=np.array(func_class.lb),
            ub=np.array(func_class.ub),
            minmax="min",
            fit_func=current_func,
            name=func_num,
            dim=dim  # 显式传递维度
        )

        # ========== 单次运行可视化 ==========
        # 初始化模型
        models = {
            'ADVCSO': ChickenSO1.ADVCSO(epoch, pop_size),
            # 'OriginalCSO': ChickenSO.OriginalCSO(epoch, pop_size),
            # 'CSO_GPS': ChickenSO.CSO_GPS(epoch, pop_size),
            # 'CSO_DRA': ChickenSO.CSO_DRA(epoch, pop_size),
            # 'CSO_MUT': ChickenSO.CSO_MUT(epoch, pop_size),
            # 'CSO_GPS_DRA': ChickenSO.CSO_GPS_DRA(epoch, pop_size),
            # 'CSO_GPS_MUT': ChickenSO.CSO_GPS_MUT(epoch, pop_size),
            # 'CSO_DRA_MUT': ChickenSO.CSO_DRA_MUT(epoch, pop_size)
        }

        # 传递Problem对象并初始化
        for name, model in models.items():
            model.problem = problem  # 传递Problem对象
            model.initialize_variables()  # 正确触发初始化

        # # 绘制ADVCSO和CSO的初始种群
        # for algo in ['ADVCSO', 'CSO']:
        #     positions = models[algo].initial_positions
        #     plot_dir = r'C:\Users\wukunwei555\Desktop\EI\image'
        #     plot_initial_positions(positions, algo, func_num, dim, plot_dir)

        # 构造问题字典（用于后续solve方法）
        problem_dict = {
            "fit_func": current_func,
            "lb": func_class.lb.tolist(),
            "ub": func_class.ub.tolist(),
            "minmax": "min",
        }

        # 运行并收集结果
        convergence_data = {}
        single_run_results = {
            "Function": func_num,
            "Year": year,
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        for name, model in models.items():
            _, best_f = model.solve(problem_dict)
            convergence_data[name] = model.history.list_global_best_fit
            single_run_results[name] = best_f  # 记录最佳适应度值

        # 保存单次运行结果到CSV
        # data_dir = r'C:\Users\wukunwei555\Desktop\EI\data'
        # os.makedirs(data_dir, exist_ok=True)
        # csv_path = os.path.join(data_dir, "single_run_results.csv")
        #
        # # 创建或追加数据
        # if os.path.exists(csv_path):
        #     pd.DataFrame([single_run_results]).to_csv(csv_path, mode='a', header=False, index=False)
        # else:
        #     pd.DataFrame([single_run_results]).to_csv(csv_path, index=False)

        # 绘制收敛曲线
        # plt.figure(figsize=(10, 6))
        # markers = ['o', '^', 's', '*', 'D', 'X', 'P', 'v']
        # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        # for (name, data), marker, color in zip(convergence_data.items(), markers, colors):
        #     plt.plot(data, linestyle='-', marker=marker, markevery=40,
        #              color=color, linewidth=2, markersize=6, alpha=0.9, label=name)
        #
        # plt.title(f'Convergence Curve: {func_num}', fontsize=14, pad=15)
        # plt.xlabel('Iteration', fontsize=12, labelpad=10)
        # plt.ylabel('Fitness', fontsize=12, labelpad=10)
        # plt.yscale('log')
        # plt.grid(True, which='both', linestyle='--', alpha=0.5)
        # plt.legend()
        #
        # # 保存图片
        # save_dir = r'C:\Users\wukunwei555\Desktop\EI\images'
        # os.makedirs(save_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_dir, f"{func_num}_convergence.png"),
        #             dpi=600, bbox_inches='tight')
        # plt.close()

        ### ========== 新增代码：绘制ADVCSO角色分配比例 ========== ###
        if 'ADVCSO' in models:
            model = models['ADVCSO']
            if hasattr(model, 'role_history') and len(model.role_history) > 0:
                # 提取角色分配历史数据
                df_roles = pd.DataFrame(model.role_history)

                # 绘制角色比例变化曲线
                plt.figure(figsize=(12, 7))
                plt.plot(
                    df_roles["epoch"], df_roles["rooster_ratio"],
                    label="Rooster Ratio",
                    color='#FF6F00',  # 橙色
                    linewidth=2,
                    linestyle='-',
                    marker='o',
                    markersize=6,
                    markevery=10,  # 每10个点显示一个标记
                    alpha=0.8
                )
                plt.plot(
                    df_roles["epoch"], df_roles["hen_ratio"],
                    label="Hen Ratio",
                    color='#6A1B9A',  # 紫色
                    linewidth=2,
                    linestyle='-',
                    marker='s',
                    markersize=6,
                    markevery=10,
                    alpha=0.8
                )
                plt.plot(
                    df_roles["epoch"], df_roles["chick_ratio"],
                    label="Chick Ratio",
                    color='#2E7D32',  # 绿色
                    linewidth=2,
                    linestyle='-',
                    marker='^',
                    markersize=6,
                    markevery=10,
                    alpha=0.8
                )

                plt.title(f"Dynamic Role Allocation in ADVCSO", fontsize=14, pad=20)
                plt.xlabel("Iteration", fontsize=12)
                plt.ylabel("Population Ratio", fontsize=12)
                plt.ylim(0, 1)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)

                plt.legend(
                    loc='upper right',
                    fontsize=10,
                    frameon=True,
                    shadow=True,
                    facecolor='white'
                )
                plt.grid(True, linestyle='--', alpha=0.5)

                # 保存图片
                role_plot_dir = r'C:\Users\wukunwei555\Desktop\EI\role_plots'
                os.makedirs(role_plot_dir, exist_ok=True)
                plt.savefig(os.path.join(role_plot_dir, f"{func_num}_role_allocation.png"),
                            dpi=1200, bbox_inches='tight')
                plt.close()

        # # 绘制并保存三维函数图
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