import data_io as io
import numpy as np
import os
import tensorly as tl
import pickle
import time
from regressor import RPCA, RTOT, TOT, ttt
from collections import defaultdict

# ----------------------------------------
# 配置区：根据你的笔记本性能调小参数
# ----------------------------------------
REPLICATIONS = 1  # 重复实验次数，设为 1 跑得快
MAX_ITR = 10  # 最大迭代次数，减少等待时间
N_SAMPLES = 50  # 样本数，设小一点 (原代码500)
DIMS = (15, 20, 5, 10)  # 维度，保持小规模
RU_VAL = 5  # Rank (秩)
PERCENTILE = 0.1  # 异常值比例

if __name__ == '__main__':
    print("开始运行模拟实验 (传统的 TOT, RTOT, RPCA)...")

    # 结果保存路径
    folder = r'./experiment-results/test-run/'
    if not os.path.exists(folder): os.makedirs(folder)

    dict_time = defaultdict(list)
    dict_Bs = defaultdict(list)
    dict_rpes = defaultdict(list)
    dict_aics = defaultdict(list)
    dict_ss = defaultdict(list)
    list_params = []

    for r in range(REPLICATIONS):
        print(f"Running Replication {r + 1}/{REPLICATIONS}...")

        # 1. 设置参数
        params = dict(
            N=N_SAMPLES,
            L=2, M=2,
            dims=DIMS,
            R=RU_VAL,
            Ru=RU_VAL,
            mu1=9e-3,
            mu2=1.5e-3,
            mu3=1.5e-3,
            tol=1e-6,
            max_itr=MAX_ITR,
            replications=REPLICATIONS,
            percentile=PERCENTILE,
            scale=10
        )

        # 2. 生成模拟数据 (Synthetic Data)
        # 这会调用 data_io.py 中的 gen_sync_data_norm
        print("Generating synthetic data...")
        params = io.gen_sync_data_norm(**params)
        list_params.append(params)

        # 3. 运行三种模型
        for model_class in [TOT, RTOT, RPCA]:
            model = model_class(**params)
            print(f"  Training {model.name.upper()}...")

            start = time.time()
            # fit 返回: rpe, B(核心张量), AIC, S(稀疏噪声)
            rpe_val, B, AIC, s = model.fit(verbose=True)
            end = time.time()

            # 计算测试集误差
            y_test = tl.partial_tensor_to_vec(params['y_test'], skip_begin=1)
            y_pre = ttt(params['x_test'], B, params['L'], params['dims'])
            y_pre = tl.partial_tensor_to_vec(y_pre, skip_begin=1)

            # 简单的均值校准
            m_test = np.mean(y_test, axis=1).reshape(-1, 1)
            m_pre = np.mean(y_pre, axis=1).reshape(-1, 1)
            y_pre = y_pre - m_pre + m_test

            test_rpe = np.linalg.norm(y_pre - y_test) / np.linalg.norm(y_test)

            # 记录结果
            dict_Bs[model.name].append(B)
            dict_rpes[model.name].append(test_rpe)
            dict_time[model.name].append(end - start)

            print(f"    -> {model.name.upper()} Done. Test RPE: {test_rpe:.4f}, Time: {end - start:.2f}s")

    # 4. 保存结果
    print("Saving results...")
    pickle.dump(list_params, open(os.path.join(folder, 'list_params.p'), 'wb'))
    pickle.dump(dict_rpes, open(os.path.join(folder, 'dict_rpes.p'), 'wb'))

    print("传统模型实验结束！")