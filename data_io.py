import numpy as np
import pandas as pd
import plotly.express as px
import scipy.sparse as sparse
import tensorly as tl

from scipy.io import loadmat
from collections import defaultdict

import numpy as np
import tensorly as tl
from tensorly import tucker_to_tensor

"""
脚本及其作用：
1.加载神经科学数据；2.生成合成数据
函数
1.ttt 定义核心算子，张量收缩积；
2.mcp函数；
3.一系列生成数据函数；
4.合成数据生成。

"""


def ttt(x, b, L, dims):
    return np.tensordot(
        x,
        b,
        axes=[
            [k + 1 for k in range(len(dims[:L]))],
            [k for k in range(len(dims[:L]))]
        ]
    )


def gen_lambda_data(**params):
    N = params['N'] = 200
    params['L'] = 2
    params['M'] = 2
    params['dims'] = (35, 29, 7, 29)

    data_lambda = loadmat('./data/clustered_data_lambda.mat')
    x = data_lambda['clustered_input'][0]
    y = data_lambda['clustered_output'][0]
    idx = list(set([i for i in range(len(x))]) - set([2, 3, 10, 13, 14, 16]))
    tmpx = []
    tmpy = []

    for rx, ry in zip(x[idx], y[idx]):
        tmpy.append(ry)
        tmpx.append(rx)

    x = np.concatenate(tmpx, axis=0).reshape(-1, 35, 29)
    y = np.concatenate(tmpy, axis=0).reshape(-1, 7, 29)

    x_train, x_test = x[:N], x[N:]
    y_train, y_test = y[:N], y[N:]

    params['x'] = x_train
    params['y'] = y_train
    params['x_test'] = x_test
    params['y_test'] = y_test

    return params


def mcp(x, a=1, lam=.8):
    return lam * abs(x) - x ** 2 / (2 * a) if abs(x) < a * lam else (a * lam ** 2) / 2


def gen_sync_data(**params):
    N = params['N'] = 200
    L = params['L'] = 2
    M = params['M'] = 2
    Ru = params['Ru']
    dims = params['dims'] = (35, 29, 7, 29)
    percentile = params['percentile']
    R = 5  # np.random.randint(low=3, high=7)

    x = [
        [
            np.random.uniform(1e-3, 1) * np.concatenate(
                [np.array([np.cos(1e-3 * np.pi * r * (j / p)) if r % 2 == 1 else np.sin(1e-3 * np.pi * r * (j / p)) for j in range(1, p + 1)]).reshape(-1, 1) for r in range(1, R + 1)],
                axis=1
            )
            for p in dims[:2]
        ] for _ in range(1, N + 1)
    ]
    x = [tl.cp_to_tensor((None, t)) for t in x]

    rns = np.random.RandomState(seed=43)
    b = [rns.normal(size=(p, Ru)) for p in dims]
    b = tl.cp_to_tensor((None, b))
    b = b / np.linalg.norm(b)
    y = ttt(x, b, L, dims)

    b2 = np.zeros_like(b)
    b2 = b2.flatten()
    b2[:50] = np.random.normal(loc=0, scale=1, size=50)
    b2 = (b2 / np.linalg.norm(b2)).reshape(dims)
    y += ttt(x, b2, L, dims)

    e = np.random.normal(loc=0, scale=1, size=y.shape)
    e = e / np.linalg.norm(e)
    y += e

    # y = tl.partial_tensor_to_vec(y, skip_begin=1)
    # df = {'y': y.flatten(), 'time': [i for i in range(203)] * N, 'n': [i for i in range(N) for _ in range(203)]}
    # fig = px.line(df, x='time', y='y', line_group='n', color='n')
    # fig.show()

    x_test, x = x[:80], x[80:]
    y_test, y = y[:80], y[80:]

    if percentile > 0:
        percentage = percentile
        sample_indices = np.random.randint(0, 120, size=int(percentage * 120))
        outlier_idx = {i: [] for i in sample_indices}
        y_with_outliers = tl.partial_tensor_to_vec(y[sample_indices], skip_begin=1)
        for idx, sample in zip(sample_indices, y_with_outliers):
            start_index = np.random.randint(100, 194)
            outlier_length = np.random.randint(10, 30)
            outlier_idx[idx] = [i for i in range(start_index, start_index + outlier_length)]
            sample[start_index: start_index + outlier_length] = np.random.uniform(.8, 2, size=min(outlier_length, 203 - start_index))
        y[sample_indices] = y_with_outliers.reshape(-1, params['dims'][2], params['dims'][3])
        params['s_idx'] = outlier_idx

    # tmp = tl.partial_tensor_to_vec(y, skip_begin=1)
    # df = {'y': tmp.flatten(), 'time': [i for i in range(203)] * 120, 'n': [i for i in range(120) for _ in range(203)]}
    # fig = px.line(df, x='time', y='y', line_group='n', color='n')
    # fig.show()
    # fig.write_html('./training-data.html')

    params['x'] = x
    params['y'] = y
    params['x_test'] = x_test
    params['y_test'] = y_test

    return params


def gen_sync_data_norm(**params):
    N = params['N'] = 500 
    L = params['L'] = 2  
    M = params['M'] = 2
    dims = params['dims'] = (15, 20, 5, 10) 
    percentile = params['percentile'] 
    Ru = params['Ru']  
    # scale = params['scale'] 

    rns = np.random.RandomState(seed=51) 
    base = [rns.normal(size=(p, 5)) for p in dims[:L]] 
    x = [[np.random.uniform(1e-3, 1) * b for b in base] for _ in range(1, N + 1)]
   
    x = [tl.cp_to_tensor((None, t)) for t in x] 



    # rns = np.random.RandomState(seed=5) this one is for normal
    rns = np.random.RandomState(seed=42)
    b = [rns.normal(size=(p, Ru)) for p in dims]  
    b = tl.cp_to_tensor((None, b)) 
    b = b / np.linalg.norm(b) 
    y = ttt(x, b, L, dims)
    e = np.random.normal(loc=0, scale=1, size=y.shape)
    e = e / np.linalg.norm(e) # 归一化
    y += e 



    x_test, x = x[:100], x[100:]
    y_test, y = y[:100], y[100:] 

    if percentile > 0:
        y = tl.partial_tensor_to_vec(y, skip_begin=1)
        indices = np.random.randint(0, 400, size=int(400 * percentile))
        outlier_idx = {i: [] for i in indices} 
        for n in indices:
            idx = np.random.randint(0, 40) 
            outlier_idx[n] = [i for i in range(idx, idx + 5)] 

            y[n][idx: idx + 5] = np.random.uniform(.8, 2, size=5)  

        y = y.reshape(-1, dims[-2], dims[-1]) 
        params['s_idx'] = outlier_idx



    
    params['x'] = x
    params['y'] = y
    params['x_test'] = x_test
    params['y_test'] = y_test

    return params




def gen_rotr_data_tucker_y_only(**params):
    # ==========================================
    # 0. 参数解析
    # ==========================================
    N = params['N'] = 2000              
    L = params['L']  =2                  
    M = params['M']  =2                  
    dims = params['dims'] = (15, 20, 5,10)    # 数据集1         # e.g., (15, 20, 5, 10)
    #dims = dims = params['dims'] = (15, 20, 10,10)  # 数据集2
    input_dims = dims[:L]              # (15, 20)
    output_dims = dims[L:]             # (5, 10)
    
    # Tucker 秩配置
    tucker_ranks = params['tucker_ranks'] 
    ranks_in = tucker_ranks[:L]        
    ranks_out = tucker_ranks[L:]       
    
    percentile = params['percentile']
    init_scale = params['init_scale_gen_core'] # 控制核心张量数值大小 

    # ==========================================
    # 1. 生成系数张量 B (正交 Tucker 结构)
    # ==========================================

    rns_b = np.random.RandomState(seed=44)
    
    factors_B = []
    for i, d in enumerate(dims):
        mat = rns_b.randn(d, tucker_ranks[i])
        Q, _ = np.linalg.qr(mat) # QR分解保证正交性
        factors_B.append(Q)
        
    core_B = rns_b.randn(*tucker_ranks) * init_scale
    B_true = tucker_to_tensor((core_B, factors_B))
    B_true = B_true / np.linalg.norm(B_true) # 归一化

    # ==========================================
    # 2. 生成输入张量 X (共享基底 Tucker 结构)
    # ==========================================
    # 对应逻辑：所有样本共享因子矩阵 U (input subspace)，但拥有不同的 Core
    rns_x = np.random.RandomState(seed=43)
    
    # 生成输入空间的共享正交基
    factors_X_shared = []
    for i in range(L):
        mat = rns_x.randn(input_dims[i], ranks_in[i])
        Q, _ = np.linalg.qr(mat)
        factors_X_shared.append(Q)
    
    # 生成 N 个样本
    X_list = []
    for _ in range(N):
        core_x = rns_x.uniform(0, 1, size=ranks_in) 
        x_i = tucker_to_tensor((core_x, factors_X_shared))
        X_list.append(x_i)
        
    X = np.array(X_list) # (N, 15, 20)

    # ==========================================
    # 3. 计算真实响应并添加稠密噪声
    # ==========================================

    y = ttt(X, B_true, L, dims) # (N, 5, 10)


    e = np.random.normal(loc=0, scale=1, size=y.shape) 
    e = e / np.linalg.norm(e)
    y += e 

    # ==========================================
    # 4. 划分数据集
    # ==========================================

    x_test, x = X[:400], X[400:] 
    y_test, y = y[:400], y[400:] 
    
    # 获取训练集样本数 (通常是 400)
    N_train = y.shape[0]

    # ==========================================
    # 5. 注入稀疏异常 (完全复刻 gen_sync_data_norm 逻辑)
    # ==========================================
    if percentile > 0:

        y_flat = tl.partial_tensor_to_vec(y, skip_begin=1)
        
        # 2. 抽取要被污染的行 (Indices)
        num_outliers = int(N_train * percentile)
        indices = np.random.randint(0, N_train, size=num_outliers)
        
        outlier_idx = {i: [] for i in indices} # 记录异常值的样本索引
        
        # 获取展平后的特征长度
        feature_len = y_flat.shape[1]

        for n in indices:

            max_start_idx = max(0, feature_len - 5)
            idx = np.random.randint(0, max_start_idx if max_start_idx > 0 else 1)
            
            outlier_idx[n] = [i for i in range(idx, idx + 5)] # 记录受损位置
            

            y_flat[n][idx: idx + 5] = np.random.uniform(0.8, 2, size=5)
            # 
            
        # 3. 还原形状

        y = y_flat.reshape(-1, *output_dims)
        
        params['s_idx'] = outlier_idx # 记录异常索引

    # ==========================================
    # 6. 返回结果
    # ==========================================
    params['x'] = x          # 训练集 X (无异常)
    params['y'] = y          # 训练集 Y (含异常)
    params['x_test'] = x_test
    params['y_test'] = y_test
    
    # 可选：如果你需要 Ground Truth B 用于评估
    params['B_true'] = B_true

    return params



def gen_rotr_data_tucker_y_and_x(**params):
    # ==========================================
    # 0. 参数解析
    # ==========================================
    N = params['N'] = 2000              
    L = params['L']  =2                  
    M = params['M']  =2                  
    dims = dims = params['dims'] = (15, 20, 5,10)             # e.g., (15, 20, 5, 10)
    input_dims = dims[:L]              # (15, 20)
    output_dims = dims[L:]             # (5, 10)
    
    # Tucker 秩配置
    tucker_ranks = params['tucker_ranks'] 
    ranks_in = tucker_ranks[:L]        
    ranks_out = tucker_ranks[L:]       
    
    percentile = params['percentile']
    init_scale = params['init_scale_gen_core'] # 控制核心张量数值大小 
    percentile_x = params['percentile_x'] # 控制x的异常值比例
    outlier_mag = params['outlier_mag'] # x的异常值生成尺度
    # ==========================================
    # 1. 生成系数张量 B (正交 Tucker 结构)
    # ==========================================

    rns_b = np.random.RandomState(seed=44)
    
    factors_B = []
    for i, d in enumerate(dims):
        mat = rns_b.randn(d, tucker_ranks[i])
        Q, _ = np.linalg.qr(mat) # QR分解保证正交性
        factors_B.append(Q)
        
    core_B = rns_b.randn(*tucker_ranks) * init_scale
    B_true = tucker_to_tensor((core_B, factors_B))
    B_true = B_true / np.linalg.norm(B_true) # 归一化

    # ==========================================
    # 2. 生成输入张量 X (共享基底 Tucker 结构)
    # ==========================================
    # 对应逻辑：所有样本共享因子矩阵 U (input subspace)，但拥有不同的 Core
    rns_x = np.random.RandomState(seed=43)
    
    # 生成输入空间的共享正交基
    factors_X_shared = []
    for i in range(L):
        mat = rns_x.randn(input_dims[i], ranks_in[i])
        Q, _ = np.linalg.qr(mat)
        factors_X_shared.append(Q)
    
    # 生成 N 个样本
    X_list = []
    for _ in range(N):
        core_x = rns_x.uniform(0, 1, size=ranks_in) 

        x_i = tucker_to_tensor((core_x, factors_X_shared))
        X_list.append(x_i)
        
    X = np.array(X_list) # (N, 15, 20)

    # ==========================================
    # 3. 计算真实响应并添加稠密噪声
    # ==========================================

    y = ttt(X, B_true, L, dims) # (N, 5, 10)


    e = np.random.normal(loc=0, scale=1, size=y.shape) 
    e = e / np.linalg.norm(e)
    y += e 

    # ==========================================
    # 4. 划分数据集
    # ==========================================

    x_test, x = X[:400], X[400:] 
    y_test, y = y[:400], y[400:] 
    
    # 获取训练集样本数 (通常是 400)
    N_train = y.shape[0]
    # ==========================================
    # 5. 注入 X 的稀疏异常 (新增逻辑)
    # ==========================================

    if percentile_x > 0:
        num_outliers_x = int(N_train * percentile_x)
        indices_x = np.random.choice(N_train, size=num_outliers_x, replace=False)
        params['x_outlier_indices'] = indices_x
        
        block_h = 5  # 高度
        block_w = 6  # 宽度
        for n in indices_x :
            # 确保起始点不会越界
            # input_dims[0] - block_h
            start_d0 = np.random.randint(0, input_dims[0] - block_h)
            start_d1 = np.random.randint(0, input_dims[1] - block_w)
            
            # 生成噪声块
            noise_block = np.random.uniform(outlier_mag, 2*outlier_mag, size=(block_h, block_w))
            sign_block = np.random.choice([-1, 1], size=(block_h, block_w))
            
            # 注入
            x[n, start_d0 : start_d0 + block_h, start_d1 : start_d1 + block_w] += noise_block * sign_block



    # ==========================================
    # 6. 注入y的稀疏异常 (完全复刻 gen_sync_data_norm 逻辑)
    # ==========================================
    if percentile > 0:

        y_flat = tl.partial_tensor_to_vec(y, skip_begin=1)
        
        # 2. 抽取要被污染的行 (Indices)
        num_outliers = int(N_train * percentile)
        indices = np.random.randint(0, N_train, size=num_outliers)
        
        outlier_idx = {i: [] for i in indices} # 记录异常值的样本索引
        
        # 获取展平后的特征长度
        feature_len = y_flat.shape[1]

        for n in indices:
            # 选择破坏起始点，确保 idx+5 不越界
            max_start_idx = max(0, feature_len - 5)
            idx = np.random.randint(0, max_start_idx if max_start_idx > 0 else 1)
            
            outlier_idx[n] = [i for i in range(idx, idx + 5)] # 记录受损位置
            
            # 核心修改：注入均匀分布异常 [0.8, 2]
            # y_[idx] ... 逻辑复刻
            y_flat[n][idx: idx + 5] = np.random.uniform(0.8, 2, size=5)
            # 
            
        # 3. 还原形状
        y = y_flat.reshape(-1, *output_dims)
        
        params['s_idx'] = outlier_idx # 记录异常索引

    # ==========================================
    # 6. 返回结果
    # ==========================================
    params['x'] = x          # 训练集 X (无异常)
    params['y'] = y          # 训练集 Y (含异常)
    params['x_test'] = x_test
    params['y_test'] = y_test
    

    params['B_true'] = B_true

    return params



def gen_rotr_high_leverage_data(**params):
    # ==========================================
    # 0. 参数解析
    # ==========================================


    dims = dims = params['dims'] = (15, 20, 5,10)    
    tucker_ranks = params['tucker_ranks']
    N = params['N'] = 2000              
    L = params['L']  =2                  
    M = params['M']  =2       
    
    input_dims = dims[:L]
    output_dims = dims[L:]
    
    # --- [关键参数] ---
    # 1. 高杠杆样本 (Good Extremes): X大, Y大, 符合模型
    leverage_ratio = params.get('leverage_ratio', 0.1)   # 5% 的样本是高杠杆点
    leverage_mag = params.get('leverage_mag', 10.0)       # 放大 10 倍 (足够显著)
    
    # 2. 真实异常样本 (Bad Outliers): X正常, Y被破坏 (或X被破坏)
    outlier_ratio = params.get('outlier_ratio', 0.05)     # 5% 的样本是坏点
    outlier_mag = params.get('outlier_mag', 10.0)         # 坏点的破坏力度
    
    # 3. 稠密噪声水平 (SNR)
    noise_level = params.get('dense_noise_level', 0.05)   # 5% 的背景噪音

    # ==========================================
    # 1. 生成系数张量 B (保持不变)
    # ==========================================
    rns_b = np.random.RandomState(seed=44)
    factors_B = []
    for i, d in enumerate(dims):
        mat = rns_b.randn(d, tucker_ranks[i])
        Q, _ = np.linalg.qr(mat)
        factors_B.append(Q)
    core_B = rns_b.randn(*tucker_ranks)
    B_true = tucker_to_tensor((core_B, factors_B))
    B_true = B_true / np.linalg.norm(B_true) 

    # ==========================================
    # 2. 生成输入张量 X (引入高杠杆逻辑)
    # ==========================================
    rns_x = np.random.RandomState(seed=43)
    
    # 共享基底
    factors_X_shared = []
    for i in range(L):
        mat = rns_x.randn(input_dims[i], tucker_ranks[i])
        Q, _ = np.linalg.qr(mat)
        factors_X_shared.append(Q)
    
    # 确定样本类型索引
    indices_all = np.arange(N)
    np.random.shuffle(indices_all)
    
    num_lev = int(N * leverage_ratio)
    num_out = int(N * outlier_ratio)
    
    idx_leverage = indices_all[:num_lev]                    # 高杠杆样本索引
    idx_outlier = indices_all[num_lev : num_lev+num_out]    # 真实异常样本索引
    idx_normal = indices_all[num_lev+num_out:]              # 正常样本索引

    X_list = []
    for n in range(N):
        core_x = rns_x.uniform(0, 1, size=tucker_ranks[:L])
        if n in idx_leverage:
            core_x *= leverage_mag 
        x_i = tucker_to_tensor((core_x, factors_X_shared))
        X_list.append(x_i)
        
    X = np.array(X_list) 


    std_x_clean = np.std(X[idx_normal]) 

    # ==========================================
    # 3. 计算 Y 
    # ==========================================

    y_clean = ttt(X, B_true, L, dims) 
    
    std_y_normal = np.std(y_clean[idx_normal])
    dense_noise = np.random.normal(0, noise_level * std_y_normal, size=y_clean.shape)
    y = y_clean + dense_noise

    # ==========================================
    # 4. 划分数据集
    # ==========================================


    # ==========================================
    # 5. 注入“真实异常” (Bad Data)
    # ==========================================

    if num_out > 0:

        bad_mag = leverage_mag * std_y_normal 
        
        y_flat = y.reshape(N, -1)
        feat_len = y_flat.shape[1]
        
        for n in idx_outlier:
            start = np.random.randint(0, max(1, feat_len - 5))
            noise = np.random.choice([-1, 1], size=5) * np.random.uniform(bad_mag, 2*bad_mag, size=5)
            y_flat[n, start:start+5] += noise
            
        y = y_flat.reshape(N, *output_dims)

    # ==========================================
    # 6. 数据切分与返回
    # ==========================================
    test_size = 400
    
    params['x'] = X[test_size:]
    params['y'] = y[test_size:]
    params['x_test'] = X[:test_size]
    params['y_test'] = y[:test_size]
    
    params['B_true'] = B_true
    

    train_indices = np.arange(test_size, N)
    
    params['idx_leverage_train'] = [i-test_size for i in idx_leverage if i >= test_size]
    params['idx_outlier_train'] = [i-test_size for i in idx_outlier if i >= test_size]

    return params




def gen_rotr_mixed_data(**params):
    # ==========================================
    # 0. 参数解析
    # ==========================================
    # 基础维度参数


    tucker_ranks = params['tucker_ranks']
    
    N = params['N'] = 2000              
    L = params['L']  =2                  
    M = params['M']  =2                  
    dims = dims = params['dims'] = (15, 20, 5,10)             # e.g., (15, 20, 5, 10)
    input_dims = dims[:L]              # (15, 20)
    output_dims = dims[L:]             # (5, 10)

    input_dims = dims[:L]
    output_dims = dims[L:]
    
    # --- [混合异常配置] ---
    
    # 1. 高杠杆样本 (High Leverage, Good): X大, Y大, Residual ≈ 0
    # 这些样本用于测试模型是否能保留有用的极端值
    leverage_ratio = params.get('leverage_ratio', 0.05)   # 10%
    leverage_mag = params.get('leverage_mag', 10.0)      # 放大倍数
    
    # 2. 坏 X 样本 (Bad X, Sx != 0): X被破坏, Y正常
    # 这些样本用于测试模型能否分离 Sx
    bad_x_ratio = params.get('bad_x_ratio', 0.05)        # 5%
    bad_x_mag = params.get('bad_x_mag', 10.0)            # 破坏强度
    
    # 3. 坏 Y 样本 (Bad Y, Sy != 0): X正常, Y被破坏
    # 这些样本用于测试模型能否分离 Sy
    bad_y_ratio = params.get('bad_y_ratio', 0.07)        # 5%
    bad_y_mag = params.get('bad_y_mag', 10.0)            # 破坏强度
    
    # 4. 背景噪音 (Dense Noise)
    noise_level = params.get('dense_noise_level', 0.05)

    # ==========================================
    # 1. 生成系数张量 B (保持不变)
    # ==========================================
    rns_b = np.random.RandomState(seed=44)
    factors_B = []
    for i, d in enumerate(dims):
        mat = rns_b.randn(d, tucker_ranks[i])
        Q, _ = np.linalg.qr(mat)
        factors_B.append(Q)
    core_B = rns_b.randn(*tucker_ranks)
    B_true = tucker_to_tensor((core_B, factors_B))
    B_true = B_true / np.linalg.norm(B_true) 

    # ==========================================
    # 2. 生成 X 的基底 (共享部分)
    # ==========================================
    rns_x = np.random.RandomState(seed=43)
    factors_X_shared = []
    for i in range(L):
        mat = rns_x.randn(input_dims[i], tucker_ranks[i])
        Q, _ = np.linalg.qr(mat)
        factors_X_shared.append(Q)
    
    # ==========================================
    # 3. 样本索引分配
    # ==========================================
    indices_all = np.arange(N)
    np.random.shuffle(indices_all)
    
    n_lev = int(N * leverage_ratio)
    n_bad_x = int(N * bad_x_ratio)
    n_bad_y = int(N * bad_y_ratio)
    
    idx_leverage = indices_all[:n_lev]
    idx_bad_x = indices_all[n_lev : n_lev + n_bad_x]
    idx_bad_y = indices_all[n_lev + n_bad_x : n_lev + n_bad_x + n_bad_y]
    idx_normal = indices_all[n_lev + n_bad_x + n_bad_y:]

    # ==========================================
    # 4. 生成 X (包含正常 X 和 高杠杆 X)
    # ==========================================
    X_list = []
    for n in range(N):
        core_x = rns_x.uniform(0, 1, size=tucker_ranks[:L])
        
        # [注入类型 1]: 高杠杆点 -> 放大核心
        if n in idx_leverage:
            core_x *= leverage_mag
            
        x_i = tucker_to_tensor((core_x, factors_X_shared))
        X_list.append(x_i)
        
    X = np.array(X_list)
    X_clean_gt = X.copy() 

    std_x_base = np.std(X[idx_normal])

    # ==========================================
    # 5. 生成 Y (基于当前的合法 X)
    # ==========================================
    y_clean = ttt(X, B_true, L, dims)
    
    # 添加背景稠密噪声
    std_y_base = np.std(y_clean[idx_normal])
    dense_noise = np.random.normal(0, noise_level * std_y_base, size=y_clean.shape)
    y = y_clean + dense_noise
    y_ground_truth = y_clean.copy() 
    # ==========================================
    # 6. 注入独立异常 (Bad Data)
    # ==========================================
    
    # --- [注入类型 2]: 坏 X (Bad X) ---
    if n_bad_x > 0:
        block_h, block_w = 5, 6
        mag_x = bad_x_mag * std_x_base
        
        for n in idx_bad_x:
            # 随机位置
            start_d0 = np.random.randint(0, max(1, input_dims[0] - block_h))
            start_d1 = np.random.randint(0, max(1, input_dims[1] - block_w))
            
            # 生成噪声块
            noise_block = np.random.uniform(mag_x, 2*mag_x, size=(block_h, block_w))
            sign_block = np.random.choice([-1, 1], size=(block_h, block_w))
            
            # 叠加破坏
            X[n, start_d0 : start_d0 + block_h, start_d1 : start_d1 + block_w] += noise_block * sign_block

    # --- [注入类型 3]: 坏 Y (Bad Y) ---
    if n_bad_y > 0:
        mag_y = bad_y_mag * std_y_base
        
        y_flat = y.reshape(N, -1)
        feat_len = y_flat.shape[1]
        
        for n in idx_bad_y:
            # 随机位置
            start = np.random.randint(0, max(1, feat_len - 5))
            
            # 生成稀疏噪声
            noise_y = np.random.choice([-1, 1], size=5) * np.random.uniform(mag_y, 2*mag_y, size=5)
            
            # 叠加破坏
            y_flat[n, start:start+5] += noise_y
            
        y = y_flat.reshape(N, *output_dims)

    # ==========================================
    # 7. 数据切分与返回
    # ==========================================
    test_size = 400
    
    params['x'] = X[test_size:]
    params['y'] = y[test_size:]
    params['x_test'] = X[:test_size]
    params['y_test'] = y[:test_size]
    params['B_true'] = B_true
    params['y_test_gt'] = y_ground_truth[:test_size] 
    params['x_test_clean'] = X_clean_gt[:test_size]

    def filter_idx(indices):
        return [i - test_size for i in indices if i >= test_size]

    # 返回各种索引，用于评估模型
    # 1. 高杠杆点 (应保留, Sx≈0, Sy≈0)
    params['idx_leverage_train'] = filter_idx(idx_leverage)
    # 2. 坏 X (应分离 Sx, Sx!=0)
    params['idx_bad_x_train'] = filter_idx(idx_bad_x)
    # 3. 坏 Y (应分离 Sy, Sy!=0)
    params['idx_bad_y_train'] = filter_idx(idx_bad_y)
    
    # 打印一些信息确认生成状态
    print(f"Data Generation Summary:")
    print(f"- Normal Std(X): {std_x_base:.4f}, Std(Y): {std_y_base:.4f}")
    print(f"- High Leverage: {len(params['idx_leverage_train'])} samples (Mag: {leverage_mag}x)")
    print(f"- Bad X Samples: {len(params['idx_bad_x_train'])} samples (Mag: {bad_x_mag}x)")
    print(f"- Bad Y Samples: {len(params['idx_bad_y_train'])} samples (Mag: {bad_y_mag}x)")

    return params



def gen_egg_data(**params):
    eeg = loadmat('./EGGData/EEG_Processed_Data.mat')
    fmri = loadmat('./EGGData/fMRI_Data_preproc.mat')
    data_eeg = eeg['mean_value'].T
    data_eeg = np.concatenate((data_eeg[:4, :, :], data_eeg[5:, :, :]), axis=0)
    data_fmri = fmri['ROI_DATA'].reshape(16, 10, 8)
    y, x = data_eeg, data_fmri

    N = params['N'] = 14
    params['L'] = 2
    params['M'] = 2
    params['dims'] = (10, 8, 37, 121)

    x_train, x_test = x[:N], x[N:]
    y_train, y_test = y[:N], y[N:]

    params['x'] = x_train
    params['y'] = y_train
    params['x_test'] = x_test
    params['y_test'] = y_test

    return params

