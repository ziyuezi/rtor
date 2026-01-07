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
函数1.ttt 定义核心算子，张量收缩积
2.mcp函数。没调用，先不管
3.gen_lambda_data ，gen_egg_data 加载真实数据
4.合成数据生成。两个，一个是gen_sync_data
一个是：gen_sync_data_norm

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
    N = params['N'] = 500  # 样本量
    L = params['L'] = 2  # x的后两个维度是收缩维度
    M = params['M'] = 2
    dims = params['dims'] = (15, 20, 5, 10) # 这个维度是规定的谁？ 系数张量的
    percentile = params['percentile'] # 注入异常值的比例
    Ru = params['Ru']  # 这个是啥参数？ 系数张量的cp秩
    # scale = params['scale'] # 这个是啥参数？

    rns = np.random.RandomState(seed=51) # normal seed=1  创建一个随机数生成对象
    base = [rns.normal(size=(p, 5)) for p in dims[:L]] # 生成15 * 5 以及 20 * 5 的两个矩阵。base现在有两个矩阵
    x = [[np.random.uniform(1e-3, 1) * b for b in base] for _ in range(1, N + 1)]
    # 内层先对生成的两个矩阵缩放。 生成500个样本。 x[0]:第一个样本，包含两个因子矩阵
    x = [tl.cp_to_tensor((None, t)) for t in x] # 生成二阶张量。500个二阶张量 x是一个list



    # rns = np.random.RandomState(seed=5) this one is for normal
    rns = np.random.RandomState(seed=42)
    b = [rns.normal(size=(p, Ru)) for p in dims]  # 生成4个因子矩阵。dims[] * 3
    b = tl.cp_to_tensor((None, b)) # 生成4阶张量。维度：(15, 20, 5, 10) 。秩为3，cp秩为3
    b = b / np.linalg.norm(b) # 归一化
    y = ttt(x, b, L, dims) # 保留了第一个维度 (N,5,10)

    e = np.random.normal(loc=0, scale=1, size=y.shape) # 生成正态分布的y
    e = e / np.linalg.norm(e) # 归一化
    y += e # 叠加噪声的y

    # y = tl.partial_tensor_to_vec(y, skip_begin=1)
    # df = {'y': y.flatten(), 'time': [i for i in range(dims[-1]*dims[-2])] * N, 'n': [i for i in range(N) for _ in range(dims[-1]*dims[-2])]}
    # fig = px.line(df, x='time', y='y', line_group='n', color='n')
    # fig.show()

    x_test, x = x[:100], x[100:] # 划分测试集和训练集
    y_test, y = y[:100], y[100:] # 注意：训练集被污染，测试集是正常

    if percentile > 0:
        y = tl.partial_tensor_to_vec(y, skip_begin=1)
        indices = np.random.randint(0, 400, size=int(400 * percentile)) # 抽取要被污染的行
        outlier_idx = {i: [] for i in indices} # 记录异常值的样本索引
        # y_with_outlier = y[indices]
        for n in indices:
            idx = np.random.randint(0, 40) # 选择破坏点
            outlier_idx[n] = [i for i in range(idx, idx + 5)] # 记录受损位置
            # y_[idx] = scale * np.sign(y_[idx]) * max(y_)
            y[n][idx: idx + 5] = np.random.uniform(.8, 2, size=5)  # 将5个连续点的数值设为异常
        # y[indices] = y_with_outlier
        y = y.reshape(-1, dims[-2], dims[-1]) # 还原形状
        params['s_idx'] = outlier_idx # 记录了稀疏异常位置的字典


    # tmp = tl.partial_tensor_to_vec(y, skip_begin=1)
    # df = {'y': tmp.flatten(), 'time': [i for i in range(dims[-1]*dims[-2])] * 400, 'n': [i for i in range(400) for _ in range(dims[-1]*dims[-2])]}
    # fig = px.line(df, x='time', y='y', line_group='n', color='n')
    # fig.show()
    # # fig.write_html('./training-data.html')
    #
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
    init_scale = params['init_scale_gen_core'] # 控制核心张量数值大小 先不要这个

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
        core_x = rns_x.uniform(0, 1, size=ranks_in) # 随机核心  这里的缩放范围可以调整
        # 调大左边，减小右边，都没效果
        x_i = tucker_to_tensor((core_x, factors_X_shared))
        X_list.append(x_i)
        
    X = np.array(X_list) # (N, 15, 20)

    # ==========================================
    # 3. 计算真实响应并添加稠密噪声
    # ==========================================

    y = ttt(X, B_true, L, dims) # (N, 5, 10)


    e = np.random.normal(loc=0, scale=1, size=y.shape) # 这里的噪声生成范围就别动了
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
            # 选择破坏起始点，确保 idx+5 不越界
            # 原代码是 idx = np.random.randint(0, 40)，这里做适应性调整
            # 注意这里做了适应性调整
            max_start_idx = max(0, feature_len - 5)
            idx = np.random.randint(0, max_start_idx if max_start_idx > 0 else 1)
            
            outlier_idx[n] = [i for i in range(idx, idx + 5)] # 记录受损位置
            
            # 核心修改：注入均匀分布异常 [0.8, 2]
            # y_[idx] ... 逻辑复刻
            y_flat[n][idx: idx + 5] = np.random.uniform(0.8, 2, size=5)
            # 
            
        # 3. 还原形状
        # 原代码：y = y.reshape(-1, dims[-2], dims[-1])
        # 这里使用 *output_dims 以适配任意输出维度
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
    init_scale = params['init_scale_gen_core'] # 控制核心张量数值大小 先不要这个
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
        core_x = rns_x.uniform(0, 1, size=ranks_in) # 随机核心  这里的缩放范围可以调整
        # 调大左边，减小右边，都没效果
        x_i = tucker_to_tensor((core_x, factors_X_shared))
        X_list.append(x_i)
        
    X = np.array(X_list) # (N, 15, 20)

    # ==========================================
    # 3. 计算真实响应并添加稠密噪声
    # ==========================================

    y = ttt(X, B_true, L, dims) # (N, 5, 10)


    e = np.random.normal(loc=0, scale=1, size=y.shape) # 这里的噪声生成范围就别动了  scale 可以调小
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
            # 原代码是 idx = np.random.randint(0, 40)，这里做适应性调整
            # 注意这里做了适应性调整
            max_start_idx = max(0, feature_len - 5)
            idx = np.random.randint(0, max_start_idx if max_start_idx > 0 else 1)
            
            outlier_idx[n] = [i for i in range(idx, idx + 5)] # 记录受损位置
            
            # 核心修改：注入均匀分布异常 [0.8, 2]
            # y_[idx] ... 逻辑复刻
            y_flat[n][idx: idx + 5] = np.random.uniform(0.8, 2, size=5)
            # 
            
        # 3. 还原形状
        # 原代码：y = y.reshape(-1, dims[-2], dims[-1])
        # 这里使用 *output_dims 以适配任意输出维度
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
        # 基础核心 U[0, 1]
        core_x = rns_x.uniform(0, 1, size=tucker_ranks[:L])
        
        # [关键修改]：如果是高杠杆点，放大核心张量
        if n in idx_leverage:
            # 放大核心 = 放大生成的 X，但保持低秩结构不变
            core_x *= leverage_mag 
            
        x_i = tucker_to_tensor((core_x, factors_X_shared))
        X_list.append(x_i)
        
    X = np.array(X_list) 

    # 记录此时 X 的统计量，用于后续生成 bad outlier 的尺度
    std_x_clean = np.std(X[idx_normal]) 

    # ==========================================
    # 3. 计算 Y (自然传播，无需手动修改)
    # ==========================================
    # 因为 idx_leverage 的 X 很大，算出来的 Y 也会很大
    # 且完全满足 Y = X * B，所以它们是“好数据”
    y_clean = ttt(X, B_true, L, dims) 
    
    # 添加背景稠密噪声 (基于正常样本的标准差，避免被高杠杆点拉偏)
    std_y_normal = np.std(y_clean[idx_normal])
    dense_noise = np.random.normal(0, noise_level * std_y_normal, size=y_clean.shape)
    y = y_clean + dense_noise

    # ==========================================
    # 4. 划分数据集
    # ==========================================
    # 简单起见，这里先处理全集，最后再分。或者你可以保留你原来的分割逻辑。
    # 这里保持你的风格，先生成全量，最后再 split

    # ==========================================
    # 5. 注入“真实异常” (Bad Data)
    # ==========================================
    # 这些是真正的破坏者，模型应该把它们剔除 (Mask掉 或 放入S)
    
    # 情况 A: 破坏 X (模拟传感器坏了，导致 X 乱了，但 Y 没对应上)
    # 或者 情况 B: 破坏 Y (模拟记录错误)
    
    # 这里演示破坏 Y (Block Outlier 或 Sparse Outlier)
    if num_out > 0:
        # 异常强度：基于高杠杆样本的强度，混淆视听
        # 让坏点的幅度跟高杠杆点差不多大，看模型能不能分得清
        bad_mag = leverage_mag * std_y_normal 
        
        y_flat = y.reshape(N, -1)
        feat_len = y_flat.shape[1]
        
        for n in idx_outlier:
            # 随机选一段注入巨大噪声
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
    
    # 返回索引以便验证：
    # idx_leverage 中的样本，模型应当保留 (Sy ≈ 0)
    # idx_outlier 中的样本，模型应当剔除 (Sy != 0)
    # 注意调整索引以匹配切分后的训练集
    train_indices = np.arange(test_size, N)
    
    # 筛选出在训练集里的那些特殊索引
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
    
    # 确保索引互斥，以便清晰验证 (实际情况可能重叠，但测试时互斥更清晰)
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
    # 计算统计量 (基于正常样本，作为破坏尺度的基准)
    std_x_base = np.std(X[idx_normal])

    # ==========================================
    # 5. 生成 Y (基于当前的合法 X)
    # ==========================================
    # 注意：此时的 X 包含高杠杆点，但没有坏点。
    # 所以 Y = X * B 对于所有样本(包括高杠杆)都是成立的。
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
    # 逻辑：在 Y 生成之后破坏 X。这样 X 变了，但 Y 没变，导致 X 无法解释 Y。
    if n_bad_x > 0:
        block_h, block_w = 5, 6
        # 异常强度：基于正常 X 的波动 * 倍率
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
    # 逻辑：在 Y 上叠加稀疏噪声。这样 Y 变了，X 无法解释 Y。
    if n_bad_y > 0:
        # 异常强度：基于正常 Y 的波动 * 倍率
        # (也可以用 leverage_mag 混淆视听，看模型是否只剔除不相关的)
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

    # 辅助函数：筛选训练集中的索引，方便后续验证
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


if __name__ == '__main__':
    # params = dict()
    # params = gen_lambda_data(**params)
    # x, y = params['x_test'], params['y_test']
    #
    # arr = np.array([y.flatten(), [i for i in range(1, 204)] * y.shape[0], [n for n in range(1, y.shape[0] + 1) for _ in range(203)]]).T
    # df = pd.DataFrame(data=arr, columns=['y', 'time', 'type'])
    # fig = px.line(df, x='time', y='y', line_group='type', color='type')
    # fig.show()
    # print(x.shape, y.shape)
    # print('done')

    params = dict(
        R=15,
        Ru=3,
        mu1=6.5e-3,
        mu2=3.5e-3,
        mu3=1e-8,
        tol=1e-4,
        max_itr=20,
        replications=20,
        percentile=.15,
        scale=2
    )
    # gen_sync_data(**params)
    params = gen_sync_data_norm(**params)
    print(params['s_idx'])
    # gen_lambda_data(**params)
    # gen_egg_data(**params)
