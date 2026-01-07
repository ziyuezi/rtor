'''
Author: ziyuezi
Date: 2025-12-23 10:22:30
LastEditTime: 2026-01-04 21:16:58
FilePath: /rotr/experiment.py
Description: Reproduce Table 2 (Best/Min RPE)
'''
import data_io as io
import numpy as np
import os as os
import tensorly as tl
import pandas as pd
import time
from regressor import RPCA, RTOT, TOT, ROTR_fist_y, ttt 
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '3' 

if __name__ == '__main__':

    replications = 10 
    Ru_fixed = 7      
    mu1_fixed = 0.009
    
    # 表格的列：D 的不同取值
    percentile_list = [0, 0.03, 0.05, 0.1, 0.15]
    
    # 存储结果结构: results[method][d_value] = [rpe_1, rpe_2, ..., rpe_rep]
    final_results = defaultdict(lambda: defaultdict(list))
    
    print(f"开始实验: R={Ru_fixed}, Replications={replications}, D_list={percentile_list}")
    print("正在运行模型以寻找最小 RPE...")

    # 1. 遍历每一个 D (percentile)
    for percentile in percentile_list:
        print(f"\nProcessing D={percentile} ...")
        
        # 2. 重复实验
        for r in range(replications):
            # 参数配置
            params = dict(
                R=Ru_fixed,
                Ru=7,   
                mu1=mu1_fixed,
                mu2=0.0005,
                mu3=1e-10,
                tol=1e-6,
                max_itr=1000,
                replications=replications,
                percentile=percentile, # 当前的 D
                percentile_x = 0.1,
                outlier_mag = 20,
                scale=10,
                gamma = 2.0,

                tucker_ranks = [5,5,5,5],
                init_scale_gen_core = 0.011,

                ranks = [7,7,7,7],
                lambda_x =1 ,
                lambda_y =  0.014,
                lambda_b =   0.057,
                fista_iter = 10 ,
                sx_max_step =  0.1,
                manifold_lr = 0.016 ,
                init_scale =  0.01 
            )
            
            # 生成数据
            params = io.gen_sync_data_norm(**params)
            
            # 定义需要对比的方法
            models_list = [
                TOT(**params),
                RPCA(**params),
            ]

            for model in models_list:

                rpe, B,sparsity_sx, sparsity_sy = model.fit(verbose=False) 


                y_test = tl.partial_tensor_to_vec(params['y_test'], skip_begin=1)
                # 预测
                y_pre = ttt(params['x_test'], B, params['L'], params['dims'])
                y_pre = tl.partial_tensor_to_vec(y_pre, skip_begin=1)
                
                # 均值对齐 (根据源代码逻辑)
                m_test = np.mean(y_test, axis=1).reshape(-1, 1)
                m_pre = np.mean(y_pre, axis=1).reshape(-1, 1)
                y_pre = y_pre - m_pre + m_test 

                # 计算误差
                current_rpe = np.linalg.norm(y_pre - y_test) / np.linalg.norm(y_test)
                
                # 存入列表
                final_results[model.name][percentile].append(current_rpe)
                

    # ==========================================
    # 打印最终表格 (Min RPE)
    # ==========================================
    print("\n" + "="*100)
    print(f"Table 2. Best Results (Minimum RPE over {replications} runs)")
    print("="*100)
    

    header = f"{'R':<5} {'Method':<10}"
    for p in percentile_list:
        header += f"{f'D={p}':<12}" 
    print(header)
    print("-" * 100)

    method_order = ['rpca', 'rtot', 'tot','rotr'] # 指定打印顺序
    
    for method_name in method_order:
        # 如果某个方法没有跑，跳过
        if method_name not in final_results:
            continue
            
        row_str = f"{Ru_fixed:<5} {method_name:<10}"
        
        for p in percentile_list:
            rpes = final_results[method_name][p]
            if rpes:

                best_rpe = np.min(rpes)
                

                cell_str = f"{best_rpe:.4f}"
            else:
                cell_str = "N/A"
            
            row_str += f"{cell_str:<12}"
        
        print(row_str)
    
    print("="*100)