'''
Author: ziyuezi
Date: 2025-12-23 10:22:30
LastEditTime: 2026-01-07 12:27:10
FilePath: /rotr/experiment.py
Description: Reproduce Table 2 (Best/Min RPE)
'''
import data_io as io
import numpy as np
import os as os
import tensorly as tl
import pandas as pd
import time
from regressor import RPCA_tucker, RTOT_tucker, TOT_tucker, ROTR_fist_y,RPCA_Double,ttt
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '5' 

if __name__ == '__main__':
    # 实验设置
    replications = 15  # 重复次数，用于从中取最小值
    Ru_fixed = 135      # 表格中的 R 列
    mu1_fixed = 0.009
    
    # 表格的列：D 的不同取值

    bad_y_ratio_list = [ 0.03,0.05,0.07,0.1,0.12,0.15]
    #percentile_list = [0]
    # 存储结果结构: results[method][d_value] = [rpe_1, rpe_2, ..., rpe_rep]
    final_results = defaultdict(lambda: defaultdict(list))
    
    print(f"开始实验: R={Ru_fixed}, Replications={replications}, D_list={bad_y_ratio_list}")
    print("正在运行模型以寻找最小 RPE...")

    # 1. 遍历每一个 D (percentile)
    for percentile in bad_y_ratio_list:
        print(f"\nProcessing D={percentile} ...")
        
        # 2. 重复实验
        for r in range(replications):
            # 参数配置
            params = dict(
                R=Ru_fixed,  # 这里调cp秩
                #Ru=120,
                mu1=mu1_fixed,
                mu2=0.0005,
                mu3=1e-10,
                tol=1e-6,
                max_itr=30,
                ROTR_max_iter = 350,
                replications=replications,
                bad_y_ratio=percentile, # 当前的 D
                #percentile_x = 0.1,
                #outlier_mag = 20,
                #scale=10,
                #gamma = 2.0,

                tucker_ranks = [5,5,5,5], # 生成数据时的tucker秩
                init_scale_gen_core = 0.011,

                ranks = [5,5,5,5], # 预设的tucker秩
                lambda_x =0.001 ,
                lambda_y =  0.016,
                lambda_b =   0.06,
                fista_iter = 100 ,
                sx_max_step =  0.1,
                manifold_lr = 0.015 ,
                init_scale =  0.02 
            )
            
            # 生成数据
            params = io.gen_rotr_mixed_data(**params)
            

            # 定义需要对比的方法

            # models_list = [

            #     RPCA_Double(**params),
            #     #ROTR_fist_y(**params)


            # ]
            # models_list = [
            #     RPCA_tucker(**params),
            #     ROTR_fist_y(**params),
            # ]
            
            models_list = [
                
                # RPCA_Double(**params),
                ROTR_fist_y(**params),
                # RTOT_tucker(**params),
                # TOT_tucker(**params)

            ]

            for model in models_list:
                while True:
                    try:
                        rpe, B,sparsity_sx, sparsity_sy = model.fit(verbose=True) 


                        y_test = tl.partial_tensor_to_vec(params['y_test_gt'], skip_begin=1)
                        # 预测
                        y_pre = ttt(params['x_test_clean'], B, params['L'], params['dims'])
                        y_pre = tl.partial_tensor_to_vec(y_pre, skip_begin=1)
                        
                        # 均值对齐 (根据源代码逻辑)
                        m_test = np.mean(y_test, axis=1).reshape(-1, 1)
                        m_pre = np.mean(y_pre, axis=1).reshape(-1, 1)
                        y_pre = y_pre - m_pre + m_test 

                        # 计算误差
                        current_rpe = np.linalg.norm(y_pre - y_test) / np.linalg.norm(y_test)
                        print(f"{model.name} RPE: {current_rpe}")
                        # 存入列表
                        final_results[model.name][percentile].append(current_rpe)
                        break
                    except  Exception as e:
                        print(f"Error occurred: {e}")
                        print(f"正在重新运行 {model.name} ...\n")
                        time.sleep(1)

    # ==========================================
    # 打印最终表格 (Min RPE)
    # ==========================================
    print("\n" + "="*100)
    print(f"Table 2. Best Results (Minimum RPE over {replications} runs)")
    print("="*100)
    

    header = f"{'R':<5} {'Method':<10}"
    for p in bad_y_ratio_list:
        header += f"{f'D={p}':<12}" 
    print(header)
    print("-" * 100)

    method_order = ['rpca', 'rtot', 'tot','rotr'] # 指定打印顺序
    
    for method_name in method_order:
        # 如果某个方法没有跑，跳过
        if method_name not in final_results:
            continue
            
        row_str = f"{Ru_fixed:<5} {method_name:<10}"
        
        for p in bad_y_ratio_list:
            rpes = final_results[method_name][p]
            if rpes:

                best_rpe = np.min(rpes)
                

                cell_str = f"{best_rpe:.4f}"
            else:
                cell_str = "N/A"
            
            row_str += f"{cell_str:<12}"
        
        print(row_str)

    print("="*100)