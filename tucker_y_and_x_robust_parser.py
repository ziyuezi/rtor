import data_io as io
import numpy as np
import os
import tensorly as tl
import pandas as pd
import time
import argparse  # 新增
from regressor import RPCA_tucker, RTOT_tucker, TOT_tucker, ROTR_fist_y, RPCA_Double, ttt
from collections import defaultdict

# 定义模型映射，方便命令行调用
MODEL_MAP = {
    'rpca': RPCA_tucker,
    'rtot': RTOT_tucker,
    'tot': TOT_tucker,
    'rotr': ROTR_fist_y,
    'rpca_double': RPCA_Double
}


def get_args():
    parser = argparse.ArgumentParser(description="Reproduce Table 2 Experiment")

    # 实验基础设置
    parser.add_argument('--replications', type=int, default=15, help='Number of replications')
    parser.add_argument('--gpu', type=str, default='5', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--bad_y_ratio_list', type=float, nargs='+',
                        default=[0.03, 0.05, 0.07, 0.1, 0.12, 0.15], help='List of D values')

    # 模型选择 (支持传入多个)
    parser.add_argument('--models', type=str, nargs='+', default=['rotr'],
                        choices=list(MODEL_MAP.keys()), help='Methods to run')

    # 超参数设置
    parser.add_argument('--R', type=int, default=135, help='CP Rank R')
    parser.add_argument('--mu1', type=float, default=0.009)
    parser.add_argument('--mu2', type=float, default=0.0005)
    parser.add_argument('--mu3', type=float, default=1e-10)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--max_itr',type=int,default=30)
    parser.add_argument('--ROTR_max_iter', type=int, default=350)

    # 张量相关参数
    parser.add_argument('--tucker_ranks', type=int, nargs='+', default=[5, 5, 5, 5])
    parser.add_argument('--ranks', type=int, nargs='+', default=[5, 5, 5, 5])
    parser.add_argument('--init_scale_gen_core',type=float,default=0.11)
    # 正则化系数
    parser.add_argument('--lambda_x', type=float, default=0.001)
    parser.add_argument('--lambda_y', type=float, default=0.016)
    parser.add_argument('--lambda_b', type=float, default=0.06)

    # 其他优化参数
    parser.add_argument('--fista_iter', type=int, default=100)
    parser.add_argument('--sx_max_step', type=float, default=0.1)
    parser.add_argument('--manifold_lr', type=float, default=0.015)
    parser.add_argument('--init_scale', type=float, default=0.02)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    final_results = defaultdict(lambda: defaultdict(list))

    print(f"开始实验: R={args.Ru}, Replications={args.replications}, D_list={args.bad_y_ratio_list}")
    print(f"运行模型: {args.models}")

    for percentile in args.bad_y_ratio_list:
        print(f"\nProcessing D={percentile} ...")

        for r in range(args.replications):
            # 将 args 转为字典并构建参数
            params = dict(
                R=args.R,
                mu1=args.mu1,
                mu2=args.mu2,
                mu3=args.mu3,
                tol=args.tol,
                max_itr=args.max_itr,
                ROTR_max_iter=args.ROTR_max_iter,
                replications=args.replications,
                bad_y_ratio=percentile,
                tucker_ranks=args.tucker_ranks,
                init_scale_gen_core=args.init_scale_gen_core,
                ranks=args.ranks,
                lambda_x=args.lambda_x,
                lambda_y=args.lambda_y,
                lambda_b=args.lambda_b,
                fista_iter=args.fista_iter,
                sx_max_step=args.sx_max_step,
                manifold_lr=args.manifold_lr,
                init_scale=args.init_scale
            )

            # 生成数据
            params = io.gen_rotr_mixed_data(**params)

            # 根据命令行参数动态实例化模型
            models_to_run = []
            for m_name in args.models:
                model_class = MODEL_MAP[m_name]
                models_to_run.append(model_class(**params))

            for model in models_to_run:
                while True:
                    try:
                        rpe, B, sparsity_sx, sparsity_sy = model.fit(verbose=True)

                        y_test = tl.partial_tensor_to_vec(params['y_test_gt'], skip_begin=1)
                        y_pre = ttt(params['x_test_clean'], B, params['L'], params['dims'])
                        y_pre = tl.partial_tensor_to_vec(y_pre, skip_begin=1)

                        m_test = np.mean(y_test, axis=1).reshape(-1, 1)
                        m_pre = np.mean(y_pre, axis=1).reshape(-1, 1)
                        y_pre = y_pre - m_pre + m_test

                        current_rpe = np.linalg.norm(y_pre - y_test) / np.linalg.norm(y_test)
                        print(f"{model.name} RPE: {current_rpe}")
                        final_results[model.name][percentile].append(current_rpe)
                        break
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        print(f"正在重新运行 {model.name} ...\n")
                        time.sleep(1)

    # 打印最终表格
    print("\n" + "=" * 100)
    print(f"Table 2. Best Results (Minimum RPE over {args.replications} runs)")
    print("=" * 100)

    header = f"{'R':<5} {'Method':<10}"
    for p in args.bad_y_ratio_list:
        header += f"{f'D={p}':<12}"
    print(header)
    print("-" * 100)

    # 按照请求的模型顺序或输入顺序打印
    for method_name in args.models:
        # 注意：model.name 可能和映射里的 key 不完全一致，这里建议统一
        # 简单处理：遍历结果字典里匹配的项
        matching_names = [k for k in final_results.keys() if method_name in k.lower()]
        for actual_name in matching_names:
            row_str = f"{args.Ru:<5} {actual_name:<10}"
            for p in args.bad_y_ratio_list:
                rpes = final_results[actual_name][p]
                cell_str = f"{np.min(rpes):.4f}" if rpes else "N/A"
                row_str += f"{cell_str:<12}"
            print(row_str)

    print("=" * 100)