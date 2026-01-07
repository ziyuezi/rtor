import pickle
import numpy as np

path_rpes = r'./experiment-results/test-run/dict_rpes.p'
path_params = r'./experiment-results/test-run/list_params.p'

print("=" * 30)
print("1. 查看误差结果 (dict_rpes.p)")
print("=" * 30)

with open(path_rpes, 'rb') as f:
    rpes = pickle.load(f)
    for model_name, errors in rpes.items():
        print(f"模型: {model_name}")
        print(f"  -> 误差列表: {errors}")
        print(f"  -> 平均误差: {np.mean(errors):.4f}")
        print("-" * 20)

print("\n" + "=" * 30)
print("2. 查看实验参数 (list_params.p)")
print("=" * 30)

with open(path_params, 'rb') as f:
    params_list = pickle.load(f)
    first_experiment = params_list[0]

    # 打印所有的 Key
    print(f"参数 keys: {list(first_experiment.keys())}")

    # 修改后的打印逻辑
    for key in ['x', 'y', 'x_test', 'y_test']:
        if key in first_experiment:
            data = first_experiment[key]
            # 如果是列表，打印长度和第一个元素的形状
            if isinstance(data, list):
                count = len(data)
                sample_shape = data[0].shape if count > 0 else "N/A"
                print(f"{key}: 这是一个列表，包含 {count} 个样本，每个样本形状为 {sample_shape}")
            # 如果是 NumPy 数组，直接打印形状
            elif hasattr(data, 'shape'):
                print(f"{key}: 这是一个数组，总形状为 {data.shape}")

    print("-" * 20)
    print(f"秩 (R): {first_experiment.get('R')}")
    print(f"维度设置 (dims): {first_experiment.get('dims')}")
    print(f"异常值比例 (percentile): {first_experiment.get('percentile')}")