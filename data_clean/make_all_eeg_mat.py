import os
import numpy as np
from scipy.io import savemat

# 直接复用你上一个脚本里的 extract_subject_feature
from make_one_subject_eeg import extract_subject_feature

if __name__ == "__main__":
    # 这里填你放 16 个被试的根目录（建议你把 ds116_sub001...sub016 都解压到同一个父目录下）
    root = r"D:\download\edge_download"

    subs = []
    for i in range(1, 17):
        sid = f"sub{i:03d}"
        # 你现在的目录是 ds116_sub010\sub010 这种结构
        sub_dir = os.path.join(root, f"ds116_{sid}", sid)
        if not os.path.isdir(sub_dir):
            raise RuntimeError(f"Missing: {sub_dir}")
        subs.append(sub_dir)

    eeg_all = []
    for sub_dir in subs:
        print("====", sub_dir)
        eeg_all.append(extract_subject_feature(sub_dir, fs=1000.0))

    eeg_all = np.stack(eeg_all, axis=0)  # (16,37,121)

    # 按作者代码：mean_value.T -> (16,37,121)
    # 所以 mean_value 存成 (37,121,16)
    mean_value = np.transpose(eeg_all, (1, 2, 0))

    out_dir = os.path.join(os.getcwd(), "EGGData")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "EEG_Processed_Data.mat")
    savemat(out_path, {"mean_value": mean_value})

    print("Saved:", out_path, "mean_value shape:", mean_value.shape)
