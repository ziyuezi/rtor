import os
import re
import csv
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans

# ================= 配置参数 =================
#ROOT = r"D:\DOWNLOAD\EDGE_DOWNLOAD"  # 你的父目录
ROOT = r"D:\download\edge_download"
OUT_DIR = r"D:\fmri_results"  # 输出目录
RUNS = [
    "task001_run001", "task001_run002", "task001_run003",
    "task002_run001", "task002_run002", "task002_run003"
]
COPES = [f"cope{i}.nii.gz" for i in range(1, 9)]  # cope1 到 cope8


# ============================================

def find_subjects(root: str):
    """查找所有被试目录"""
    out = []
    pat = re.compile(r"^ds116_sub(\d+)$", re.IGNORECASE)
    for name in os.listdir(root):
        m = pat.match(name)
        if not m:
            continue
        num = int(m.group(1))
        ds_path = os.path.join(root, name)
        sub_name = f"sub{num:03d}"
        sub_path = os.path.join(ds_path, sub_name)
        if os.path.isdir(sub_path):
            out.append((sub_path, sub_name))
    out.sort(key=lambda x: int(x[1][3:]))
    return out


def get_standard_mask_indices(mask_path: str):
    """
    加载 FSL 的标准空间 mask，并利用 K-Means 将其聚类为 10 个空间脑区(ROIs)。
    返回每个 ROI 对应的坐标点字典，以确保所有被试使用绝对一致的特征提取模板。
    """
    print(f"正在基于 {mask_path} 生成 10 个空间的脑区模板...")
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    # 找到所有属于大脑的体素坐标 (x, y, z)
    coords = np.argwhere(mask_data > 0)

    # 将这些体素分为 10 个空间簇
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    # 创建字典，存放每个区域(0-9)对应的体素坐标
    roi_indices = {i: coords[labels == i] for i in range(10)}
    return roi_indices


def extract_fmri_feature(sub_dir: str, roi_indices: dict) -> np.ndarray:
    """
    提取一个被试的 fMRI 特征，形状为 (10, 8)
    10: 空间脑区
    8: 实验对比度(COPEs)
    """
    fmri_tensor = np.zeros((10, 8))
    model_root = os.path.join(sub_dir, "model", "model001")

    for c_idx, cope_file in enumerate(COPES):
        run_copes = []
        for run in RUNS:
            # 【关键】必须使用 reg_standard 下的 stats，这样所有被试的影像空间大小才对齐
            cope_path = os.path.join(model_root, f"{run}.feat", "reg_standard", "stats", cope_file)

            if os.path.isfile(cope_path):
                data = nib.load(cope_path).get_fdata()
                run_copes.append(data)
            else:
                pass  # 如果某个 run 缺失，则跳过

        if not run_copes:
            raise FileNotFoundError(f"该被试没有任何有效的 {cope_file}")

        # 将该被试 6 个 Run 的当前 COPE 影像做平均
        avg_cope = np.mean(run_copes, axis=0)

        # 提取 10 个 ROI 的均值
        for roi_idx in range(10):
            coords = roi_indices[roi_idx]
            # 根据索引提取这些坐标上的激活值
            vals = avg_cope[coords[:, 0], coords[:, 1], coords[:, 2]]
            # 求该区域的平均激活度，填入张量
            fmri_tensor[roi_idx, c_idx] = np.nanmean(vals)

    return fmri_tensor


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subs = find_subjects(ROOT)
    if not subs:
        raise RuntimeError(f"在 {ROOT} 下未找到 ds116_subXXX 文件夹。")

    # 1. 寻找第一个存在的标准 mask 作为全组 10 个脑区的基准划分模板
    roi_indices = None
    for sub_path, sub_id in subs:
        mask_path = os.path.join(sub_path, "model", "model001", "task001_run001.feat", "reg_standard", "mask.nii.gz")
        if os.path.isfile(mask_path):
            roi_indices = get_standard_mask_indices(mask_path)
            break

    if roi_indices is None:
        raise RuntimeError("未能找到任何 reg_standard/mask.nii.gz 文件！")

    # 2. 正式提取所有被试的特征
    manifest_path = os.path.join(OUT_DIR, "manifest_fmri.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sub_id", "status", "out_file", "error"])

        for sub_path, sub_id in subs:
            try:
                print(f"==== 正在处理 fMRI: {sub_id} ====")
                feat = extract_fmri_feature(sub_path, roi_indices)

                # 检查形状是否符合论文的 (10, 8)
                assert feat.shape == (10, 8), f"形状错误: {feat.shape}"

                out_file = os.path.join(OUT_DIR, f"{sub_id}_fmri.npy")
                np.save(out_file, feat)
                print(f"保存成功: {out_file}")

                w.writerow([sub_id, "OK", out_file, ""])
            except Exception as e:
                print(f"处理失败: {sub_id} -> {repr(e)}")
                w.writerow([sub_id, "FAILED", "", repr(e)])

    print("\n全部完成！")


if __name__ == "__main__":
    main()