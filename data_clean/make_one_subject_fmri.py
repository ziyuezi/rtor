import os
import numpy as np
import nibabel as nib
from scipy.io import savemat

RUNS = [
    "task001_run001",
    "task001_run002",
    "task001_run003",
    "task002_run001",
    "task002_run002",
    "task002_run003",
]

def load_mask(feat_dir: str):
    """Prefer FEAT mask.nii.gz"""
    mask_path = os.path.join(feat_dir, "mask.nii.gz")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Missing mask: {mask_path}")
    m = nib.load(mask_path).get_fdata()
    return m > 0

def zstat_quantile_features(stats_dir: str, mask_bool: np.ndarray,
                            n_contrasts: int = 8, n_feats_per_contrast: int = 10) -> np.ndarray:
    """
    For each zstat{i}, compute 10 quantiles within mask -> 10 dims.
    Concatenate 8 contrasts -> 80 dims.
    """
    qs = np.linspace(0.05, 0.95, n_feats_per_contrast)  # 10分位点
    feats = []

    for i in range(1, n_contrasts + 1):
        zpath = os.path.join(stats_dir, f"zstat{i}.nii.gz")
        if not os.path.isfile(zpath):
            raise FileNotFoundError(f"Missing: {zpath}")

        z = nib.load(zpath).get_fdata()
        if z.shape != mask_bool.shape:
            raise RuntimeError(f"Shape mismatch zstat{i}: {z.shape} vs mask: {mask_bool.shape}")

        v = z[mask_bool].astype(float)
        # 清理 NaN/Inf
        v = v[np.isfinite(v)]
        if v.size < 1000:
            raise RuntimeError(f"Too few voxels in mask for zstat{i}: {v.size}")

        feats.append(np.quantile(v, qs))

    out = np.concatenate(feats, axis=0)  # (8*10,) = (80,)
    return out

def extract_subject_roi_data(sub_dir: str) -> np.ndarray:
    """
    sub_dir like: ...\ds116_sub010\sub010
    returns (80,) feature vector averaged across 6 runs
    """
    model_root = os.path.join(sub_dir, "model", "model001")
    run_feats = []

    for run in RUNS:
        feat_dir = os.path.join(model_root, f"{run}.feat")
        stats_dir = os.path.join(feat_dir, "stats")
        if not os.path.isdir(stats_dir):
            raise FileNotFoundError(f"Missing stats dir: {stats_dir}")

        mask_bool = load_mask(feat_dir)
        f80 = zstat_quantile_features(stats_dir, mask_bool, n_contrasts=8, n_feats_per_contrast=10)
        assert f80.shape == (80,)
        run_feats.append(f80)
        print("OK:", run, f80.mean(), f80.std())

    sub80 = np.mean(run_feats, axis=0)  # (80,)
    print("SUB feature:", sub80.shape, sub80.mean(), sub80.std())
    return sub80

if __name__ == "__main__":
    sub010 = r"D:\download\edge_download\ds116_sub010\sub010"  # 改成你的
    sub80 = extract_subject_roi_data(sub010)
    print(np.isfinite(sub80).all())
