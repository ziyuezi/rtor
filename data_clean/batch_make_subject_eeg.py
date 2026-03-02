import os
import re
import csv
import numpy as np
from scipy.io import loadmat

# ====== 你可以按需改这些参数 ======
FS = 1000.0          # 你已经验证过是对的
TMIN = -0.1
TMAX = 0.6
N_TIME = 121

RUNS = [
    "task001_run001",
    "task001_run002",
    "task001_run003",
    "task002_run001",
    "task002_run002",
    "task002_run003",
]
COND_FILES = ["cond001.txt", "cond002.txt", "cond003.txt"]


def load_onsets(txt_path: str) -> np.ndarray:
    """FSL onset format: onset_sec duration_sec amplitude"""
    arr = np.loadtxt(txt_path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 0].astype(float)


def extract_run_feature(eeg_mat: str, onset_dir: str,
                        fs: float = FS, tmin: float = TMIN, tmax: float = TMAX,
                        n_time: int = N_TIME) -> np.ndarray:
    m = loadmat(eeg_mat)

    # 自动找 37×T 的 EEG（你这里就是 data_reref）
    X = None
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == 37:
            X = v.astype(float)
            break
    if X is None:
        raise RuntimeError(f"No 37×T EEG array found in: {eeg_mat}")

    n0 = int(round(tmin * fs))
    n1 = int(round(tmax * fs))
    W = n1 - n0

    feats = []
    for cf in COND_FILES:
        p = os.path.join(onset_dir, cf)
        onsets = load_onsets(p)

        epochs = []
        for s in onsets:
            center = int(round(s * fs))
            a = center + n0
            b = center + n1
            if a >= 0 and b <= X.shape[1]:
                epochs.append(X[:, a:b])

        if not epochs:
            raise RuntimeError(f"No valid epochs for {p} (fs/onsets mismatch?)")

        E = np.stack(epochs, axis=2)      # (37, W, n_trials)
        erp = E.mean(axis=2)              # (37, W)

        # 重采样到 121
        t_old = np.linspace(0.0, 1.0, W)
        t_new = np.linspace(0.0, 1.0, n_time)
        erp121 = np.vstack([np.interp(t_new, t_old, erp[ch]) for ch in range(37)])
        feats.append(erp121)

    return np.mean(feats, axis=0)  # (37,121)


def extract_subject_feature(sub_dir: str) -> np.ndarray:
    """
    sub_dir: ...\ds116_sub010\sub010
    """
    eeg_root = os.path.join(sub_dir, "EEG")
    onset_root = os.path.join(sub_dir, "model", "model001", "onsets")

    run_feats = []
    for run in RUNS:
        eeg_mat = os.path.join(eeg_root, run, "EEG_rereferenced.mat")
        onset_dir = os.path.join(onset_root, run)

        if not os.path.isfile(eeg_mat):
            raise FileNotFoundError(f"Missing EEG mat: {eeg_mat}")
        for cf in COND_FILES:
            if not os.path.isfile(os.path.join(onset_dir, cf)):
                raise FileNotFoundError(f"Missing onset file: {os.path.join(onset_dir, cf)}")

        rf = extract_run_feature(eeg_mat, onset_dir)
        run_feats.append(rf)

    return np.mean(run_feats, axis=0)  # (37,121)


def find_subjects(root: str):
    """
    root contains folders like ds116_sub010
    returns list of tuples: (ds_folder, sub_folder, sub_id)
    """
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
            out.append((ds_path, sub_path, sub_name))
    # 按被试编号排序
    out.sort(key=lambda x: int(x[2][3:]))
    return out


def main():
    # ====== 你只需要改这两个路径 ======
    ROOT = r"D:\download\edge_download"   # 你截图里的父目录
    OUT_DIR = r"D:\eeg_results"           # 输出每个被试的小文件到这里
    # =================================

    os.makedirs(OUT_DIR, exist_ok=True)

    subs = find_subjects(ROOT)
    if not subs:
        raise RuntimeError(f"No ds116_subXXX found under {ROOT}")

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sub_id", "sub_path", "status", "out_file", "error"])

        for _, sub_path, sub_id in subs:
            try:
                print(f"==== Processing {sub_id} ====")
                feat = extract_subject_feature(sub_path)
                assert feat.shape == (37, 121)

                out_file = os.path.join(OUT_DIR, f"{sub_id}_eeg.npy")
                np.save(out_file, feat)

                print("Saved:", out_file)
                w.writerow([sub_id, sub_path, "OK", out_file, ""])
            except Exception as e:
                print("FAILED:", sub_id, "->", repr(e))
                w.writerow([sub_id, sub_path, "FAILED", "", repr(e)])

    print("\nDone. Manifest:", manifest_path)
    print("Tip: copy all *_eeg.npy from both computers into one folder, then merge.")


if __name__ == "__main__":
    main()
