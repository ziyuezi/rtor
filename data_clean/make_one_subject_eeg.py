import os
import numpy as np
from scipy.io import loadmat

def load_onsets(txt_path: str) -> np.ndarray:
    arr = np.loadtxt(txt_path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 0].astype(float)

def extract_run_feature(eeg_mat: str, onset_dir: str, fs: float = 1000.0,
                        tmin: float = -0.1, tmax: float = 0.6, n_time: int = 121):
    m = loadmat(eeg_mat)
    X = None
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == 37:
            X = v.astype(float)
            break
    if X is None:
        raise RuntimeError(f"No 37×T EEG array found in {eeg_mat}")

    n0 = int(round(tmin * fs))
    n1 = int(round(tmax * fs))
    W = n1 - n0

    feats = []
    for cf in ["cond001.txt", "cond002.txt", "cond003.txt"]:
        p = os.path.join(onset_dir, cf)
        onsets = load_onsets(p)

        epochs = []
        for s in onsets:
            center = int(round(s * fs))
            a = center + n0
            b = center + n1
            if a >= 0 and b <= X.shape[1]:
                epochs.append(X[:, a:b])

        E = np.stack(epochs, axis=2)      # (37, W, n_trials)
        erp = E.mean(axis=2)              # (37, W)

        # resample to 121
        t_old = np.linspace(0.0, 1.0, W)
        t_new = np.linspace(0.0, 1.0, n_time)
        erp121 = np.vstack([np.interp(t_new, t_old, erp[ch]) for ch in range(37)])
        feats.append(erp121)

    return np.mean(feats, axis=0)  # (37,121)

def extract_subject_feature(sub_dir: str, fs: float = 1000.0) -> np.ndarray:
    """
    sub_dir like: ...\ds116_sub010\sub010
    """
    eeg_root = os.path.join(sub_dir, "EEG")
    onset_root = os.path.join(sub_dir, "model", "model001", "onsets")

    runs = [
        ("task001_run001", "task001_run001"),
        ("task001_run002", "task001_run002"),
        ("task001_run003", "task001_run003"),
        ("task002_run001", "task002_run001"),
        ("task002_run002", "task002_run002"),
        ("task002_run003", "task002_run003"),
    ]

    run_feats = []
    for eeg_run, onset_run in runs:
        eeg_mat = os.path.join(eeg_root, eeg_run, "EEG_rereferenced.mat")
        onset_dir = os.path.join(onset_root, onset_run)
        rf = extract_run_feature(eeg_mat, onset_dir, fs=fs)
        run_feats.append(rf)
        print("OK:", eeg_run, rf.shape)

    sub_feat = np.mean(run_feats, axis=0)  # (37,121)
    print("SUB feat:", sub_feat.shape)
    return sub_feat

if __name__ == "__main__":
    sub010 = r"D:\download\edge_download\ds116_sub010\sub010"
    feat = extract_subject_feature(sub010, fs=1000.0)

    # 从路径里自动提取 sub010
    sub_id = os.path.basename(sub010)

    out_dir = r"D:\eeg_results"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{sub_id}_eeg.npy")
    np.save(out_path, feat)

    print("Saved:", out_path)
