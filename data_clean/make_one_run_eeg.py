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
    # 自动找 37×T 的数据
    X = None
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == 37:
            X = v.astype(float)
            data_key = k
            break
    if X is None:
        raise RuntimeError("No 37×T EEG array found in mat.")

    n0 = int(round(tmin * fs))
    n1 = int(round(tmax * fs))
    W = n1 - n0

    feats = []
    dropped_total = 0
    kept_total = 0

    for cf in ["cond001.txt", "cond002.txt", "cond003.txt"]:
        onsets = load_onsets(onset_dir + "\\" + cf)
        epochs = []
        dropped = 0

        for s in onsets:
            center = int(round(s * fs))
            a = center + n0
            b = center + n1
            if a >= 0 and b <= X.shape[1]:
                epochs.append(X[:, a:b])
            else:
                dropped += 1

        if not epochs:
            raise RuntimeError(f"{cf}: no valid epochs. fs or paths wrong?")

        E = np.stack(epochs, axis=2)      # (37, W, n_trials)
        erp = E.mean(axis=2)              # (37, W)

        # 重采样到121
        t_old = np.linspace(0.0, 1.0, W)
        t_new = np.linspace(0.0, 1.0, n_time)
        erp121 = np.vstack([np.interp(t_new, t_old, erp[ch]) for ch in range(37)])

        feats.append(erp121)
        dropped_total += dropped
        kept_total += E.shape[2]

        print(f"{cf}: events={len(onsets)} kept={E.shape[2]} dropped={dropped}")

    run_feat = np.mean(feats, axis=0)  # (37,121)

    print("EEG key:", data_key, "shape:", X.shape)
    print("run_feat:", run_feat.shape, "kept_total:", kept_total, "dropped_total:", dropped_total)
    return run_feat

if __name__ == "__main__":
    eeg_mat = r"D:\download\edge_download\ds116_sub010\sub010\EEG\task001_run001\EEG_rereferenced.mat"
    onset_dir = r"D:\download\edge_download\ds116_sub010\sub010\model\model001\onsets\task001_run001"
    feat = extract_run_feature(eeg_mat, onset_dir, fs=1000.0)
