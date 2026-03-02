from scipy.io import loadmat
import numpy as np

p = r"D:\download\edge_download\ds116_sub010\sub010\EEG\task001_run001\EEG_rereferenced.mat"
#p = r"D:\download\edge_download\ds116_sub010\sub010\EEG\task001_run001\EEG_raw.mat"
print(p)
m = loadmat(p, squeeze_me=True, struct_as_record=False)

for k in sorted([k for k in m.keys() if not k.startswith("__")]):
    v = m[k]
    if isinstance(v, np.ndarray):
        print(k, "ndarray", v.shape, v.dtype)
    else:
        print(k, type(v), v)
