import nibabel as nib

#path = r"D:\data\sub-01\...\task001_run001.feat\filtered_func_data.nii.gz"  # 改成你的实际路径
path = r"D:\download\edge_download\ds116_sub001\sub001\model\model001\task001_run001.feat\filtered_func_data.nii.gz"
img = nib.load(path)
print("shape:", img.shape)
print("zooms:", img.header.get_zooms())  # 最后一个通常是TR（秒）
