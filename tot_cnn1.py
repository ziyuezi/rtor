import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tensorly as tl
import os
import time
import pickle
import data_io as io  # 确保你已经按照之前的指示创建了 data_io.py
from collections import defaultdict
from itertools import product
# tot_cnn with no pytorch-lighting

# ==========================================
# 1. Dataset 定义 (保持不变)
# ==========================================
class CNNDataset(Dataset):
    def __init__(self, **params):
        self.dims = params['dims']
        self.y = params['y']
        self.x = params['x']

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # 确保数据是 float32 类型
        x = self.x[idx].reshape(1, self.dims[0], self.dims[1])
        y = self.y[idx].reshape(1, self.dims[2], self.dims[3])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ==========================================
# 2. 模型定义 (改为继承 nn.Module)
# ==========================================
class TOTCNN(nn.Module):
    def __init__(self, **params):
        super(TOTCNN, self).__init__()
        self.dims = params['dims']
        self.name = 'totcnn'

        # 网络结构保持不变
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4)
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=4)
        self.conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4)

        # 根据维度动态选择最后一层
        if self.dims[-1] * self.dims[-2] == 203:
            self.conv6 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(29, 1))
        elif self.dims[-1] * self.dims[-2] == (37 * 121):
            self.conv6 = nn.Conv2d(in_channels=64, out_channels=1, padding=(14, 57), kernel_size=(2, 2))
        else:
            # 默认 fallback，针对 gen_sync_data_norm 的 dims (15, 20, 5, 10) 需要适配
            # 这里的 kernel_size 需要根据数据维度调整，防止输出尺寸不匹配
            # 为了通用性，这里使用 AdaptiveAvgPool 或者修改 kernel
            # 假设按照原代码逻辑走:
            self.conv6 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(11, 11))

        self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.selu(self.conv2(x))
        x = self.selu(self.conv3(x))
        x = self.selu(self.conv4(x))
        x = self.selu(self.conv5(x))
        x = self.conv6(x)
        return x


# ==========================================
# 3. 自定义损失函数逻辑 (提取自原代码)
# ==========================================
def custom_loss_function(y_pre, y):
    # 原代码逻辑: 减去预测均值，加上真实均值
    y_pre = y_pre - torch.mean(y_pre, dim=(2, 3), keepdim=True) + torch.mean(y, dim=(2, 3), keepdim=True)
    loss = torch.norm(y_pre - y) / torch.norm(y)
    return loss, y_pre


# ==========================================
# 4. 训练与主循环
# ==========================================
def train_model():
    # 参数设置
    replications = 1  # 演示用，设为 1
    max_epochs = 100
    train_batch_size = 400
    test_batch_size = 100
    learning_rate = 1e-3  # 默认 Adam 学习率

    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 循环不同的参数配置 (模拟原代码逻辑)
    # 为了演示方便，这里只跑一组参数
    for percentile, Ru in product([.1], [7]):
        print(f"Start Experiment: Percentile={percentile}, Ru={Ru}")

        folder = r'./experiment-results/sync-data-normal-{}-{}/'.format(percentile, Ru)
        log_dir = f'./cnn-logs/log-sync-{percentile}-{Ru}'
        model_save_dir = f'./cnn-models/sync-{percentile}-{Ru}'

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # -------------------------------------------------
        # 数据生成/加载
        # 注意：这里直接调用 io.gen_sync_data_norm 生成数据
        # 如果你有 pickle 文件，可以取消注释原本的加载逻辑
        # -------------------------------------------------
        # 模拟加载参数列表
        params_template = dict(
            R=15, Ru=Ru, mu1=6.5e-3, mu2=3.5e-3, mu3=1e-8,
            tol=1e-4, max_itr=20, replications=replications,
            percentile=percentile, scale=2,
            # 必须设置正确的 dims 以匹配模型定义
            dims=(15, 20, 5, 10)  # 对应 gen_sync_data_norm
        )

        # 为了演示，我们现场生成数据，而不是从 pickle 读取
        # 真实复现时，请确保这里的数据源和你之前的实验一致
        list_params = []
        for _ in range(replications):
            p = params_template.copy()
            p = io.gen_sync_data_norm(**p)  # 使用 data_io 生成数据
            list_params.append(p)

        dict_rpes = defaultdict(list)

        for r in range(replications):
            params = list_params[r]
            dims = params['dims']

            # 准备 Dataset 和 DataLoader
            data_train = CNNDataset(x=params['x'], y=params['y'], dims=dims)
            data_test = CNNDataset(x=params['x_test'], y=params['y_test'], dims=dims)

            train_loader = DataLoader(dataset=data_train, batch_size=train_batch_size, shuffle=True)
            test_loader = DataLoader(dataset=data_test, batch_size=test_batch_size, shuffle=False)

            # 初始化模型、优化器、Writer
            model = TOTCNN(**params).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            writer = SummaryWriter(log_dir=os.path.join(log_dir, f'rep-{r}'))

            best_val_loss = float('inf')
            best_model_path = ""

            start_time = time.time()

            # ==========================
            # Epoch 循环
            # ==========================
            for epoch in range(max_epochs):

                # --- Training ---
                model.train()
                train_losses = []
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    output = model(batch_x)

                    loss, _ = custom_loss_function(output, batch_y)

                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                avg_train_loss = np.mean(train_losses)
                writer.add_scalar('Loss/Train', avg_train_loss, epoch)

                # --- Validation ---
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        output = model(batch_x)
                        loss, _ = custom_loss_function(output, batch_y)
                        val_losses.append(loss.item())

                avg_val_loss = np.mean(val_losses)
                writer.add_scalar('Loss/Valid', avg_val_loss, epoch)

                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(model_save_dir, f'best_model_rep{r}.pth')
                    torch.save(model.state_dict(), best_model_path)

                # 简单的进度打印
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Rep {r}, Epoch {epoch + 1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            end_time = time.time()

            # ==========================
            # 测试 (Evaluation)
            # ==========================
            print(f"Training finished. Loading best model from {best_model_path}")
            # 加载最佳模型参数
            model.load_state_dict(torch.load(best_model_path))
            model.eval()

            # 在整个测试集上计算最终指标
            # 这里的逻辑模仿原代码的后处理
            x_test_tensor = torch.tensor(params['x_test'], dtype=torch.float32).reshape(-1, 1, dims[0], dims[1]).to(
                device)

            with torch.no_grad():
                y_pre_tensor = model(x_test_tensor)

            # 转回 Numpy
            y_pre = y_pre_tensor.cpu().numpy()
            y_test = params['y_test']  # 原始 numpy 数据

            # TensorLy 处理 (模仿原代码)
            y_test_vec = tl.partial_tensor_to_vec(y_test, skip_begin=1)
            y_pre_vec = tl.partial_tensor_to_vec(y_pre, skip_begin=1)

            m_test = np.mean(y_test_vec, axis=1).reshape(-1, 1)
            m_pre = np.mean(y_pre_vec, axis=1).reshape(-1, 1)

            # 最终校准
            y_pre_final = y_pre_vec - m_pre + m_test

            rpe = np.linalg.norm(y_pre_final - y_test_vec) / np.linalg.norm(y_test_vec)
            dict_rpes['totcnn'].append(rpe)

            print(f"Result: Ru={Ru}, Rep={r}, Model=TOTCNN, RPE={rpe:.4f}, Time={end_time - start_time:.2f}s")
            print("============")

            writer.close()

        # 保存结果 (可选)
        if not os.path.exists(folder): os.makedirs(folder)
        with open(os.path.join(folder, f'dict_rpes-p={percentile}.p.split.cnn'), 'wb') as f:
            pickle.dump(dict_rpes, f)

        print("Done.")


if __name__ == '__main__':
    train_model()