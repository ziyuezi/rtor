import numpy as np
import tensorly as tl
from tensorly.tenalg import mode_dot, multi_mode_dot
from scipy.sparse.linalg import cg, LinearOperator
from scipy.linalg import svd, norm

# 设置 Tensorly 后端为 numpy
tl.set_backend('numpy')


class ROTR:
    def __init__(self, ranks, lambdas, params):
        self.ranks = ranks
        # 正则化参数
        self.lambda_x = lambdas.get('lambda_x', 0.1)
        self.lambda_y = lambdas.get('lambda_y', 0.1)
        self.lambda_b = lambdas.get('lambda_b', 0.1)

        # 优化参数
        self.max_iter = params.get('max_iter', 100)
        self.tol = params.get('tol', 1e-4)
        self.fista_iter = params.get('fista_iter', 10)
        self.sx_max_step = params.get('sx_max_step', 0.01)
        self.manifold_lr = params.get('manifold_lr', 1e-4)
        self.init_scale = params.get('init_scale', 0.1)

        # 内部状态
        self.G = None
        self.Uk = []
        self.Sx = None
        self.Sy = None
        self.B_full = None
        self.L = 0

    def _soft_thresholding(self, x, threshold):
        # 软阈值算子
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _contract_product(self, X, B, L_dims):
        # 张量收缩 <X, B>_L
        x_ndim = X.ndim
        axes_x = list(range(1, x_ndim))
        axes_b = list(range(0, len(axes_x)))
        return np.tensordot(X, B, axes=(axes_x, axes_b))

    def _reconstruct_B(self):
        # 重构系数张量
        return multi_mode_dot(self.G, self.Uk)

    def fit(self, X, Y):
        N = X.shape[0]
        self.L = X.ndim - 1
        L = self.L
        M = Y.ndim - 1

        input_dims = X.shape[1:]
        output_dims = Y.shape[1:]
        total_modes = L + M
        current_ranks = self.ranks

        # --- 1. 初始化 ---
        self.Uk = []
        for i in range(L):
            U, _, _ = svd(np.random.randn(input_dims[i], current_ranks[i]), full_matrices=False)
            self.Uk.append(U)
        for i in range(M):
            U, _, _ = svd(np.random.randn(output_dims[i], current_ranks[L + i]), full_matrices=False)
            self.Uk.append(U)

        # 【修改点1】增大初始化规模，不要让它接近0
        # 让初始预测具备一定的量级
        self.G = np.random.randn(*current_ranks) * 1.0

        self.Sx = np.zeros_like(X)
        self.Sy = np.zeros_like(Y)

        loss_history = []
        print(f"Starting ROTR Optimization (Max Iter: {self.max_iter})...")

        for iteration in range(self.max_iter):
            # 【修改点2】预热策略 (Warm-up Strategy)
            # 前 15 轮，强制 Lambda 无穷大，禁止 Sy 和 Sx 更新
            # 迫使 B 去拟合主要数据（包括异常）
            if iteration < 15:
                curr_lambda_y = 1e10
                curr_lambda_x = 1e10
            else:
                curr_lambda_y = self.lambda_y
                curr_lambda_x = self.lambda_x

            B = self._reconstruct_B()

            # ==========================================
            # Step 1: 更新响应异常 Sy
            # ==========================================
            X_clean = X - self.Sx
            Y_pred = self._contract_product(X_clean, B, L)
            Residual_Sy = Y - Y_pred

            # 使用动态 lambda
            self.Sy = self._soft_thresholding(Residual_Sy, curr_lambda_y)

            # ==========================================
            # Step 2: 更新预测变量异常 Sx
            # ==========================================
            Y_tilde = Y - self.Sy
            lipschitz_const = norm(B.ravel()) ** 2 + 1e-8
            step_size_Sx = 1.0 / lipschitz_const
            step_size_Sx = min(step_size_Sx, self.sx_max_step)

            for k_ista in range(self.fista_iter):
                X_curr = X - self.Sx
                pred = self._contract_product(X_curr, B, L)
                Residual_Sx = Y_tilde - pred
                axes_res = list(range(1, M + 1))
                axes_B_out = list(range(L, L + M))
                Gradient_Step = np.tensordot(Residual_Sx, B, axes=(axes_res, axes_B_out))
                Sx_temp = self.Sx + step_size_Sx * Gradient_Step
                # 使用动态 lambda
                self.Sx = self._soft_thresholding(Sx_temp, curr_lambda_x * step_size_Sx)
        # ==========================================
            # Step 3: 更新核心张量 G (CG + Implicit Operator)
            # ==========================================
            X_clean = X - self.Sx
            Y_clean = Y - self.Sy

            X_tilde = X_clean.copy()
            for i in range(L):
                X_tilde = mode_dot(X_tilde, self.Uk[i].T, mode=i + 1)

            Y_tilde_core = Y_clean.copy()
            for i in range(M):
                Y_tilde_core = mode_dot(Y_tilde_core, self.Uk[L + i].T, mode=i + 1)

            RHS_G = np.tensordot(X_tilde, Y_tilde_core, axes=([0], [0]))
            b_vec = RHS_G.ravel()
            dim_G = np.prod(self.ranks)

            def matvec_G(v):
                G_curr = v.reshape(self.ranks)
                axes_X = list(range(1, L + 1))
                axes_G = list(range(L))
                Y_pred_core = np.tensordot(X_tilde, G_curr, axes=(axes_X, axes_G))
                delta_G = np.tensordot(X_tilde, Y_pred_core, axes=([0], [0]))
                return (delta_G + self.lambda_b * G_curr).ravel()

            A_op = LinearOperator((dim_G, dim_G), matvec=matvec_G)
            G_opt_vec, info = cg(A_op, b_vec, x0=self.G.ravel(), tol=1e-5)
            self.G = G_opt_vec.reshape(self.ranks)

            # ==========================================
            # Step 4: 更新因子矩阵 Uk (流形梯度下降)
            # ==========================================
            B_curr = self._reconstruct_B()
            Pred_Global = self._contract_product(X_clean, B_curr, L)
            Diff = Pred_Global - Y_clean
            Grad_B = np.tensordot(X_clean, Diff, axes=([0], [0]))

            for k in range(total_modes):
                current_Uk = self.Uk[k]
                B_neg_k = self.G.copy()
                for i in range(total_modes):
                    if i == k: continue
                    B_neg_k = mode_dot(B_neg_k, self.Uk[i], mode=i)
                axes_contract = [i for i in range(total_modes) if i != k]
                Grad_Euc = np.tensordot(Grad_B, B_neg_k, axes=(axes_contract, axes_contract))

                Ut_G = np.dot(current_Uk.T, Grad_Euc)
                sym_Ut_G = 0.5 * (Ut_G + Ut_G.T)
                Grad_Riem = Grad_Euc - np.dot(current_Uk, sym_Ut_G)

                # 【修改点3】调大流形学习率
                # 初始阶段需要更大的步长来打破僵局
                Uk_new_temp = current_Uk - self.manifold_lr * Grad_Riem
                U_svd, _, Vh_svd = svd(Uk_new_temp, full_matrices=False)
                self.Uk[k] = np.dot(U_svd, Vh_svd)

            # ==========================================
            # 监控与收敛检查
            # ==========================================
            loss = 0.5 * norm(Y - self.Sy - self._contract_product(X - self.Sx, self._reconstruct_B(), L)) ** 2
            loss += self.lambda_x * np.sum(np.abs(self.Sx))
            loss += self.lambda_y * np.sum(np.abs(self.Sy))
            loss += 0.5 * self.lambda_b * (norm(self.G) ** 2)

            loss_history.append(loss)

            # NaN 检查
            if np.isnan(loss):
                print("Error: Loss is NaN. Stopping training.")
                break

            if iteration % 10 == 0:
                print(f"Iter {iteration}: Loss = {loss:.4f} | |Sy|={norm(self.Sy):.2f} | |Sx|={norm(self.Sx):.2f}")

            if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

        self.B_full = self._reconstruct_B()
        return self

    def predict(self, X):
        return self._contract_product(X, self.B_full, self.L)


# -----------------------------------------------------------
# 主程序：仿真与评估
# -----------------------------------------------------------
if __name__ == "__main__":

    # ==========================================
    # 1. 全局配置
    # ==========================================
    CONFIG = {
        'N_train': 400,
        'N_test': 100,
        'P': [15, 20],  # 输入维度
        'O': [5, 10],  # 输出维度
        'true_ranks': [5, 5, 5, 5],
        'model_ranks': [5, 5, 5, 5],

        # 异常参数
        'outlier_density': 0.1,  # 异常样本比例
        'sy_magnitude': 10.0,  # Sy 幅度
        'sx_magnitude': 20.0,  # Sx 幅度
        'random_seed': 42,

        # 模型参数
        'lambdas': {
            'lambda_x': 0.1,
            'lambda_y': 5.0,
            'lambda_b': 0.1
        },
        'opt_params': {
            'max_iter': 100,
            'tol': 1e-4,
            'fista_iter': 10,
            'sx_max_step': 0.05,
            'manifold_lr': 1e-2,
            'init_scale': 1.0
        }
    }

    print("=== Simulation Setup ===")
    for k, v in CONFIG.items():
        if isinstance(v, dict): continue
        print(f"{k}: {v}")

    np.random.seed(CONFIG['random_seed'])

    # ==========================================
    # 2. 数据生成 (Data Simulation)
    # ==========================================
    N_total = CONFIG['N_train'] + CONFIG['N_test']
    P = CONFIG['P']
    O = CONFIG['O']
    ranks = CONFIG['true_ranks']
    L = len(P)

    # 2.1 生成纯净输入
    X_total = np.random.randn(N_total, *P)

    # 2.2 生成真实的 B (使用 Tucker 结构)
    G_true = np.random.randn(*ranks)
    U_in = [np.linalg.qr(np.random.randn(p, r))[0] for p, r in zip(P, ranks[:L])]
    U_out = [np.linalg.qr(np.random.randn(o, r))[0] for o, r in zip(O, ranks[L:])]

    B_full = mode_dot(G_true, U_in[0], 0)
    for i in range(1, L):
        B_full = mode_dot(B_full, U_in[i], i)
    for i in range(len(U_out)):
        B_full = mode_dot(B_full, U_out[i], L + i)

    # 2.3 生成 Y = <X, B> + E
    Y_clean = np.tensordot(X_total, B_full, axes=(list(range(1, L + 1)), list(range(L))))
    Noise = np.random.randn(*Y_clean.shape)
    Y_total = Y_clean + Noise

    # ==========================================
    # 3. 数据集划分 & 异常注入
    # ==========================================
    X_train = X_total[:CONFIG['N_train']].copy()
    Y_train = Y_total[:CONFIG['N_train']].copy()
    X_test = X_total[CONFIG['N_train']:].copy()
    Y_test = Y_total[CONFIG['N_train']:].copy()

    # 注入异常 (仅训练集)
    n_outliers = int(CONFIG['N_train'] * CONFIG['outlier_density'])
    outlier_indices = np.random.choice(CONFIG['N_train'], n_outliers, replace=False)

    S_y_true_train = np.zeros_like(Y_train)
    S_x_true_train = np.zeros_like(X_train)

    print(f"\nInjecting outliers into {n_outliers} samples...")

    # 对选中的样本进行 "稀疏块" 污染，而不是全量污染，更符合实际
    for idx in outlier_indices:
        # Sy 污染：随机选择一段连续区间 (模拟 RTOT 论文设置)
        # 将张量展平后选择一段
        y_flat_len = np.prod(O)
        start_idx = np.random.randint(0, y_flat_len - 5)
        # 生成掩码
        mask = np.zeros(y_flat_len)
        mask[start_idx: start_idx + 10] = 1  # 污染长度为10
        # 赋值
        S_y_true_train[idx] = mask.reshape(O) * CONFIG['sy_magnitude'] * np.random.choice([-1, 1])

        # Sx 污染：随机选择一个块 (Block)
        # 假设 P=[15, 20]，我们污染一个 3x3 的块
        p1_start = np.random.randint(0, P[0] - 3)
        p2_start = np.random.randint(0, P[1] - 3)
        S_x_true_train[idx, p1_start:p1_start + 3, p2_start:p2_start + 3] = CONFIG['sx_magnitude']

    # 叠加异常
    Y_train += S_y_true_train
    X_train += S_x_true_train

    # ==========================================
    # 4. 模型训练
    # ==========================================
    model = ROTR(
        ranks=CONFIG['model_ranks'],
        lambdas=CONFIG['lambdas'],
        params=CONFIG['opt_params']
    )

    model.fit(X_train, Y_train)

    # ==========================================
    # 5. 结果评估
    # ==========================================
    print("\n=== Evaluation Report ===")

    # --- 指标 A: RPE (在干净测试集上) ---
    Y_pred_test = model.predict(X_test)
    rpe = np.linalg.norm(Y_test - Y_pred_test) / np.linalg.norm(Y_test)

    print(f"【Prediction】")
    print(f"RPE (Test Set): {rpe:.4f}")

    # --- 指标 B: 异常检测 (在含污染的训练集上) ---
    print(f"\n【Outlier Detection (Train Set)】")

    # 定义检测阈值 (忽略微小的计算残差)
    tol_detect = 1e-4

    # Sy 评估
    true_mask_y = np.abs(S_y_true_train) > tol_detect
    pred_mask_y = np.abs(model.Sy) > tol_detect

    tp_y = np.sum(true_mask_y & pred_mask_y)
    fn_y = np.sum(true_mask_y & (~pred_mask_y))
    fp_y = np.sum((~true_mask_y) & pred_mask_y)
    tn_y = np.sum((~true_mask_y) & (~pred_mask_y))

    recall_y = tp_y / (tp_y + fn_y + 1e-8)
    precision_y = tp_y / (tp_y + fp_y + 1e-8)
    f1_y = 2 * precision_y * recall_y / (precision_y + recall_y + 1e-8)

    print(f"Sy Metrics:")
    print(f"  Recall (TPR): {recall_y:.4f} ({tp_y}/{tp_y + fn_y})")
    print(f"  Precision:    {precision_y:.4f}")
    print(f"  F1 Score:     {f1_y:.4f}")

    # Sx 评估
    true_mask_x = np.abs(S_x_true_train) > tol_detect
    pred_mask_x = np.abs(model.Sx) > tol_detect

    tp_x = np.sum(true_mask_x & pred_mask_x)
    fn_x = np.sum(true_mask_x & (~pred_mask_x))
    fp_x = np.sum((~true_mask_x) & pred_mask_x)

    recall_x = tp_x / (tp_x + fn_x + 1e-8)
    precision_x = tp_x / (tp_x + fp_x + 1e-8)
    f1_x = 2 * precision_x * recall_x / (precision_x + recall_x + 1e-8)

    print(f"Sx Metrics:")
    print(f"  Recall (TPR): {recall_x:.4f} ({tp_x}/{tp_x + fn_x})")
    print(f"  Precision:    {precision_x:.4f}")
    print(f"  F1 Score:     {f1_x:.4f}")

    # --- 指标 C: 范数对比 ---
    print(f"\n【Norm Comparison】")
    print(f"Sy Norm | True: {norm(S_y_true_train):.2f} | Est: {norm(model.Sy):.2f}")
    print(f"Sx Norm | True: {norm(S_x_true_train):.2f} | Est: {norm(model.Sx):.2f}")

    print("\nDone.")