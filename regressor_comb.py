import math
import torch
import numpy as np
import tensorly as tl
from scipy.linalg import svd
from scipy.sparse.linalg import LinearOperator, cg
from numpy.linalg import norm
from tensorly.tenalg.proximal import soft_thresholding, svd_thresholding
from tensorly.decomposition import robust_pca
from tensorly import tenalg


# 辅助函数
def ttt(x, b, L, dims):
    num_input_dims = len(dims[:L])
    return np.tensordot(
        x, b,
        axes=[
            [k + 1 for k in range(num_input_dims)],
            [k for k in range(num_input_dims)]
        ]
    )


def aic(y_true, y_pre, s_pre, r, b):
    rss = np.linalg.norm(y_true - y_pre - s_pre) ** 2
    k = np.count_nonzero(s_pre)
    return np.log(rss + 1e-10) + 2 * np.log(k + 1e-5) + 0.5 * r


# ==========================================
# 2. TOT (Tensor-on-Tensor Regression)
# ==========================================
class TOT:
    def __init__(self, **params):
        self.dims = params['dims']
        self.L = params['L']
        self.R = params['R']
        self.max_itr = params.get('max_itr', 20)

        # --- 关键修复：强制转换为 numpy array 且为 float32 ---
        self.x = np.array([t.tolist() if hasattr(t, 'tolist') else t for t in params['x']], dtype=np.float32)
        self.y = np.array([t.tolist() if hasattr(t, 'tolist') else t for t in params['y']], dtype=np.float32)

        # 预处理 Y
        self.y_vec = tl.tensor_to_vec(self.y)
        # 预先 unfold y，减少循环内的计算
        self.yms = [tl.unfold(self.y, i) for i in range(len(self.y.shape))]

        # 转到 GPU/CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_vec_cuda = torch.tensor(self.y_vec, device=device).float()
        self.yms_cuda = [torch.tensor(ym, device=device).float() for ym in self.yms]

        self.name = 'tot'

    def fit(self, verbose=False):
        results = []
        # 初始化因子 (避免 0 值)
        U = [np.random.rand(p, self.R).astype(np.float32) + 0.01 for p in self.dims]
        itr = 0
        device = self.y_vec_cuda.device

        while itr < self.max_itr:
            # 1. Update Input Factors
            for i in range(len(self.dims[:self.L])):
                C = []
                for r in range(self.R):
                    cr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != i]
                    cr = tl.cp_to_tensor((None, cr))

                    axes_input = [k + 1 for k in range(len(self.dims[:self.L])) if k != i]
                    idx_cr = [k for k in range(len(cr.shape)) if k < len(axes_input)]

                    # 此时 self.x 已经是 numpy array，tensordot 安全
                    cr = np.tensordot(self.x, cr, axes=[axes_input, idx_cr])
                    cr = tl.unfold(cr, mode=1)
                    C.append(cr)

                C = np.concatenate(C)
                C_torch = torch.tensor(C, device=device).float()

                # 求解 u = pinv(C^T) * y (注意维度转置)
                # TOT 逻辑: vec(Y) approx C * vec(U) -> vec(U) = pinv(C) * vec(Y)
                # 这里 C 的形状取决于 unfold，如果是 (N_samples..., Dim_U)，则用 pinv(C)
                # 根据原代码逻辑保持一致：
                try:
                    u = (torch.linalg.pinv(C_torch.T) @ self.y_vec_cuda).reshape(-1, self.R)
                    U[i] = u.cpu().numpy()
                except RuntimeError:
                    # 捕获 SVD 不收敛，通常是因为数值爆炸
                    if verbose: print("Warning: SVD did not converge in TOT")
                    return 1e5, None, 0, None

            # 2. Update Output Factors
            for i in range(len(self.dims[self.L:])):
                D = []
                for r in range(self.R):
                    dr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != self.L + i]
                    dr = tl.cp_to_tensor((None, dr))
                    dr = ttt(self.x, dr, self.L, self.dims)
                    D.append(tl.tensor_to_vec(dr).reshape(-1, 1))

                D = np.concatenate(D, axis=1)
                D_torch = torch.tensor(D, device=device).float()
                target = self.yms_cuda[i + 1].T

                try:
                    U[self.L + i] = (torch.linalg.pinv(D_torch) @ target).T.cpu().numpy()
                except RuntimeError:
                    if verbose: print("Warning: SVD did not converge in TOT")
                    return 1e5, None, 0, None

            B = tl.cp_to_tensor((None, U))
            y_pre = ttt(self.x, B, self.L, self.dims)
            rpe = np.linalg.norm(self.y - y_pre) / (np.linalg.norm(self.y) + 1e-8)

            if len(results) > 0 and abs(rpe - results[-1][0]) < 1e-4: break
            results.append((rpe, B, 0, 0))
            itr += 1

        results = sorted(results, key=lambda x: x[0])
        return results[0][0], results[0][1], results[0][2], results[0][3]


# ==========================================
# 3. RTOT (Robust Tensor-on-Tensor)
# ==========================================
class RTOT:
    def __init__(self, **params):
        self.dims = params['dims']
        self.L = params['L']
        self.R = params['R']
        self.max_itr = params.get('max_itr', 20)

        # --- 关键修复 ---
        self.x = np.array([t.tolist() if hasattr(t, 'tolist') else t for t in params['x']], dtype=np.float32)
        self.y = np.array([t.tolist() if hasattr(t, 'tolist') else t for t in params['y']], dtype=np.float32)

        self.mu1 = params.get('mu1', 1e-3)
        self.mu2 = params.get('mu2', 1e-3)
        self.mu3 = params.get('mu3', 1e-3)

        self.y_vec = tl.tensor_to_vec(self.y)
        self.yms = [tl.unfold(self.y, i) for i in range(len(self.y.shape))]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_vec_cuda = torch.tensor(self.y_vec, device=device).float()
        self.yms_cuda = [torch.tensor(ym, device=device).float() for ym in self.yms]

        self.name = 'rtot'

    def fit(self, verbose=False):
        U = [np.random.rand(p, self.R).astype(np.float32) for p in self.dims]
        J = [np.random.rand(p, self.R).astype(np.float32) for p in self.dims]
        Z = [np.zeros_like(u) for u in U]
        S = np.zeros_like(self.y)

        results = []
        Bs = [tl.cp_to_tensor((None, U))]
        Js = [tl.cp_to_tensor((None, J))]
        itr = 0
        device = self.y_vec_cuda.device

        while itr < self.max_itr:
            # 1. Update J
            for i in range(len(J)):
                J[i] = svd_thresholding(U[i] + (1 / self.mu3) * Z[i], 1 / self.mu3)

            # 2. Update U (Input)
            for i in range(len(self.dims[:self.L])):
                C = []
                for r in range(self.R):
                    cr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != i]
                    cr = tl.cp_to_tensor((None, cr))
                    axes_input = [k + 1 for k in range(len(self.dims[:self.L])) if k != i]
                    idx_cr = [k for k in range(len(cr.shape)) if k < len(axes_input)]
                    cr = np.tensordot(self.x, cr, axes=[axes_input, idx_cr])
                    cr = tl.unfold(cr, mode=1)
                    C.append(cr)

                C = np.concatenate(C)
                C_torch = torch.tensor(C, device=device).float()
                CCT = C_torch @ C_torch.T

                s_vec = torch.tensor(S, device=device).float().flatten()
                j_vec = torch.tensor(J[i], device=device).float().flatten()
                z_vec = torch.tensor(Z[i], device=device).float().flatten()

                reg = self.mu3 * torch.eye(CCT.shape[0], device=device)
                term_inv = torch.linalg.pinv(self.mu1 * CCT + reg)
                term_rhs = self.mu1 * C_torch @ (self.y_vec_cuda - s_vec) + self.mu3 * j_vec - z_vec
                U[i] = (term_inv @ term_rhs).reshape(-1, self.R).cpu().numpy()

            # 3. Update U (Output)
            for i in range(len(self.dims[self.L:])):
                D = []
                for r in range(self.R):
                    dr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != self.L + i]
                    dr = tl.cp_to_tensor((None, dr))
                    dr = ttt(self.x, dr, self.L, self.dims)
                    D.append(tl.tensor_to_vec(dr).reshape(-1, 1))

                D = np.concatenate(D, axis=1)
                D_torch = torch.tensor(D, device=device).float()
                DTD = D_torch.T @ D_torch

                ym = self.yms_cuda[i + 1].T
                sm = torch.tensor(tl.unfold(S, mode=i + 1).T, device=device).float()
                J_cuda = torch.tensor(J[self.L + i], device=device).float()
                Z_cuda = torch.tensor(Z[self.L + i], device=device).float()

                reg = self.mu3 * torch.eye(DTD.shape[0], device=device)
                term_inv = torch.linalg.pinv(self.mu1 * DTD + reg)
                term_rhs = self.mu1 * D_torch.T @ (ym - sm) + self.mu3 * J_cuda.T - Z_cuda.T
                U[self.L + i] = (term_inv @ term_rhs).T.cpu().numpy()

            # 4. Update Z
            for i in range(len(Z)):
                Z[i] += self.mu3 * (U[i] - J[i])

            # 5. Update S
            B = tl.cp_to_tensor((None, U))
            J_ = tl.cp_to_tensor((None, J))
            y_pre = ttt(self.x, B, self.L, self.dims)
            S = soft_thresholding(self.y - y_pre, self.mu2 / self.mu1)

            rpe = np.linalg.norm(self.y - y_pre) / (np.linalg.norm(self.y) + 1e-8)
            norm_r = np.linalg.norm(Bs[-1] - B) ** 2
            AIC = aic(self.y, y_pre, S, self.R, B)

            if len(results) > 1 and abs(rpe - results[-1][0]) < 1e-4: break

            results.append((rpe, B, norm_r, AIC, S))
            Bs.append(B)
            Js.append(J_)
            itr += 1

        results = sorted(results, key=lambda x: x[0])
        return results[0][0], results[0][1], results[0][2], results[0][3]


# ==========================================
# 4. RPCA (Wrapper)
# ==========================================
class RPCA:
    def __init__(self, **params):
        self.params = params
        self.reg = params.get('reg', 1.5e-1)
        self.name = 'rpca'

    def fit(self, verbose=False):
        # 1. 先对 Y 做 Robust PCA
        # 强制转换为 numpy array 避免 tensorly 兼容性问题
        y_np = np.array([t.tolist() if hasattr(t, 'tolist') else t for t in self.params['y']], dtype=np.float32)

        # 将 tensor 展平成矩阵 (Samples, Features) 进行 RPCA
        y_shape = y_np.shape
        y_mat = y_np.reshape(y_shape[0], -1)

        try:
            L_mat, S_mat = robust_pca(y_mat)
        except:
            # Fallback for older versions
            L_mat, S_mat = y_mat, np.zeros_like(y_mat)

        L = L_mat.reshape(y_shape)
        S = S_mat.reshape(y_shape)

        # 2. 使用去噪后的 Y (L) 训练 TOT
        tot_params = self.params.copy()
        tot_params['y'] = L

        tot = TOT(**tot_params)
        rpe, B, aic_val, _ = tot.fit(verbose=verbose)

        return rpe, B, aic_val, S


# ==========================================
# 5. ROTR (Robust Orthogonal Tucker)
# ==========================================
class ROTR:
    def __init__(self, ranks, lambdas, params):
        self.ranks = ranks
        self.lambda_x = lambdas.get('lambda_x', 0.1)
        self.lambda_y = lambdas.get('lambda_y', 0.1)
        self.lambda_b = lambdas.get('lambda_b', 0.1)
        self.max_iter = params.get('max_iter', 100)
        self.tol = params.get('tol', 1e-4)
        self.fista_iter = params.get('fista_iter', 10)
        self.sx_max_step = params.get('sx_max_step', 0.01)
        self.manifold_lr = params.get('manifold_lr', 1e-4)
        self.init_scale = params.get('init_scale', 0.1)
        self.G = None
        self.Uk = []
        self.Sx = None
        self.Sy = None
        self.B_full = None
        self.name = 'rotr'

    def _contract_product(self, X, B, L):
        x_ndim = X.ndim
        axes_x = list(range(1, x_ndim))
        axes_b = list(range(0, len(axes_x)))
        return np.tensordot(X, B, axes=(axes_x, axes_b))

    def _reconstruct_B(self):
        return tl.tucker_to_tensor((self.G, self.Uk))

    def fit(self, X, Y, verbose=False):
        # 强制类型转换
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        N = X.shape[0]
        self.L = X.ndim - 1
        L = self.L
        M = Y.ndim - 1
        input_dims = X.shape[1:]
        output_dims = Y.shape[1:]
        total_modes = L + M
        current_ranks = self.ranks

        # 初始化
        self.Uk = []
        for i in range(L):
            U, _, _ = svd(np.random.randn(input_dims[i], current_ranks[i]), full_matrices=False)
            self.Uk.append(U)
        for i in range(M):
            U, _, _ = svd(np.random.randn(output_dims[i], current_ranks[L + i]), full_matrices=False)
            self.Uk.append(U)
        self.G = np.random.randn(*current_ranks) * self.init_scale
        self.Sx = np.zeros_like(X)
        self.Sy = np.zeros_like(Y)
        loss_history = []

        for iteration in range(self.max_iter):
            B = self._reconstruct_B()

            # 1. Update Sy
            X_clean = X - self.Sx
            Y_pred = self._contract_product(X_clean, B, L)
            self.Sy = soft_thresholding(Y - Y_pred, self.lambda_y)

            # 2. Update Sx (skipped if lambda_x is 0)
            if self.lambda_x > 0:
                Y_tilde = Y - self.Sy
                lipschitz_const = np.linalg.norm(B.ravel()) ** 2 + 1e-8
                step_size_Sx = min(1.0 / lipschitz_const, self.sx_max_step)
                for _ in range(self.fista_iter):
                    X_curr = X - self.Sx
                    pred = self._contract_product(X_curr, B, L)
                    Residual_Sx = Y_tilde - pred
                    axes_res = list(range(1, M + 1))
                    axes_B_out = list(range(L, L + M))
                    Gradient_Step = np.tensordot(Residual_Sx, B, axes=(axes_res, axes_B_out))
                    self.Sx = soft_thresholding(self.Sx + step_size_Sx * Gradient_Step, self.lambda_x * step_size_Sx)

            # 3. Update Core G
            X_clean = X - self.Sx
            Y_clean = Y - self.Sy
            X_tilde = X_clean.copy()
            for i in range(L):
                X_tilde = tenalg.mode_dot(X_tilde, self.Uk[i].T, mode=i + 1)
            Y_tilde_core = Y_clean.copy()
            for i in range(M):
                Y_tilde_core = tenalg.mode_dot(Y_tilde_core, self.Uk[L + i].T, mode=i + 1)

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
            G_opt_vec, _ = cg(A_op, b_vec, x0=self.G.ravel(), tol=1e-5)
            self.G = G_opt_vec.reshape(self.ranks)

            # 4. Update Uk
            B_curr = self._reconstruct_B()
            Pred_Global = self._contract_product(X_clean, B_curr, L)
            Diff = Pred_Global - Y_clean
            Grad_B = np.tensordot(X_clean, Diff, axes=([0], [0]))

            for k in range(total_modes):
                current_Uk = self.Uk[k]
                temp_G = self.G.copy()
                for i in range(total_modes):
                    if i == k: continue
                    temp_G = tenalg.mode_dot(temp_G, self.Uk[i], mode=i)

                grad_b_unf = tl.unfold(Grad_B, mode=k)
                b_neg_unf = tl.unfold(temp_G, mode=k)
                Grad_Euc = np.dot(grad_b_unf, b_neg_unf.T)
                Ut_G = np.dot(current_Uk.T, Grad_Euc)
                sym_Ut_G = 0.5 * (Ut_G + Ut_G.T)
                Grad_Riem = Grad_Euc - np.dot(current_Uk, sym_Ut_G)

                # Update on Manifold
                Uk_new_temp = current_Uk - self.manifold_lr * Grad_Riem
                U_svd, _, Vh_svd = svd(Uk_new_temp, full_matrices=False)
                self.Uk[k] = np.dot(U_svd, Vh_svd)

            # Check Convergence
            loss = 0.5 * norm(Y - self.Sy - self._contract_product(X - self.Sx, self._reconstruct_B(), L)) ** 2
            loss_history.append(loss)
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                break

        self.B_full = self._reconstruct_B()
        y_pre = self._contract_product(X, self.B_full, L)
        rpe = norm(Y - y_pre) / (norm(Y) + 1e-8)
        return rpe, self.B_full, 0, self.Sy