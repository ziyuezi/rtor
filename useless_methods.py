
class ROTR:
    def __init__(self ,**params):
        self.ranks = params['ranks']
        self.x = np.array(params['x'])
        self.y = np.array(params['y'])
        self.L = params['L']
        self.M = params['M']

        # 正则化参数
        self.lambda_x = params['lambda_x']
        self.lambda_y = params['lambda_y']
        self.lambda_b = params['lambda_b']

        # 优化参数
        self.max_iter = params['max_itr']
        self.tol = params['tol']
        self.fista_iter = params['fista_iter']
        self.sx_max_step = params['sx_max_step']
        self.manifold_lr = params['manifold_lr']
        self.init_scale = params['init_scale'] # 控制核张量的数值大小
        self.name = 'rotr'

        # 内部状态
        self.G = None
        self.Uk = []
        self.Sx = None
        self.Sy = None
        self.B_full = None



    def _contract_product(self, X, B, L_dims):
        # 张量收缩 <X, B>_L
        x_ndim = X.ndim
        axes_x = list(range(1, x_ndim))
        axes_b = list(range(0, len(axes_x)))
        return np.tensordot(X, B, axes=(axes_x, axes_b))

    def _reconstruct_B(self):
        # 重构系数张量
        return multi_mode_dot(self.G, self.Uk)

    def fit(self ,verbose=True):
        X = self.x
        Y = self.y
        N = X.shape[0]
        L = self.L
        M = self.M

        input_dims = X.shape[1:]
        output_dims = Y.shape[1:]
        total_modes = L + M
        current_ranks = self.ranks

        # --- 1. 初始化 ---
        self.Uk = []
        for i in range(L):
            U, _, _ = svd(np.random.randn(input_dims[i], current_ranks[i]), full_matrices=False)
            # 换成qr分解
            # U,_ = np.linalg.qr(np.random.randn(input_dims[i], current_ranks[i]))
            self.Uk.append(U)
        for i in range(M):
            U, _, _ = svd(np.random.randn(output_dims[i], current_ranks[L + i]), full_matrices=False)
            # U,_ = np.linalg.qr(np.random.randn(input_dims[i], current_ranks[L + i]))
            self.Uk.append(U)

        # 核心张量初始化 (缩放防止初始预测过大)
        self.G = np.random.randn(*current_ranks) * self.init_scale

        self.Sx = np.zeros_like(X)
        self.Sy = np.zeros_like(Y)

        loss_history = []
        print(f"Starting ROTR Optimization (Max Iter: {self.max_iter})...")

        for iteration in range(self.max_iter):
            B = self._reconstruct_B()

            # ==========================================
            # Step 1: 更新响应异常 Sy (Soft Thresholding)
            # ==========================================
            X_clean = X - self.Sx
            Y_pred = self._contract_product(X_clean, B, L)
            Residual_Sy = Y - Y_pred
            self.Sy = soft_thresholding(Residual_Sy, self.lambda_y)

            # ==========================================
            # Step 2: 更新预测变量异常 Sx (ISTA)
            # ==========================================
            Y_tilde = Y - self.Sy

            # 动态计算步长
            lipschitz_const = norm(B.ravel()) ** 2 + 1e-8
            step_size_Sx = 1.0 / lipschitz_const
            step_size_Sx = min(step_size_Sx, self.sx_max_step)  # 截断步长

            for k_ista in range(self.fista_iter):
                X_curr = X - self.Sx
                pred = self._contract_product(X_curr, B, L)
                Residual_Sx = Y_tilde - pred

                axes_res = list(range(1, M + 1))
                axes_B_out = list(range(L, L + M))
                Gradient_Step = np.tensordot(Residual_Sx, B, axes=(axes_res, axes_B_out))

                # 更新 Sx
                Sx_temp = self.Sx + step_size_Sx * Gradient_Step # 梯度更新方向 取负号，甚至都不影响结果？ 呃呃呃。肯定不影响啊，
                # 你就没有Sx的异常值
                self.Sx = soft_thresholding(Sx_temp, self.lambda_x * step_size_Sx)

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

                Uk_new_temp = current_Uk - self.manifold_lr * Grad_Riem # 黎曼梯度这里取负号呢？ 理论上是取负号，我改成正号了。。
                U_svd, _, Vh_svd = svd(Uk_new_temp, full_matrices=False)
                self.Uk[k] = np.dot(U_svd, Vh_svd)

            # ==========================================
            # 监控与收敛检查
            # ==========================================
            # 需要rpe,sparsity
            # 1. 更新全局系数 B (因为 G 和 Uk 刚刚更新过)
            self.B_full = self._reconstruct_B()
            # 2. 预测时必须使用去除异常后的输入: X - Sx
            X_clean_eval = X - self.Sx
            Y_hat = self._contract_product(X_clean_eval, self.B_full, L)
            # 3. 计算 RPE: 必须与原始观测 Y 比较
            norm_y = norm(Y)
            norm_diff = norm(Y - Y_hat)
            rpe = norm_diff / (norm_y + 1e-10)
            # 4.计算稀疏度
            sparsity_sx = (self.Sx.size - np.count_nonzero(self.Sx)) / self.Sx.size
            sparsity_sy = (self.Sy.size - np.count_nonzero(self.Sy)) / self.Sy.size

            loss_history.append(rpe)
            if verbose and iteration % 10 == 0:
                print \
                    (f"Iter {iteration}: RPE={rpe:.5f} | Sparsity(Sx)={sparsity_sx:.2%} | Sparsity(Sy)={sparsity_sy:.2%}")
            # 5. 收敛检查
            if loss_history[-1 ] >1.2:
                break
            if iteration >10 and (sparsity_sx > 0.91 or sparsity_sy > 0.91):
                break
            if iteration > 0:
                loss_change = loss_history[-1] - loss_history[-2]

                # 损失轻微上升：可能是过拟合或收敛振荡
                if 0 < loss_change < self.tol:
                    if verbose:
                        print(f"Stopping due to slight increase at iteration {iteration}")
                    break

                # 正常收敛：损失下降且变化很小
                if abs(loss_change) < self.tol:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break
        return rpe, self.B_full, sparsity_sx, sparsity_sy


class ROTR_weighted:
    def __init__(self, **params):
        self.ranks = params['ranks']
        self.x = np.array(params['x'])
        self.y = np.array(params['y'])
        self.L = params['L']
        self.M = params['M']

        # 正则化参数
        self.lambda_x = params['lambda_x']
        self.lambda_y = params['lambda_y']
        self.lambda_b = params['lambda_b']
        # 初始化 EIV 参数 gamma
        self.gamma = params['gamma']
        # 优化参数
        self.max_iter = params['max_itr']
        self.tol = params['tol']
        self.fista_iter = params['fista_iter']
        self.sx_max_step = params['sx_max_step']
        self.manifold_lr = params['manifold_lr']
        self.init_scale = params['init_scale']  # 控制核张量的数值大小
        self.name = 'rotr_weighted'

        # 内部状态
        self.G = None
        self.Uk = []
        self.Sx = None
        self.Sy = None
        self.B_full = None

    def _contract_product(self, X, B, L_dims):
        # 张量收缩 <X, B>_L
        x_ndim = X.ndim
        axes_x = list(range(1, x_ndim))
        axes_b = list(range(0, len(axes_x)))
        return np.tensordot(X, B, axes=(axes_x, axes_b))

    def _reconstruct_B(self):
        # 重构系数张量
        return multi_mode_dot(self.G, self.Uk)

    def _compute_weights(self, Sx):
        # 1. 计算每个样本 Sx 的 L2 范数 (即异常程度)
        N = Sx.shape[0]
        sx_flat = Sx.reshape(N, -1)
        sx_norms = np.linalg.norm(sx_flat, axis=1)

        # 2. 计算权重 (高斯核 / Welsch)
        # 归一化因子：防止 gamma 难以调整，用平均值做 scale
        scale = np.mean(sx_norms) + 1e-10
        weights = np.exp(-self.gamma * (sx_norms / scale) ** 2)

        return weights  # shape (N,)

    def fit(self, verbose=True):
        X = self.x
        Y = self.y
        N = X.shape[0]
        L = self.L
        M = self.M

        input_dims = X.shape[1:]
        output_dims = Y.shape[1:]
        total_modes = L + M
        current_ranks = self.ranks

        # --- 1. 初始化 ---
        self.Uk = []
        for i in range(L):
            U, _, _ = svd(np.random.randn(input_dims[i], current_ranks[i]), full_matrices=False)
            # 换成qr分解
            # U,_ = np.linalg.qr(np.random.randn(input_dims[i], current_ranks[i]))
            self.Uk.append(U)
        for i in range(M):
            U, _, _ = svd(np.random.randn(output_dims[i], current_ranks[L + i]), full_matrices=False)
            # U,_ = np.linalg.qr(np.random.randn(input_dims[i], current_ranks[L + i]))
            self.Uk.append(U)

        # 核心张量初始化 (缩放防止初始预测过大)
        self.G = np.random.randn(*current_ranks) * self.init_scale

        self.Sx = np.zeros_like(X)
        self.Sy = np.zeros_like(Y)

        loss_history = []
        print(f"Starting ROTR Optimization (Max Iter: {self.max_iter})...")

        for iteration in range(self.max_iter):
            B = self._reconstruct_B()

            # ==========================================
            # Step 1: 更新响应异常 Sy (Soft Thresholding)
            # ==========================================
            X_clean = X - self.Sx
            Y_pred = self._contract_product(X_clean, B, L)
            Residual_Sy = Y - Y_pred
            self.Sy = soft_thresholding(Residual_Sy, self.lambda_y)

            # ==========================================
            # Step 2: 更新预测变量异常 Sx (ISTA)
            # ==========================================
            Y_tilde = Y - self.Sy

            # 动态计算步长
            lipschitz_const = norm(B.ravel()) ** 2 + 1e-8
            step_size_Sx = 1.0 / lipschitz_const
            step_size_Sx = min(step_size_Sx, self.sx_max_step)  # 截断步长

            for k_ista in range(self.fista_iter):
                X_curr = X - self.Sx
                pred = self._contract_product(X_curr, B, L)
                Residual_Sx = Y_tilde - pred

                axes_res = list(range(1, M + 1))
                axes_B_out = list(range(L, L + M))
                Gradient_Step = np.tensordot(Residual_Sx, B, axes=(axes_res, axes_B_out))

                # 更新 Sx
                Sx_temp = self.Sx + step_size_Sx * Gradient_Step  # 梯度更新方向 取负号，甚至都不影响结果？ 呃呃呃。肯定不影响啊，
                # 你就没有Sx的异常值
                self.Sx = soft_thresholding(Sx_temp, self.lambda_x * step_size_Sx)
            # ==========================================
            # Step 2.5: 计算样本权重 (EIV 核心)
            # ==========================================
            # 根据刚算出的 Sx，判断哪些样本是“脏”的
            weights = self._compute_weights(self.Sx)

            # 准备加权数据 (Hacker's Trick: Multiply by sqrt(w))
            # 这样后续的标准最小二乘求解器会自动变为加权最小二乘
            w_sqrt = np.sqrt(weights)

            # 扩展维度以便广播乘法
            # X shape: (N, d1, d2...), w shape: (N, 1, 1...)
            w_shape_x = [-1] + [1] * (X.ndim - 1)
            w_shape_y = [-1] + [1] * (Y.ndim - 1)

            w_sqrt_x = w_sqrt.reshape(w_shape_x)
            w_sqrt_y = w_sqrt.reshape(w_shape_y)

            # 计算用于更新 G 的加权纯净数据
            X_clean = X - self.Sx
            Y_clean = Y - self.Sy

            X_weighted = X_clean * w_sqrt_x
            Y_weighted = Y_clean * w_sqrt_y
            # ==========================================
            # Step 3: 更新核心张量 G (CG + Implicit Operator)
            # ==========================================

            X_tilde = X_weighted.copy()
            for i in range(L):
                X_tilde = mode_dot(X_tilde, self.Uk[i].T, mode=i + 1)

            Y_tilde_core = Y_weighted.copy()
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
            Diff_Weighted = Diff * weights.reshape(w_shape_y)
            Grad_B = np.tensordot(X_clean, Diff_Weighted, axes=([0], [0]))

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

                Uk_new_temp = current_Uk - self.manifold_lr * Grad_Riem  # 黎曼梯度这里取负号呢？ 理论上是取负号，我改成正号了。。
                U_svd, _, Vh_svd = svd(Uk_new_temp, full_matrices=False)
                self.Uk[k] = np.dot(U_svd, Vh_svd)

            # ==========================================
            # 监控与收敛检查
            # ==========================================
            # 需要rpe,sparsity
            # 1. 更新全局系数 B (因为 G 和 Uk 刚刚更新过)
            self.B_full = self._reconstruct_B()
            # 2. 预测时必须使用去除异常后的输入: X - Sx
            X_clean_eval = X - self.Sx
            Y_hat = self._contract_product(X_clean_eval, self.B_full, L)
            # 3. 计算 RPE: 必须与原始观测 Y 比较
            norm_y = norm(Y)
            norm_diff = norm(Y - Y_hat)
            rpe = norm_diff / (norm_y + 1e-10)
            # 4.计算稀疏度
            sparsity_sx = (self.Sx.size - np.count_nonzero(self.Sx)) / self.Sx.size
            sparsity_sy = (self.Sy.size - np.count_nonzero(self.Sy)) / self.Sy.size

            loss_history.append(rpe)
            if verbose and iteration % 10 == 0:
                print(
                    f"Iter {iteration}: RPE={rpe:.5f} | Sparsity(Sx)={sparsity_sx:.2%} | Sparsity(Sy)={sparsity_sy:.2%}")
            # 5. 收敛检查
            if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration} with RPE={rpe:.5f}")

                break
        return rpe, self.B_full, sparsity_sx, sparsity_sy


class ROTR_test:
    def __init__(self, **params):
        self.ranks = params['ranks']
        self.x = np.array(params['x'])
        self.y = np.array(params['y'])
        self.L = params['L']
        self.M = params['M']

        # 正则化参数
        self.lambda_x = params['lambda_x']
        self.lambda_y = params['lambda_y']
        self.lambda_b = params['lambda_b']

        # 优化参数
        self.max_iter = params['max_itr']
        self.tol = params['tol']
        self.fista_iter = params['fista_iter']
        self.sx_max_step = params['sx_max_step']
        self.manifold_lr = params['manifold_lr']
        self.init_scale = params['init_scale']  # 控制核张量的数值大小
        self.name = 'rotr'

        # 内部状态
        self.G = None
        self.Uk = []
        self.Sx = None
        self.Sy = None
        self.B_full = None

    def _contract_product(self, X, B, L_dims):
        # 张量收缩 <X, B>_L
        x_ndim = X.ndim
        axes_x = list(range(1, x_ndim))
        axes_b = list(range(0, len(axes_x)))
        return np.tensordot(X, B, axes=(axes_x, axes_b))

    def _reconstruct_B(self):
        # 重构系数张量
        return multi_mode_dot(self.G, self.Uk)

    def fit(self, verbose=True):
        X = self.x
        Y = self.y
        N = X.shape[0]
        L = self.L
        M = self.M

        input_dims = X.shape[1:]
        output_dims = Y.shape[1:]
        total_modes = L + M
        current_ranks = self.ranks

        # --- 1. 初始化 ---
        self.Uk = []
        for i in range(L):
            U, _, _ = svd(np.random.randn(input_dims[i], current_ranks[i]), full_matrices=False)
            # 换成qr分解
            # U,_ = np.linalg.qr(np.random.randn(input_dims[i], current_ranks[i]))
            self.Uk.append(U)
        for i in range(M):
            U, _, _ = svd(np.random.randn(output_dims[i], current_ranks[L + i]), full_matrices=False)
            # U,_ = np.linalg.qr(np.random.randn(input_dims[i], current_ranks[L + i]))
            self.Uk.append(U)

        # 核心张量初始化 (缩放防止初始预测过大)
        self.G = np.random.randn(*current_ranks) * self.init_scale

        self.Sx = np.zeros_like(X)
        self.Sy = np.zeros_like(Y)

        loss_history = []
        print(f"Starting ROTR Optimization (Max Iter: {self.max_iter})...")

        for iteration in range(self.max_iter):
            B = self._reconstruct_B()

            # ==========================================
            # Step 1: 更新响应异常 Sy (Soft Thresholding)
            # ==========================================
            X_clean = X - self.Sx
            Y_pred = self._contract_product(X_clean, B, L)
            Residual_Sy = Y - Y_pred
            Res_before_Sy = Y - Y_pred
            energy_before_Sy = np.linalg.norm(Res_before_Sy)
            self.Sy = soft_thresholding(Residual_Sy, self.lambda_y)
            Res_after_Sy = Res_before_Sy - self.Sy
            energy_after_Sy = np.linalg.norm(Res_after_Sy)

            # ==========================================
            # Step 2: 更新预测变量异常 Sx (ISTA)
            # ==========================================
            Y_tilde = Y - self.Sy
            X_curr = X - self.Sx
            pred = self._contract_product(X_curr, B, L)
            Res_before_Sx = Y_tilde - pred  # 这理论上等于 Res_after_Sy
            energy_before_Sx = np.linalg.norm(Res_before_Sx)

            # 动态计算步长
            lipschitz_const = norm(B.ravel()) ** 2 + 1e-8
            step_size_Sx = 1.0 / lipschitz_const
            step_size_Sx = min(step_size_Sx, self.sx_max_step)  # 截断步长

            for k_ista in range(self.fista_iter):
                X_curr = X - self.Sx
                pred = self._contract_product(X_curr, B, L)
                Residual_Sx = Y_tilde - pred

                axes_res = list(range(1, M + 1))
                axes_B_out = list(range(L, L + M))
                Gradient_Step = np.tensordot(Residual_Sx, B, axes=(axes_res, axes_B_out))

                # 更新 Sx
                Sx_temp = self.Sx + step_size_Sx * Gradient_Step  # 梯度更新方向 取负号，甚至都不影响结果？ 呃呃呃。肯定不影响啊，
                # 你就没有Sx的异常值
                self.Sx = soft_thresholding(Sx_temp, self.lambda_x * step_size_Sx)
            X_new = X - self.Sx
            pred_new = self._contract_product(X_new, B, L)
            Res_after_Sx = Y_tilde - pred_new
            energy_after_Sx = np.linalg.norm(Res_after_Sx)

            if iteration % 10 == 0:
                print(f"Iter {iteration}:")
                print(
                    f"  Sy 消除能量: {energy_before_Sy:.2f} -> {energy_after_Sy:.2f} (Delta: {energy_before_Sy - energy_after_Sy:.2f})")
                print(
                    f"  Sx 消除能量: {energy_before_Sx:.2f} -> {energy_after_Sx:.2f} (Delta: {energy_before_Sx - energy_after_Sx:.2f})")
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

                Uk_new_temp = current_Uk - self.manifold_lr * Grad_Riem  # 黎曼梯度这里取负号呢？ 理论上是取负号，我改成正号了。。
                U_svd, _, Vh_svd = svd(Uk_new_temp, full_matrices=False)
                self.Uk[k] = np.dot(U_svd, Vh_svd)

            # ==========================================
            # 监控与收敛检查
            # ==========================================
            # 需要rpe,sparsity
            # 1. 更新全局系数 B (因为 G 和 Uk 刚刚更新过)
            self.B_full = self._reconstruct_B()
            # 2. 预测时必须使用去除异常后的输入: X - Sx
            X_clean_eval = X - self.Sx
            Y_hat = self._contract_product(X_clean_eval, self.B_full, L)
            # 3. 计算 RPE: 必须与原始观测 Y 比较
            norm_y = norm(Y)
            norm_diff = norm(Y - Y_hat)
            rpe = norm_diff / (norm_y + 1e-10)
            # 4.计算稀疏度
            sparsity_sx = (self.Sx.size - np.count_nonzero(self.Sx)) / self.Sx.size
            sparsity_sy = (self.Sy.size - np.count_nonzero(self.Sy)) / self.Sy.size

            loss_history.append(rpe)
            if verbose and iteration % 10 == 0:
                print(
                    f"Iter {iteration}: RPE={rpe:.5f} | Sparsity(Sx)={sparsity_sx:.2%} | Sparsity(Sy)={sparsity_sy:.2%}")
            # 5. 收敛检查
            if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration} with RPE={rpe:.5f}")

                break
        return rpe, self.B_full, sparsity_sx, sparsity_sy


class ROTR_change_y_and_x:
    def __init__(self, **params):
        self.ranks = params['ranks']
        self.x = np.array(params['x'])
        self.y = np.array(params['y'])
        self.L = params['L']
        self.M = params['M']

        # 正则化参数
        self.lambda_x = params['lambda_x']
        self.lambda_y = params['lambda_y']
        self.lambda_b = params['lambda_b']

        # 优化参数
        self.max_iter = params['max_itr']
        self.tol = params['tol']
        self.fista_iter = params['fista_iter']
        self.sx_max_step = params['sx_max_step']
        self.manifold_lr = params['manifold_lr']
        self.init_scale = params['init_scale']  # 控制核张量的数值大小
        self.name = 'rotr'

        # 内部状态
        self.G = None
        self.Uk = []
        self.Sx = None
        self.Sy = None
        self.B_full = None

    def _contract_product(self, X, B, L_dims):
        # 张量收缩 <X, B>_L
        x_ndim = X.ndim
        axes_x = list(range(1, x_ndim))
        axes_b = list(range(0, len(axes_x)))
        return np.tensordot(X, B, axes=(axes_x, axes_b))

    def _reconstruct_B(self):
        # 重构系数张量
        return multi_mode_dot(self.G, self.Uk)

    def fit(self, verbose=True):
        X = self.x
        Y = self.y
        N = X.shape[0]
        L = self.L
        M = self.M

        input_dims = X.shape[1:]
        output_dims = Y.shape[1:]
        total_modes = L + M
        current_ranks = self.ranks

        # --- 1. 初始化 ---
        self.Uk = []
        for i in range(L):
            U, _, _ = svd(np.random.randn(input_dims[i], current_ranks[i]), full_matrices=False)
            # 换成qr分解
            # U,_ = np.linalg.qr(np.random.randn(input_dims[i], current_ranks[i]))
            self.Uk.append(U)
        for i in range(M):
            U, _, _ = svd(np.random.randn(output_dims[i], current_ranks[L + i]), full_matrices=False)
            # U,_ = np.linalg.qr(np.random.randn(input_dims[i], current_ranks[L + i]))
            self.Uk.append(U)

        # 核心张量初始化 (缩放防止初始预测过大)
        self.G = np.random.randn(*current_ranks) * self.init_scale

        self.Sx = np.zeros_like(X)
        self.Sy = np.zeros_like(Y)

        loss_history = []
        print(f"Starting ROTR Optimization (Max Iter: {self.max_iter})...")

        for iteration in range(self.max_iter):
            B = self._reconstruct_B()

            # ==========================================
            # Step 1: 先更新预测变量异常 Sx (顺序已交换)
            # ==========================================
            # 此时的 Sy 是上一轮遗留的（或初始0）。
            # 我们让 Sx 尝试去拟合 "Y - Sy_old"
            Y_tilde = Y - self.Sy

            # 动态计算步长
            lipschitz_const = norm(B.ravel()) ** 2 + 1e-8
            step_size_Sx = min(1.0 / lipschitz_const, self.sx_max_step)

            for k_ista in range(self.fista_iter):
                X_curr = X - self.Sx
                pred = self._contract_product(X_curr, B, L)
                Residual_Sx = Y_tilde - pred

                axes_res = list(range(1, M + 1))
                axes_B_out = list(range(L, L + M))
                Gradient_Step = np.tensordot(Residual_Sx, B, axes=(axes_res, axes_B_out))

                # 更新 Sx
                Sx_temp = self.Sx + step_size_Sx * Gradient_Step
                self.Sx = soft_thresholding(Sx_temp, self.lambda_x * step_size_Sx)

            # ==========================================
            # Step 2: 后更新响应异常 Sy
            # ==========================================
            # 关键点：使用 **最新更新的 Sx** 计算预测值
            # 这样 Sy 面对的残差是 Sx 实在解释不了的部分
            X_new = X - self.Sx
            Y_pred_new = self._contract_product(X_new, B, L)

            Residual_Sy = Y - Y_pred_new
            self.Sy = soft_thresholding(Residual_Sy, self.lambda_y)

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

                Uk_new_temp = current_Uk - self.manifold_lr * Grad_Riem  # 黎曼梯度这里取负号呢？ 理论上是取负号，我改成正号了。。
                U_svd, _, Vh_svd = svd(Uk_new_temp, full_matrices=False)
                self.Uk[k] = np.dot(U_svd, Vh_svd)

            # ==========================================
            # 监控与收敛检查
            # ==========================================
            # 需要rpe,sparsity
            # 1. 更新全局系数 B (因为 G 和 Uk 刚刚更新过)
            self.B_full = self._reconstruct_B()
            # 2. 预测时必须使用去除异常后的输入: X - Sx
            X_clean_eval = X - self.Sx
            Y_hat = self._contract_product(X_clean_eval, self.B_full, L)
            # 3. 计算 RPE: 必须与原始观测 Y 比较
            norm_y = norm(Y)
            norm_diff = norm(Y - Y_hat)
            rpe = norm_diff / (norm_y + 1e-10)
            # 4.计算稀疏度
            sparsity_sx = (self.Sx.size - np.count_nonzero(self.Sx)) / self.Sx.size
            sparsity_sy = (self.Sy.size - np.count_nonzero(self.Sy)) / self.Sy.size

            loss_history.append(rpe)
            if verbose and iteration % 10 == 0:
                print(
                    f"Iter {iteration}: RPE={rpe:.5f} | Sparsity(Sx)={sparsity_sx:.2%} | Sparsity(Sy)={sparsity_sy:.2%}")
            # 5. 收敛检查
            if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration} with RPE={rpe:.5f}")

                break
        return rpe, self.B_full, sparsity_sx, sparsity_sy


class ROTR_HOSVD:
    def __init__(self, **params):
        self.ranks = params['ranks']
        self.x = np.array(params['x'])
        self.y = np.array(params['y'])
        self.L = params['L']
        self.M = params['M']

        # 正则化参数
        self.lambda_x = params['lambda_x']
        self.lambda_y = params['lambda_y']
        self.lambda_b = params['lambda_b']

        # 优化参数
        self.max_iter = params['max_itr']
        self.tol = params['tol']
        self.fista_iter = params['fista_iter']
        self.sx_max_step = params['sx_max_step']
        self.manifold_lr = params['manifold_lr']
        self.init_scale = params['init_scale']  # 控制核张量的数值大小
        self.name = 'rotr'

        # 内部状态
        self.G = None
        self.Uk = []
        self.Sx = None
        self.Sy = None
        self.B_full = None

    def _contract_product(self, X, B, L_dims):
        # 张量收缩 <X, B>_L
        x_ndim = X.ndim
        axes_x = list(range(1, x_ndim))
        axes_b = list(range(0, len(axes_x)))
        return np.tensordot(X, B, axes=(axes_x, axes_b))

    def _reconstruct_B(self):
        # 重构系数张量
        return multi_mode_dot(self.G, self.Uk)

    def fit(self, verbose=True):
        X = self.x
        Y = self.y
        N = X.shape[0]
        L = self.L
        M = self.M

        input_dims = X.shape[1:]
        output_dims = Y.shape[1:]
        total_modes = L + M
        current_ranks = self.ranks

        # 1. 初始化: HOSVD (核心修改)
        # ==========================================
        self.Uk = []

        # --- 对输入 X 做 HOSVD 初始化 ---
        # 即使 X 包含稀疏异常，SVD 依然能提取出主要子空间
        for i in range(L):
            # 将第 i+1 个特征模态移到最前，然后展平
            # X shape: (N, d1, d2...) -> Mode i corresponds to axis i+1
            mat = np.moveaxis(X, i + 1, 0).reshape(input_dims[i], -1)
            U, _, _ = svd(mat, full_matrices=False)
            self.Uk.append(U[:, :current_ranks[i]])

        # --- 对输出 Y 做 HOSVD 初始化 ---
        for i in range(M):
            mat = np.moveaxis(Y, i + 1, 0).reshape(output_dims[i], -1)
            U, _, _ = svd(mat, full_matrices=False)
            self.Uk.append(U[:, :current_ranks[L + i]])

        self.G = np.random.randn(*current_ranks) * self.init_scale  # G 还是随机，影响不大

        # 核心张量初始化 (缩放防止初始预测过大)
        # self.G = np.random.randn(*current_ranks) * self.init_scale

        self.Sx = np.zeros_like(X)
        self.Sy = np.zeros_like(Y)

        loss_history = []
        print(f"Starting ROTR Optimization (Max Iter: {self.max_iter})...")

        for iteration in range(self.max_iter):
            B = self._reconstruct_B()

            # ==========================================
            # Step 1: 先更新 Sx
            # ==========================================
            Y_target = Y - self.Sy
            lipschitz_const = norm(B.ravel()) ** 2 + 1e-8
            step_size_Sx = min(1.0 / lipschitz_const, self.sx_max_step)

            for k_ista in range(self.fista_iter):
                X_curr = X - self.Sx
                pred = self._contract_product(X_curr, B, L)
                Residual_Sx = Y_target - pred
                axes_res = list(range(1, M + 1))
                axes_B_out = list(range(L, L + M))
                Gradient_Step = np.tensordot(Residual_Sx, B, axes=(axes_res, axes_B_out))
                self.Sx = soft_thresholding(self.Sx + step_size_Sx * Gradient_Step,
                                            self.lambda_x * step_size_Sx)

            # ==========================================
            # Step 2: 后更新 Sy
            # ==========================================
            X_new = X - self.Sx
            Y_pred_new = self._contract_product(X_new, B, L)
            Residual_Sy = Y - Y_pred_new
            self.Sy = soft_thresholding(Residual_Sy, self.lambda_y)

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

                Uk_new_temp = current_Uk - self.manifold_lr * Grad_Riem  # 黎曼梯度这里取负号呢？ 理论上是取负号，我改成正号了。。
                U_svd, _, Vh_svd = svd(Uk_new_temp, full_matrices=False)
                self.Uk[k] = np.dot(U_svd, Vh_svd)

            # ==========================================
            # 监控与收敛检查
            # ==========================================
            # 需要rpe,sparsity
            # 1. 更新全局系数 B (因为 G 和 Uk 刚刚更新过)
            self.B_full = self._reconstruct_B()
            # 2. 预测时必须使用去除异常后的输入: X - Sx
            X_clean_eval = X - self.Sx
            Y_hat = self._contract_product(X_clean_eval, self.B_full, L)
            # 3. 计算 RPE: 必须与原始观测 Y 比较
            norm_y = norm(Y)
            norm_diff = norm(Y - Y_hat)
            rpe = norm_diff / (norm_y + 1e-10)
            # 4.计算稀疏度
            sparsity_sx = (self.Sx.size - np.count_nonzero(self.Sx)) / self.Sx.size
            sparsity_sy = (self.Sy.size - np.count_nonzero(self.Sy)) / self.Sy.size

            loss_history.append(rpe)
            if verbose and iteration % 10 == 0:
                print(
                    f"Iter {iteration}: RPE={rpe:.5f} | Sparsity(Sx)={sparsity_sx:.2%} | Sparsity(Sy)={sparsity_sy:.2%}")
            # 5. 收敛检查
            if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration} with RPE={rpe:.5f}")

                break
        return rpe, self.B_full, sparsity_sx, sparsity_sy


class ROTR_HOSVD_pre_solve_G:
    def __init__(self, **params):
        self.ranks = params['ranks']
        self.x = np.array(params['x'])
        self.y = np.array(params['y'])
        self.L = params['L']
        self.M = params['M']

        # 正则化参数
        self.lambda_x = params['lambda_x']
        self.lambda_y = params['lambda_y']
        self.lambda_b = params['lambda_b']

        # 优化参数
        self.max_iter = params['max_itr']
        self.tol = params['tol']
        self.fista_iter = params['fista_iter']
        self.sx_max_step = params['sx_max_step']
        self.manifold_lr = params['manifold_lr']
        self.init_scale = params['init_scale']  # 控制核张量的数值大小
        self.name = 'rotr'

        # 内部状态
        self.G = None
        self.Uk = []
        self.Sx = None
        self.Sy = None
        self.B_full = None

    def _contract_product(self, X, B, L_dims):
        # 张量收缩 <X, B>_L
        x_ndim = X.ndim
        axes_x = list(range(1, x_ndim))
        axes_b = list(range(0, len(axes_x)))
        return np.tensordot(X, B, axes=(axes_x, axes_b))

    def _reconstruct_B(self):
        # 重构系数张量
        return multi_mode_dot(self.G, self.Uk)

    # 封装 G 的更新逻辑，因为要用两次
    def _update_G(self, X_clean, Y_clean):
        L, M = self.L, self.M
        # 1. 全压缩：将 X, Y 投影到核心空间
        X_tilde = X_clean.copy()
        for i in range(L):
            X_tilde = mode_dot(X_tilde, self.Uk[i].T, mode=i + 1)

        Y_tilde_core = Y_clean.copy()
        for i in range(M):
            Y_tilde_core = mode_dot(Y_tilde_core, self.Uk[L + i].T, mode=i + 1)

        # 2. 构建 CG 问题
        # RHS = X_tilde^T * Y_tilde (注意 tensordot 轴)
        RHS_G = np.tensordot(X_tilde, Y_tilde_core, axes=([0], [0]))
        b_vec = RHS_G.ravel()
        dim_G = np.prod(self.ranks)

        # 线性算子 A * vec(G)
        def matvec_G(v):
            G_curr = v.reshape(self.ranks)
            axes_X = list(range(1, L + 1))
            axes_G = list(range(L))
            # Predict: X_tilde * G
            Y_pred_core = np.tensordot(X_tilde, G_curr, axes=(axes_X, axes_G))
            # Backproj: X_tilde^T * Prediction
            delta_G = np.tensordot(X_tilde, Y_pred_core, axes=([0], [0]))
            return (delta_G + self.lambda_b * G_curr).ravel()

        A_op = LinearOperator((dim_G, dim_G), matvec=matvec_G)

        # 初始猜测：如果是第一次，用0；否则用当前的 G
        x0 = self.G.ravel() if self.G is not None else np.zeros(dim_G)

        G_opt_vec, info = cg(A_op, b_vec, x0=x0, tol=1e-5)
        self.G = G_opt_vec.reshape(self.ranks)

    def fit(self, verbose=True):
        X = self.x
        Y = self.y
        N = X.shape[0]
        L = self.L
        M = self.M

        input_dims = X.shape[1:]
        output_dims = Y.shape[1:]
        total_modes = L + M
        current_ranks = self.ranks

        # ==========================================
        # 1. 初始化: HOSVD + Pre-solve G
        # ==========================================
        self.Uk = []

        # HOSVD on X
        for i in range(L):
            mat = np.moveaxis(X, i + 1, 0).reshape(input_dims[i], -1)
            U, _, _ = svd(mat, full_matrices=False)
            self.Uk.append(U[:, :current_ranks[i]])

        # HOSVD on Y
        for i in range(M):
            mat = np.moveaxis(Y, i + 1, 0).reshape(output_dims[i], -1)
            U, _, _ = svd(mat, full_matrices=False)
            self.Uk.append(U[:, :current_ranks[L + i]])

        self.Sx = np.zeros_like(X)
        self.Sy = np.zeros_like(Y)

        # --- 关键修正：Step 0 ---
        # 基于刚算好的 Uk，求解最优的 G，建立 X 和 Y 的联系！
        # 此时 Sx, Sy 都是 0，相当于先做一次非鲁棒的 Tensor Regression 初始化
        print("Pre-solving G based on HOSVD factors...")
        self._update_G(X, Y)  # 这里的 G 不再是随机的了！

        loss_history = []
        print(f"Starting ROTR (HOSVD + Pre-solve G, Max Iter: {self.max_iter})...")

        for iteration in range(self.max_iter):
            B = self._reconstruct_B()

            # ==========================================
            # Step 1: 先更新 Sx
            # ==========================================
            Y_target = Y - self.Sy
            lipschitz_const = norm(B.ravel()) ** 2 + 1e-8
            step_size_Sx = min(1.0 / lipschitz_const, self.sx_max_step)

            for k_ista in range(self.fista_iter):
                X_curr = X - self.Sx
                pred = self._contract_product(X_curr, B, L)
                Residual_Sx = Y_target - pred
                axes_res = list(range(1, M + 1))
                axes_B_out = list(range(L, L + M))
                Gradient_Step = np.tensordot(Residual_Sx, B, axes=(axes_res, axes_B_out))
                self.Sx = soft_thresholding(self.Sx + step_size_Sx * Gradient_Step,
                                            self.lambda_x * step_size_Sx)

            # ==========================================
            # Step 2: 后更新 Sy
            # ==========================================
            X_new = X - self.Sx
            Y_pred_new = self._contract_product(X_new, B, L)
            Residual_Sy = Y - Y_pred_new
            self.Sy = soft_thresholding(Residual_Sy, self.lambda_y)

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

                Uk_new_temp = current_Uk - self.manifold_lr * Grad_Riem  # 黎曼梯度这里取负号呢？ 理论上是取负号，我改成正号了。。
                U_svd, _, Vh_svd = svd(Uk_new_temp, full_matrices=False)
                self.Uk[k] = np.dot(U_svd, Vh_svd)

            # ==========================================
            # 监控与收敛检查
            # ==========================================
            # 需要rpe,sparsity
            # 1. 更新全局系数 B (因为 G 和 Uk 刚刚更新过)
            self.B_full = self._reconstruct_B()
            # 2. 预测时必须使用去除异常后的输入: X - Sx
            X_clean_eval = X - self.Sx
            Y_hat = self._contract_product(X_clean_eval, self.B_full, L)
            # 3. 计算 RPE: 必须与原始观测 Y 比较
            norm_y = norm(Y)
            norm_diff = norm(Y - Y_hat)
            rpe = norm_diff / (norm_y + 1e-10)
            # 4.计算稀疏度
            sparsity_sx = (self.Sx.size - np.count_nonzero(self.Sx)) / self.Sx.size
            sparsity_sy = (self.Sy.size - np.count_nonzero(self.Sy)) / self.Sy.size

            loss_history.append(rpe)
            if verbose and iteration % 10 == 0:
                print(
                    f"Iter {iteration}: RPE={rpe:.5f} | Sparsity(Sx)={sparsity_sx:.2%} | Sparsity(Sy)={sparsity_sy:.2%}")
            # 5. 收敛检查
            if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration} with RPE={rpe:.5f}")

                break
        return rpe, self.B_full, sparsity_sx, sparsity_sy


class ROTR_mixture:
    def __init__(self, **params):
        self.ranks = params['ranks']
        self.x = np.array(params['x'])
        self.y = np.array(params['y'])
        self.L = params['L']
        self.M = params['M']

        # 正则化参数
        self.lambda_x = params['lambda_x']
        self.lambda_y = params['lambda_y']
        self.lambda_b = params['lambda_b']

        # 优化参数
        self.max_iter = params['max_itr']
        self.tol = params['tol']
        self.fista_iter = params['fista_iter']
        self.sx_max_step = params['sx_max_step']
        self.manifold_lr = params['manifold_lr']
        self.init_scale = params['init_scale']  # 控制核张量的数值大小
        self.name = 'rotr'

        # 内部状态
        self.G = None
        self.Uk = []
        self.Sx = None
        self.Sy = None
        self.B_full = None
        # === 新增参数：重启轮次 ===
        self.restart_epoch = 15

    def _contract_product(self, X, B, L_dims):
        # 张量收缩 <X, B>_L
        x_ndim = X.ndim
        axes_x = list(range(1, x_ndim))
        axes_b = list(range(0, len(axes_x)))
        return np.tensordot(X, B, axes=(axes_x, axes_b))

    def _reconstruct_B(self):
        # 重构系数张量
        return multi_mode_dot(self.G, self.Uk)

    # 封装 G 的更新逻辑，因为要用两次
    def _update_G(self, X_clean, Y_clean):
        L, M = self.L, self.M
        # 1. 全压缩：将 X, Y 投影到核心空间
        X_tilde = X_clean.copy()
        for i in range(L):
            X_tilde = mode_dot(X_tilde, self.Uk[i].T, mode=i + 1)

        Y_tilde_core = Y_clean.copy()
        for i in range(M):
            Y_tilde_core = mode_dot(Y_tilde_core, self.Uk[L + i].T, mode=i + 1)

        # 2. 构建 CG 问题
        # RHS = X_tilde^T * Y_tilde (注意 tensordot 轴)
        RHS_G = np.tensordot(X_tilde, Y_tilde_core, axes=([0], [0]))
        b_vec = RHS_G.ravel()
        dim_G = np.prod(self.ranks)

        # 线性算子 A * vec(G)
        def matvec_G(v):
            G_curr = v.reshape(self.ranks)
            axes_X = list(range(1, L + 1))
            axes_G = list(range(L))
            # Predict: X_tilde * G
            Y_pred_core = np.tensordot(X_tilde, G_curr, axes=(axes_X, axes_G))
            # Backproj: X_tilde^T * Prediction
            delta_G = np.tensordot(X_tilde, Y_pred_core, axes=([0], [0]))
            return (delta_G + self.lambda_b * G_curr).ravel()

        A_op = LinearOperator((dim_G, dim_G), matvec=matvec_G)

        # 初始猜测：如果是第一次，用0；否则用当前的 G
        x0 = self.G.ravel() if self.G is not None else np.zeros(dim_G)

        G_opt_vec, info = cg(A_op, b_vec, x0=x0, tol=1e-5)
        self.G = G_opt_vec.reshape(self.ranks)

    def fit(self, verbose=True):
        X = self.x
        Y = self.y
        N = X.shape[0]
        L = self.L
        M = self.M

        input_dims = X.shape[1:]
        output_dims = Y.shape[1:]
        total_modes = L + M
        current_ranks = self.ranks

        # ==========================================
        # 1. 初始：完全随机 (Cold Start)
        # ==========================================
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
        print(f"Starting ROTR (Random Init -> Restart at {self.restart_epoch})...")

        for iteration in range(self.max_iter):

            # ==========================================
            # 关键步骤：在第 15 轮进行“热重启”
            # ==========================================
            if iteration == self.restart_epoch:
                print("\n>>> Triggering Warm Restart (HOSVD on Cleaned Data)...")

                # 1. 获取清洗后的数据
                X_clean_est = X - self.Sx
                # (可选：对 Y 也清洗，如果 Y 异常很多)
                Y_clean_est = Y - self.Sy

                # 2. 重新初始化 Uk (基于清洗数据)
                new_Uk = []
                # Input modes
                for i in range(L):
                    mat = np.moveaxis(X_clean_est, i + 1, 0).reshape(input_dims[i], -1)
                    U, _, _ = svd(mat, full_matrices=False)
                    new_Uk.append(U[:, :current_ranks[i]])
                # Output modes
                for i in range(M):
                    mat = np.moveaxis(Y_clean_est, i + 1, 0).reshape(output_dims[i], -1)
                    U, _, _ = svd(mat, full_matrices=False)
                    new_Uk.append(U[:, :current_ranks[L + i]])
                self.Uk = new_Uk

                # 3. 重新求解最优 G
                self._update_G(X_clean_est, Y_clean_est)

                # 4. 降低流形学习率 (因为现在 Uk 已经很好了)
                self.manifold_lr = 0.05
                print(">>> Restart Done. Reduced manifold_lr. Continuing...\n")

            # 以下是标准循环
            B = self._reconstruct_B()

            # Step 1: Sx
            Y_target = Y - self.Sy
            lipschitz_const = norm(B.ravel()) ** 2 + 1e-8
            step_size_Sx = min(1.0 / lipschitz_const, self.sx_max_step)

            for k_ista in range(self.fista_iter):
                X_curr = X - self.Sx
                pred = self._contract_product(X_curr, B, L)
                Residual_Sx = Y_target - pred
                axes_res = list(range(1, M + 1))
                axes_B_out = list(range(L, L + M))
                Gradient_Step = np.tensordot(Residual_Sx, B, axes=(axes_res, axes_B_out))

                self.Sx = soft_thresholding(self.Sx + step_size_Sx * Gradient_Step,
                                            self.lambda_x * step_size_Sx)

            # Step 2: Sy
            X_new = X - self.Sx
            Y_pred_new = self._contract_product(X_new, B, L)
            Residual_Sy = Y - Y_pred_new
            self.Sy = soft_thresholding(Residual_Sy, self.lambda_y)

            # Step 3: G
            self._update_G(X - self.Sx, Y - self.Sy)

            # Step 4: Uk
            X_clean = X - self.Sx
            Y_clean = Y - self.Sy
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

                Uk_new_temp = current_Uk - self.manifold_lr * Grad_Riem
                U_svd, _, Vh_svd = svd(Uk_new_temp, full_matrices=False)
                self.Uk[k] = np.dot(U_svd, Vh_svd)

            # Monitor
            self.B_full = self._reconstruct_B()
            Y_target_clean = Y - self.Sy
            Y_hat = self._contract_product(X - self.Sx, self.B_full, L)
            clean_rpe = norm(Y_target_clean - Y_hat) / (norm(Y_target_clean) + 1e-10)

            loss_history.append(clean_rpe)

            sparsity_sx = (self.Sx.size - np.count_nonzero(self.Sx)) / self.Sx.size
            sparsity_sy = (self.Sy.size - np.count_nonzero(self.Sy)) / self.Sy.size

            if verbose and iteration % 5 == 0:
                print(
                    f"Iter {iteration}: Clean RPE={clean_rpe:.5f} | Sp(Sx)={sparsity_sx:.2%} | Sp(Sy)={sparsity_sy:.2%}")

            if iteration > self.restart_epoch + 5 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                if verbose: print(f"Converged at {iteration}")
                break

        return clean_rpe, self.B_full, sparsity_sx, sparsity_sy


