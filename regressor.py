import math

import torch
import tensorly as tl
import numpy as np
import data_io as io
import pandas as pd
import plotly.express as px

from tensorly.tenalg.proximal import soft_thresholding, svd_thresholding
from tensorly.decomposition import robust_pca
from numpy.linalg import norm
from tensorly.tenalg import mode_dot, multi_mode_dot
import os
from scipy.linalg import svd, norm
from scipy.sparse.linalg import cg, LinearOperator

def ttt(x, b, L, dims):
    return np.tensordot(
        x,
        b,
        axes=[
            [k + 1 for k in range(len(dims[:L]))],
            [k for k in range(len(dims[:L]))]
        ]
    )



def aic(y_true, y_pre, s_pre, r, b):
    # print(np.log(norm(y_true - y_pre - s_pre) ** 2))
    # print(np.count_nonzero(s_pre), s_pre.size, np.log(np.count_nonzero(s_pre)))
    # print(.5 * r)
    return np.log(norm(y_true - y_pre - s_pre) ** 2) + 2 * np.log(np.count_nonzero(s_pre)) + .5 * r


class TOT:
    def __init__(self, **params):
        self.dims = params['dims']
        self.idx = [i for i in range(len(self.dims))]
        self.L = params['L']
        self.M = params['M']
        self.N = params['N']
        self.R = params['R']
        self.max_itr = params['max_itr']
        self.x = params['x']
        self.y = params['y']
        self.y_vec = tl.tensor_to_vec(params['y'])
        self.yms = [tl.unfold(self.y, i) for i in range(len(self.y.shape))]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_vec_cuda = torch.tensor(self.y_vec, device=device)
        self.yms_cuda = [torch.tensor(ym, device=device) for ym in self.yms]
        self.name = 'tot'

    def fit(self, verbose=False):
        results = []
        U = [np.random.rand(p, self.R) for p in self.dims]
        itr = 0

        if verbose:
            print('======================')
            print('         TOT          ')
            print('======================')

        while itr < self.max_itr:
            for i in range(len(self.dims[:self.L])):
                C = []
                for r in range(self.R):
                    cr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != i]
                    cr = tl.cp_to_tensor((None, cr))
                    cr = np.tensordot(
                        self.x,
                        cr,
                        axes=[
                            [k + 1 for k in range(len(self.dims[:self.L])) if k != i],
                            [k for k in [l for l in range(len(cr.shape))][:-self.M]]
                        ]
                    )
                    cr = tl.unfold(cr, mode=1)
                    C.append(cr)
                C = np.concatenate(C)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                C = torch.tensor(C, device=device)
                u = (torch.pinverse(C.T) @ self.y_vec_cuda).reshape(-1, self.R)
                U[i] = u.cpu().numpy()
                # U[i] = (np.linalg.pinv(C.T) @ tl.tensor_to_vec(self.y)).reshape(-1, self.R)

            for i in range(len(self.dims[self.L:])):
                D = []
                for r in range(self.R):
                    dr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != self.L + i]
                    dr = tl.cp_to_tensor((None, dr))
                    dr = ttt(self.x, dr, self.L, self.dims)
                    D.append(tl.tensor_to_vec(dr).reshape(-1, 1))
                D = np.concatenate(D, axis=1)
                D = torch.tensor(D, device=device)
                U[self.L + i] = (torch.pinverse(D) @ self.yms_cuda[i + 1].T).T.cpu().numpy()
                # U[self.L + i] = (np.linalg.pinv(D) @ tl.unfold(self.y, mode=i + 1).T).T

            B = tl.cp_to_tensor((None, U))
            y_pre = ttt(self.x, B, self.L, self.dims)
            rpe = (np.linalg.norm(self.y - y_pre)) / np.linalg.norm(self.y)

            if len(results) > 1 and abs(rpe - results[-1][0]) < 1e-4:
                break
            if verbose:
                msg = 'itr={}, rpe={:.4f}'.format(itr + 1, rpe)
                print(msg)

            results.append((rpe, B, 0, 0))
            itr += 1

        if verbose:
            print('======================')
        results = sorted(results, key=lambda x: x[0])
        return results[-1][0], results[-1][1], results[-1][2], results[-1][3]

class RTOT:
    def __init__(self, **params):
        self.dims = params['dims']
        self.idx = [i for i in range(len(self.dims))]
        self.L = params['L']
        self.M = params['M']
        self.N = params['N']
        self.R = params['R']
        self.max_itr = params['max_itr']
        self.x = params['x']
        self.y = params['y']
        self.mu1 = params['mu1']
        self.mu2 = params['mu2']
        self.mu3 = params['mu3']
        self.tol = params['tol']
        self.y_vec = tl.tensor_to_vec(params['y'])
        self.yms = [tl.unfold(self.y, i) for i in range(len(self.y.shape))]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_vec_cuda = torch.tensor(self.y_vec, device=device)
        self.yms_cuda = [torch.tensor(ym, device=device) for ym in self.yms]
        self.name = 'rtot'

    def fit(self, verbose=False):
        # U = [np.random.rand(p, self.R) for p in self.dims]
        # J = [np.random.rand(p, self.R) for p in self.dims]
        U = [np.random.normal(0,0.1,size =(p, self.R)) for p in self.dims]
        J = [np.random.normal(0,0.1,size =(p, self.R)) for p in self.dims]
        Z = [np.zeros_like(u) for u in U]
        S = np.zeros_like(self.y)
        results = []
        Bs = [tl.cp_to_tensor((None, U))]
        Js = [tl.cp_to_tensor((None, J))]
        itr = 0

        if verbose:
            print('======================')
            print('        RTOT          ')
            print('======================')

        while itr < self.max_itr:
            for i in range(len(J)):
                J[i] = svd_thresholding(U[i] + (1 / self.mu3) * Z[i], 1 / self.mu3)

            for i in range(len(self.dims[:self.L])):
                C = []
                for r in range(self.R):
                    cr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != i]
                    cr = tl.cp_to_tensor((None, cr))
                    cr = np.tensordot(
                        self.x,
                        cr,
                        axes=[
                            [k + 1 for k in range(len(self.dims[:self.L])) if k != i],
                            [k for k in [l for l in range(len(cr.shape))][:-self.M]]
                        ]
                    )
                    cr = tl.unfold(cr, mode=1)
                    C.append(cr)
                C = np.concatenate(C)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                C = torch.tensor(C, device=device)
                CCT = C @ C.T
                s_vec = torch.tensor(S, device=device).flatten()
                j_vec = torch.tensor(J[i], device=device).flatten()
                z_vec = torch.tensor(Z[i], device=device).flatten()
                U[i] = (torch.pinverse(self.mu1 * CCT + self.mu3 * torch.eye(n=CCT.shape[0], device=device)) @ (self.mu1 * C @ (self.y_vec_cuda - s_vec) + self.mu3 * j_vec - z_vec)).reshape(-1,
                                                                                                                                                                                              self.R).cpu().numpy()
                # y_vec = tl.tensor_to_vec(self.y)
                # s_vec = tl.tensor_to_vec(S)
                # j_vec = tl.tensor_to_vec(J[i])
                # z_vec = tl.tensor_to_vec(Z[i])
                # CCT = C @ C.T
                # U[i] = (np.linalg.pinv(self.mu1 * CCT + self.mu3 * np.eye(CCT.shape[0])) @ (self.mu1 * C @ (y_vec - s_vec) + self.mu3 * j_vec - z_vec)).reshape(-1, self.R)

            for i in range(len(self.dims[self.L:])):
                D = []
                for r in range(self.R):
                    dr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != self.L + i]
                    dr = tl.cp_to_tensor((None, dr))
                    dr = ttt(self.x, dr, self.L, self.dims)
                    D.append(tl.tensor_to_vec(dr).reshape(-1, 1))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                D = np.concatenate(D, axis=1)
                D = torch.tensor(D, device=device)
                DTD = D.T @ D
                ym = self.yms_cuda[i + 1].T
                sm = torch.tensor(tl.unfold(S, mode=i + 1).T, device=device)
                J_cuda = torch.tensor(J[self.L + i], device=device)
                Z_cuda = torch.tensor(Z[self.L + i], device=device)
                U[self.L + i] = (torch.pinverse(self.mu1 * DTD + self.mu3 * torch.eye(n=DTD.shape[0], device=device)) @ (self.mu1 * D.T @ (ym - sm) + self.mu3 * J_cuda.T - Z_cuda.T)).T.cpu().numpy()
                # DTD = D.T @ D
                # ym = tl.unfold(self.y, mode=i + 1).T
                # sm = tl.unfold(S, mode=i + 1).T
                # U[self.L + i] = (np.linalg.pinv(self.mu1 * DTD + self.mu3 * np.eye(DTD.shape[0])) @ (self.mu1 * D.T @ (ym - sm) + self.mu3 * J[self.L + i].T - Z[self.L + i].T)).T

            for i in range(len(Z)):
                Z[i] += self.mu3 * (U[i] - J[i])

            B = tl.cp_to_tensor((None, U))
            J_ = tl.cp_to_tensor((None, J))
            y_pre = ttt(self.x, B, self.L, self.dims)
            S = soft_thresholding(self.y - y_pre, self.mu2 / self.mu1)
            rpe = (np.linalg.norm(self.y - y_pre)) / np.linalg.norm(self.y)
            norm_s = tl.norm(Js[-1] - J_, 2) ** 2  # 衡量变量更新幅度  U
            norm_r = tl.norm(Bs[-1] - B, 2) ** 2 #  衡量变量一致性。
            sparsity = (S.size - np.count_nonzero(S)) / S.size
            AIC = aic(self.y, y_pre, S, self.R, B)

            #need to uncomment when normal use
            if norm_r > 2 * norm_s:
                self.mu3 *= 2
            elif norm_s > 2 * norm_r:
                self.mu3 /= 2

            if verbose and itr % 10 == 0:
                msg = 'itr={}, rpe={:.4f}, norm_s={:.4f}, norm_r={:.4f}, sparsity={:.2f}'.format(itr + 1, rpe, norm_s, norm_r, sparsity)
                print(msg)

            if len(results) > 1 and abs(rpe - results[-1][0]) < 1e-4:
                break
            if rpe > 0.5:
                break
            if itr > self.max_itr or (norm_r < self.tol and norm_s < self.tol):
                break
            if norm_r < math.inf:
                results.append((rpe, B, norm_r, AIC, S))

            Bs.append(B)
            Js.append(J_)

            itr += 1

        if verbose:
            print('======================')

        # results = sorted(results, key=lambda x: x[0])
        return results[-1][0], results[-1][1], results[-1][3], results[-1][4]

class RPCA:
    def __init__(self, **params):
        self.params = params
        self.reg = params['reg'] if 'reg' in params else 1.5e-1
        self.name = 'rpca'

    def fit(self, verbose=False):
        if verbose:
            print('======================')
            print('        RPCA          ')
            print('======================')

        y, S = robust_pca(self.params['y'], reg_E=self.reg, verbose=verbose)


        if verbose:
            print('sparsity={:.2f}'.format((S.size - np.count_nonzero(S)) / S.size))

        self.params['y'] = y
        tot = TOT(**self.params)
        return tot.fit(verbose=verbose)





class TOT_tucker:
    def __init__(self, **params):
        self.dims = params['dims']
        self.idx = [i for i in range(len(self.dims))]
        self.L = params['L']
        self.M = params['M']
        self.N = params['N']
        self.R = params['R']
        self.max_itr = params['max_itr']
        self.x = params['x']
        self.y = params['y']
        self.y_vec = tl.tensor_to_vec(params['y'])
        self.yms = [tl.unfold(self.y, i) for i in range(len(self.y.shape))]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_vec_cuda = torch.tensor(self.y_vec, device=device)
        self.yms_cuda = [torch.tensor(ym, device=device) for ym in self.yms]
        self.name = 'tot'

    def fit(self, verbose=False):
        results = []
        #U = [np.random.rand(p, self.R) for p in self.dims]
        U = [np.random.normal(0,0.1,size =(p, self.R)) for p in self.dims]
        itr = 0

        if verbose:
            print('======================')
            print('         TOT          ')
            print('======================')

        while itr < self.max_itr:
            for i in range(len(self.dims[:self.L])):
                C = []
                for r in range(self.R):
                    cr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != i]
                    cr = tl.cp_to_tensor((None, cr))
                    cr = np.tensordot(
                        self.x,
                        cr,
                        axes=[
                            [k + 1 for k in range(len(self.dims[:self.L])) if k != i],
                            [k for k in [l for l in range(len(cr.shape))][:-self.M]]
                        ]
                    )
                    cr = tl.unfold(cr, mode=1)
                    C.append(cr)
                C = np.concatenate(C)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                C = torch.tensor(C, device=device)
                u = (torch.pinverse(C.T) @ self.y_vec_cuda).reshape(-1, self.R)
                U[i] = u.cpu().numpy()
                # U[i] = (np.linalg.pinv(C.T) @ tl.tensor_to_vec(self.y)).reshape(-1, self.R)

            for i in range(len(self.dims[self.L:])):
                D = []
                for r in range(self.R):
                    dr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != self.L + i]
                    dr = tl.cp_to_tensor((None, dr))
                    dr = ttt(self.x, dr, self.L, self.dims)
                    D.append(tl.tensor_to_vec(dr).reshape(-1, 1))
                D = np.concatenate(D, axis=1)
                D = torch.tensor(D, device=device)
                U[self.L + i] = (torch.pinverse(D) @ self.yms_cuda[i + 1].T).T.cpu().numpy()
                # U[self.L + i] = (np.linalg.pinv(D) @ tl.unfold(self.y, mode=i + 1).T).T

            B = tl.cp_to_tensor((None, U))
            y_pre = ttt(self.x, B, self.L, self.dims)
            rpe = (np.linalg.norm(self.y - y_pre)) / np.linalg.norm(self.y)

            if len(results) > 1 and abs(rpe - results[-1][0]) < 1e-4:
                break
            if verbose:
                msg = 'itr={}, rpe={:.4f}'.format(itr + 1, rpe)
                print(msg)

            results.append((rpe, B, 0, 0))
            itr += 1

        if verbose:
            print('======================')
        results = sorted(results, key=lambda x: x[0])
        return results[-1][0], results[-1][1], results[-1][2], results[-1][3]

class RTOT_tucker:
    def __init__(self, **params):
        self.dims = params['dims']
        self.idx = [i for i in range(len(self.dims))]
        self.L = params['L']
        self.M = params['M']
        self.N = params['N']
        self.R = params['R']
        self.max_itr = params['max_itr']
        self.x = params['x']
        self.y = params['y']
        self.mu1 = params['mu1']
        self.mu2 = params['mu2']
        self.mu3 = params['mu3']
        self.tol = params['tol']
        self.y_vec = tl.tensor_to_vec(params['y'])
        self.yms = [tl.unfold(self.y, i) for i in range(len(self.y.shape))]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_vec_cuda = torch.tensor(self.y_vec, device=device)
        self.yms_cuda = [torch.tensor(ym, device=device) for ym in self.yms]
        self.name = 'rtot'

    def fit(self, verbose=False):
        # 修改为: 使用更小的初始化，
        U = [np.random.normal(0,0.1,size =(p, self.R)) for p in self.dims]
        J = [np.random.normal(0,0.1,size =(p, self.R)) for p in self.dims]
        Z = [np.zeros_like(u) for u in U]
        S = np.zeros_like(self.y)
        results = []
        Bs = [tl.cp_to_tensor((None, U))]
        Js = [tl.cp_to_tensor((None, J))]
        itr = 0

        if verbose:
            print('======================')
            print('        RTOT          ')
            print('======================')

        while itr < self.max_itr:
            for i in range(len(J)):
                J[i] = svd_thresholding(U[i] + (1 / self.mu3) * Z[i], 1 / self.mu3)

            for i in range(len(self.dims[:self.L])):
                C = []
                for r in range(self.R):
                    cr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != i]
                    cr = tl.cp_to_tensor((None, cr))
                    cr = np.tensordot(
                        self.x,
                        cr,
                        axes=[
                            [k + 1 for k in range(len(self.dims[:self.L])) if k != i],
                            [k for k in [l for l in range(len(cr.shape))][:-self.M]]
                        ]
                    )
                    cr = tl.unfold(cr, mode=1)
                    C.append(cr)
                C = np.concatenate(C)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                C = torch.tensor(C, device=device)
                CCT = C @ C.T
                s_vec = torch.tensor(S, device=device).flatten()
                j_vec = torch.tensor(J[i], device=device).flatten()
                z_vec = torch.tensor(Z[i], device=device).flatten()
                U[i] = (torch.pinverse(self.mu1 * CCT + self.mu3 * torch.eye(n=CCT.shape[0], device=device)) @ (self.mu1 * C @ (self.y_vec_cuda - s_vec) + self.mu3 * j_vec - z_vec)).reshape(-1,
                                                                                                                                                                                              self.R).cpu().numpy()
                # y_vec = tl.tensor_to_vec(self.y)
                # s_vec = tl.tensor_to_vec(S)
                # j_vec = tl.tensor_to_vec(J[i])
                # z_vec = tl.tensor_to_vec(Z[i])
                # CCT = C @ C.T
                # U[i] = (np.linalg.pinv(self.mu1 * CCT + self.mu3 * np.eye(CCT.shape[0])) @ (self.mu1 * C @ (y_vec - s_vec) + self.mu3 * j_vec - z_vec)).reshape(-1, self.R)

            for i in range(len(self.dims[self.L:])):
                D = []
                for r in range(self.R):
                    dr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != self.L + i]
                    dr = tl.cp_to_tensor((None, dr))
                    dr = ttt(self.x, dr, self.L, self.dims)
                    D.append(tl.tensor_to_vec(dr).reshape(-1, 1))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                D = np.concatenate(D, axis=1)
                D = torch.tensor(D, device=device)
                DTD = D.T @ D
                ym = self.yms_cuda[i + 1].T
                sm = torch.tensor(tl.unfold(S, mode=i + 1).T, device=device)
                J_cuda = torch.tensor(J[self.L + i], device=device)
                Z_cuda = torch.tensor(Z[self.L + i], device=device)
                U[self.L + i] = (torch.pinverse(self.mu1 * DTD + self.mu3 * torch.eye(n=DTD.shape[0], device=device)) @ (self.mu1 * D.T @ (ym - sm) + self.mu3 * J_cuda.T - Z_cuda.T)).T.cpu().numpy()
                # DTD = D.T @ D
                # ym = tl.unfold(self.y, mode=i + 1).T
                # sm = tl.unfold(S, mode=i + 1).T
                # U[self.L + i] = (np.linalg.pinv(self.mu1 * DTD + self.mu3 * np.eye(DTD.shape[0])) @ (self.mu1 * D.T @ (ym - sm) + self.mu3 * J[self.L + i].T - Z[self.L + i].T)).T

            for i in range(len(Z)):
                Z[i] += self.mu3 * (U[i] - J[i])

            B = tl.cp_to_tensor((None, U))
            J_ = tl.cp_to_tensor((None, J))
            y_pre = ttt(self.x, B, self.L, self.dims)
            S = soft_thresholding(self.y - y_pre, self.mu2 / self.mu1)
            rpe = (np.linalg.norm(self.y - y_pre)) / np.linalg.norm(self.y)
            norm_s = tl.norm(Js[-1] - J_, 2) ** 2  # 衡量变量更新幅度  U
            norm_r = tl.norm(Bs[-1] - B, 2) ** 2 #  衡量变量一致性。
            sparsity = (S.size - np.count_nonzero(S)) / S.size
            AIC = aic(self.y, y_pre, S, self.R, B)

            #need to uncomment when normal use
            if norm_r > 2 * norm_s:
                self.mu3 *= 2
            elif norm_s > 2 * norm_r:
                self.mu3 /= 2

            if verbose and itr % 10 == 0:
                msg = 'itr={}, rpe={:.4f}, norm_s={:.4f}, norm_r={:.4f}, sparsity={:.2f}'.format(itr + 1, rpe, norm_s, norm_r, sparsity)
                print(msg)

            if len(results) > 1 and abs(rpe - results[-1][0]) < 1e-4:
                break
            if itr > self.max_itr or (norm_r < self.tol and norm_s < self.tol):
                break
            if norm_r < math.inf:
                results.append((rpe, B, norm_r, AIC, S))

            Bs.append(B)
            Js.append(J_)

            itr += 1

        if verbose:
            print('======================')

        # results = sorted(results, key=lambda x: x[0])
        return results[-1][0], results[-1][1], results[-1][3], results[-1][4]

class RPCA_tucker:
    def __init__(self, **params):
        self.params = params
        self.reg = params['reg'] if 'reg' in params else 1.5e-1
        self.name = 'rpca'

    def fit(self, verbose=False):
        if verbose:
            print('======================')
            print('        RPCA          ')
            print('======================')

        y, S = robust_pca(self.params['y'], reg_E=self.reg, verbose=verbose)


        if verbose:
            print('sparsity={:.2f}'.format((S.size - np.count_nonzero(S)) / S.size))

        self.params['y'] = y
        tot = TOT_tucker(**self.params)
        return tot.fit(verbose=verbose)

class ROTR_fist_y:
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
        self.max_iter = params['ROTR_max_iter']
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
            # Step 1: 更新预测变量异常 Sx (ISTA)
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
                # 维度能匹配上吗？能匹配上，输出空间的维度是O1 到M  输入空间的维度是：P1 到PL
                # 这里能够实现维度计算回输入空间


                # 更新 Sx
                #Sx_temp = self.Sx + step_size_Sx * Gradient_Step  # 梯度更新方向 取负号，甚至都不影响结果？
                # 似乎应该是减。。。梯度下降就是该减。。。
                Sx_temp = self.Sx - step_size_Sx * Gradient_Step
                # 你就没有Sx的异常值
                self.Sx = soft_thresholding(Sx_temp, self.lambda_x * step_size_Sx)

            # ==========================================
            # Step 2: 更新响应异常 Sy (Soft Thresholding)
            # ==========================================
            X_clean = X - self.Sx
            Y_pred = self._contract_product(X_clean, B, L)
            Residual_Sy = Y - Y_pred
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
            b_vec = RHS_G.ravel() # ravel() 列向量化
            dim_G = np.prod(self.ranks)

            def matvec_G(v): # matvec_G 作用是给输入g，计算输出Ag
                G_curr = v.reshape(self.ranks)
                axes_X = list(range(1, L + 1))
                axes_G = list(range(L))
                Y_pred_core = np.tensordot(X_tilde, G_curr, axes=(axes_X, axes_G))
                delta_G = np.tensordot(X_tilde, Y_pred_core, axes=([0], [0]))
                return (delta_G + self.lambda_b * G_curr).ravel()
                # 函数看着没啥问题  
                # 函数的实现是参照公式12来的
            A_op = LinearOperator((dim_G, dim_G), matvec=matvec_G)
            G_opt_vec, info = cg(A_op, b_vec, x0=self.G.ravel(), rtol=1e-5)
            # 第一个传入A，第二个传b 。A（G），如果传不了A，就得传算子
            self.G = G_opt_vec.reshape(self.ranks)
            # 返回个结果出来就OK

            # ==========================================
            # Step 4: 更新因子矩阵 Uk (流形梯度下降)
            # ==========================================
            B_curr = self._reconstruct_B()
            Pred_Global = self._contract_product(X_clean, B_curr, L)
            Diff = Pred_Global - Y_clean
            Grad_B = np.tensordot(X_clean, Diff, axes=([0], [0]))
            # 为啥这梯度只有1个维度收缩？确实只有1个维度

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




class RPCA_Double:
    def __init__(self, **params):
        self.params = params
        default_reg = params.get('reg', 1.5e-1)
        self.reg_x = params.get('reg_x', default_reg)
        self.reg_y = params.get('reg_y', default_reg)
        self.name = 'rpca_double'

    def fit(self, verbose=False):
        if verbose:
            print('========================================')
            print('   Double-sided RPCA (Benchmark)        ')
            print('========================================')

        if verbose: print(f"Runing RPCA on Y (reg={self.reg_y})...")
        y_clean, S_y = robust_pca(self.params['y'], reg_E=self.reg_y, verbose=False)


        if verbose: print(f"Runing RPCA on X (reg={self.reg_x})...")
        x_clean, S_x = robust_pca(self.params['x'], reg_E=self.reg_x, verbose=False)

        if verbose:
            sparsity_sy = (S_y.size - np.count_nonzero(S_y)) / S_y.size
            sparsity_sx = (S_x.size - np.count_nonzero(S_x)) / S_x.size
            print(f'Done. Sparsity(Sy)={sparsity_sy:.2%}, Sparsity(Sx)={sparsity_sx:.2%}')

        self.params['x'] = x_clean
        self.params['y'] = y_clean
        

        if verbose: print("Fitting TOT model on cleaned data...")
        tot = TOT(**self.params)
        

        return tot.fit(verbose=verbose)



