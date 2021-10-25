from __future__ import print_function, division, absolute_import, unicode_literals

import math
import numpy as np
import torch

from Regularizers.RegularizerClass import RegularizerClass


def L(x: torch.Tensor):
    """
    :rtype: y -> shape = (m, n)
    :type x -> shape = (2, m, n) where p = x[0] and q = x[1].
    """
    y = x.clone()
    # print("x.shape: ", x.shape,"y.shape: ", y.shape)
    y[0, 1:, :] = x[0, 1:, :] - x[0, :-1, :]
    y[1, :, 1:] = x[1, :, 1:] - x[1, :, :-1]
    y = y[0, :, :] + y[1, :, :]
    return y


def L_T(x: torch.Tensor):
    """
    :rtype: y -> shape = (2, m, n) where p = y[0] and q = y[1].
    :type x -> shape = (m, n)
    """
    m, n = x.shape
    y = torch.zeros(2, m, n)
    if x.is_cuda:
        y = y.cuda()
    y[0, :-1, :] = x[:-1, :] - x[1:, :]
    y[1, :, :-1] = x[:, :-1] - x[:, 1:]
    return y


def Pc(x: torch.Tensor, C):
    x[x < C[0]] = C[0]
    x[x > C[1]] = C[1]
    return x


def Pp(x: torch.Tensor):
    bottom = torch.sqrt(x[0, :, :] ** 2 + x[1, :, :] ** 2)
    bottom[bottom < 1] = 1
    return x / bottom


class TVGPUClass(RegularizerClass):
    def __init__(self, recon_shape, Lambda, N=20):
        """
        :param Lambda: regularization parameters
        :param N: Number of iterations
        """
        super().__init__()

        self.Lambda = Lambda
        self.N      = N
        self.recon_shape = recon_shape

    def init(self, **kwargs):
        p = np.zeros((2, self.recon_shape[0], self.recon_shape[1]))
        return p

    def eval(self, x,  **kwargs):
        y = L_T(x)
        y = torch.sum(torch.sqrt(y[0, :, :] ** 2 + y[1, :, :] ** 2))
        y = self.Lambda * y
        return y

    def prox(self, x, step, pin):
        # print("x.shape: ",x.shape)
        # if not torch.is_tensor(x):
        #     x = torch.from_numpy(x).float()
        # if not torch.is_tensor(pin):
        #     pin = torch.from_numpy(pin).float()

        z, pout = self.fit(x, Lambda=self.Lambda*step, p=pin, N=self.N)
        return z, pout

    @staticmethod
    def fit(b, Lambda, p, N, C=(-float('inf'), float('inf')), verbose=False):
        """
        :param b: observed image
        """
        m, n = b.shape
        pq = p  # p and q are concat together.
        rs = pq  # r and s are concat together.
        if b.is_cuda:
            pq = pq.cuda()
            rs = rs.cuda()
        t = 1
        iter_ = tqdm(range(N), desc='TVDenoiser') if verbose else range(N)
        for K in iter_:
            tLast  = t
            pqLast = pq.clone()
            pq = Pp(rs + (1 / (8 * Lambda)) * L_T(Pc(b - Lambda * L(rs), C)))
            t  = (1 + math.sqrt(1 + 4 * (t ** 2))) / 2
            rs = pq + (tLast - 1) / t * (pq - pqLast)
        xStart = Pc(b - Lambda * L(pq), C)
        del rs
        return xStart, pq

