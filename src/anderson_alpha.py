# -*- coding: utf-8 -*-

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RAA(object):
    def __init__(self, num_critics, use_restart, reg=0.1):
        self.size = num_critics
        self.reg = reg                 # regularization
        self.use_restart = use_restart
        self.count = 0
        self.interval = 5000
        self.errors = torch.zeros(self.interval).to(device)
        self.opt_error = torch.tensor(0.).to(device)

    def calculate(self, Qs, F_Qs):
        Qs = Qs.t()
        F_Qs = F_Qs.t()
        delta_Qs = F_Qs - Qs
        cur_size = Qs.size(1)

        del_mat = delta_Qs.t().mm(delta_Qs)
        alpha = del_mat / torch.abs(torch.mean(del_mat))
        alpha += self.reg * torch.eye(cur_size).to(device)

        alpha = torch.sum(alpha.inverse(), 1)
        alpha = torch.unsqueeze(alpha / torch.sum(alpha), 1)

        # restart checking
        self.count += 1
        self.errors[self.count % self.interval] = torch.mean(torch.pow(delta_Qs[:, -1], 2)).detach()

        if self.use_restart:
            if self.count % self.interval == 0:
                error = torch.mean(self.errors)
                if self.count == self.interval:
                    self.opt_error = error
                else:
                    self.opt_error = torch.min(self.opt_error, error)

                if (self.count > self.interval and error > self.opt_error) or self.count > 100000:
                    print(error, self.opt_error)
                    restart = True
                    self.count = 0
                else:
                    restart = False
            else:
                restart = False
        else:
            restart = False

        return alpha, restart

    def calculate_newReg(self, Qs, F_Qs): # Qs/F_Qs: m * |S*A|
        # (1) delta matrix: m by m
        Qs = Qs.t()
        F_Qs = F_Qs.t()
        delta_Qs = F_Qs - Qs  # compute delta matrix
        cur_size = Qs.size(1) # m
        # (2) regularization matrix
        g = delta_Qs.detach()
        Y = g[:,1:] - g[:,:cur_size-1] # (N, m-1)
        S = F_Qs[:,1:] - F_Qs[:,:cur_size-1] # v^{k+1}=\sum alpha_i * F(v^k)
        delta_k = delta_Qs[:,cur_size - 1] # N*1, current target
        # (3) solve gamma
        temp = Y.t().mm(Y)
        temp = temp / torch.abs(torch.mean(temp))
        temp += self.reg * (torch.norm(S, p='fro')**2 + torch.norm(Y, p='fro')**2) * torch.eye(cur_size-1).to(device)
        gamma = temp.inverse().mm(Y.t().mm(torch.unsqueeze(delta_k,1)))
        # (4) transform from gamma to alpha
        m = gamma.shape[0]

        gamma = gamma.squeeze()
        if m == 1:
            gamma = gamma.unsqueeze(0)

        alpha = torch.zeros(m+1)
        alpha[0] = gamma[0]
        alpha[1:m] = gamma[1:m] - gamma[0:m-1]
        alpha[m] = 1-gamma[m-1]
        alpha = torch.unsqueeze(alpha, 1)

        # restart checking
        self.count += 1
        self.errors[self.count % self.interval] = torch.mean(torch.pow(delta_Qs[:, -1], 2)).detach()

        if self.use_restart:
            if self.count % self.interval == 0:
                error = torch.mean(self.errors)
                if self.count == self.interval:
                    self.opt_error = error
                else:
                    self.opt_error = torch.min(self.opt_error, error)

                if (self.count > self.interval and error > self.opt_error) or self.count > 100000:
                    print(error, self.opt_error)
                    restart = True
                    self.count = 0
                else:
                    restart = False
            else:
                restart = False
        else:
            restart = False
        return alpha.to(device), restart