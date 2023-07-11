from common import *
# https://github.com/nachiket273/lookahead_pytorch


import math
import torch
from torch.optim import Optimizer
from collections import defaultdict

class RAdam(Optimizer):
    r"""Implements RAdam algorithm.
    It has been proposed in `ON THE VARIANCE OF THE ADAPTIVE LEARNING
    RATE AND BEYOND(https://arxiv.org/pdf/1908.03265.pdf)`_.

    Arguments:
        params (iterable):      iterable of parameters to optimize or dicts defining
                                parameter groups
        lr (float, optional):   learning rate (default: 1e-3)
        betas (Tuple[float, float], optional):  coefficients used for computing
                                                running averages of gradient and
                                                its square (default: (0.9, 0.999))
        eps (float, optional):  term added to the denominator to improve
                                numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional):    whether to use the AMSGrad variant of this
                                        algorithm from the paper `On the Convergence
                                        of Adam and Beyond`_(default: False)

        sma_thresh:             simple moving average threshold.
                                Length till where the variance of adaptive lr is intracable.
                                Default: 4 (as per paper)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, sma_thresh=4):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(RAdam, self).__init__(params, defaults)

        self.radam_buffer = [[None, None, None] for ind in range(10)]
        self.sma_thresh = sma_thresh

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                old = p.data.float()

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                buffer = self.radam_buffer[int(state['step'] % 10)]

                if buffer[0] == state['step']:
                    sma_t, step_size = buffer[1], buffer[2]
                else:
                    sma_max_len = 2 / (1 - beta2) - 1
                    beta2_t = beta2 ** state['step']
                    sma_t = sma_max_len - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffer[0] = state['step']
                    buffer[1] = sma_t

                    if sma_t > self.sma_thresh:
                        rt = math.sqrt(
                            ((sma_t - 4) * (sma_t - 2) * sma_max_len) / ((sma_max_len - 4) * (sma_max_len - 2) * sma_t))
                        step_size = group['lr'] * rt * math.sqrt((1 - beta2_t)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffer[2] = step_size

                if group['weight_decay'] != 0:
                    p.data.add_(old, alpha=-group['weight_decay'] * group['lr'])

                if sma_t > self.sma_thresh:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value= -step_size, )
                else:
                    p.data.add_(exp_avg, alpha=-step_size)

        return loss


class Lookahead(Optimizer):
    r'''Implements Lookahead optimizer.

    It's been proposed in paper: Lookahead Optimizer: k steps forward, 1 step back
    (https://arxiv.org/pdf/1907.08610.pdf)

    Args:
        optimizer: The optimizer object used in inner loop for fast weight updates.
        alpha:     The learning rate for slow weight update.
                   Default: 0.5
        k:         Number of iterations of fast weights updates before updating slow
                   weights.
                   Default: 5

    Example:
        > optim = Lookahead(optimizer)
        > optim = Lookahead(optimizer, alpha=0.6, k=10)
    '''

    def __init__(self, optimizer, alpha=0.5, k=5):
        assert (0.0 <= alpha <= 1.0)
        assert (k >= 1)
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.k_counter = 0
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.slow_weights = [[param.clone().detach() for param in group['params']] for group in self.param_groups]
        self.defaults = {}

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.k_counter += 1
        if self.k_counter >= self.k:
            for group, slow_weight in zip(self.param_groups, self.slow_weights):
                for param, weight in zip(group['params'], slow_weight):
                    weight.data.add_((param.data - weight.data), alpha=self.alpha)
                    param.data.copy_(weight.data)
            self.k_counter = 0
        return loss

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'k': self.k,
            'k_counter': self.k_counter
        }

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)