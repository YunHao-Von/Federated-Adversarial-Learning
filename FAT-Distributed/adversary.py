import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *
import torch.optim as optim
import copy

def kl_adv(model,x_natural,dataset,marloss,local_rank):
    """Use the kl loss to generated the adversarial exmpales."""
    sets = {"mnist": dict(step_size=0.01, epsilon=0.3),
        "fmnist": dict(step_size=0.01, epsilon=0.3),
        "cifar10": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        "svhn": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        "cifar100": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        }
    cfgs = sets[dataset]
    epsilon, step_size = cfgs['epsilon'],cfgs['step_size']
    num_steps = 40 if dataset == "mnist" else 10
    criterion_kl = nn.KLDivLoss(size_average=False)  
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda(local_rank).detach()
    for __ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(
                F.log_softmax(marloss(model(x_adv)),dim=1),
                F.softmax(marloss(model(x_natural)),dim=1)
            )
        grad = torch.autograd.grad(loss_kl,[x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv

def model_reload(new_model,flatten_tensor,requires_grad_list):
    param_index = 0
    for param, requires_grad in zip(new_model.parameters(), requires_grad_list):
        param_shape = param.data.shape
        param.data = flatten_tensor[param_index:param_index+param.numel()].reshape(param_shape)
        param.requires_grad = requires_grad
        param_index += param.numel()
    return new_model

def eval_adv_test_whitebox(model,test_loader,dataset):
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data,requires_grad=True),Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, dataset)
        robust_err_total += err_robust
        natural_err_total += err_natural
    return natural_err_total, robust_err_total

def _pgd_whitebox(model, X, y, dataset):
    sets = {"mnist": dict(step_size=0.01, epsilon=0.3),
        "fmnist": dict(step_size=0.01, epsilon=0.3),
        "cifar10": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        "svhn": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        "cifar100": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        }
    cfgs = sets[dataset]
    epsilon, step_size = cfgs["epsilon"], cfgs["step_size"]
    num_steps = 40 if dataset == "mnist" else 20
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()  # 生成噪声
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)  # 生成对抗样本

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd