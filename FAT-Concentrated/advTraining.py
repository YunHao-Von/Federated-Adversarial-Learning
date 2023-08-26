import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *
import torch.optim as optim
import copy




def kl_adv(model,x_natural,cfgs,dataset,marloss):
    """Use the kl loss to generated the adversarial exmpales."""
    epsilon, step_size = cfgs['epsilon'],cfgs['step_size']
    num_steps = 40 if dataset == "mnist" else 10
    criterion_kl = nn.KLDivLoss(size_average=False)  
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
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

def client_train(idx,args,model,cls_num_list,train_dataset,net_dataidx_map):
    args['cls_num_list'] = cls_num_list[idx]
    train_data_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset,net_dataidx_map[idx]),batch_size=args['local_bs'],shuffle=True,num_workers=4)  # 根据第i个client上的数据制作dataloader
    model.train()
    if args['dataset'] == 'cifar10':
        weight_decay = 2e-3
    else:
        weight_decay = 2e-4
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],momentum=0.9,weight_decay=weight_decay)  # 生成优化器
    criterion = maxMarginLoss(cls_num_list = args['cls_num_list']).cuda()  # 生成损失函数
    criterion_kl = maxMarginLoss_kl(cls_num_list = args['cls_num_list']).cuda()  # 生成kl损失函数s
    for iter in range(args['local_ep']):  # 进入客户端执行epoch
        for batch_idx, (images, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()  # 清空优化器中的梯度
            data, target = images.cuda(), labels.cuda()  # 将数据转移到GPU中
            x_adv = kl_adv(model,data,args['cfgs'],args['dataset'],criterion_kl)
            model.train()
            output = model(x_adv)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
    return model.state_dict()


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def _pgd_whitebox(model, X, y, cfgs, dataset):
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



def eval_adv_test_whitebox(model,test_loader,cfgs,dataset):
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data,requires_grad=True),Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, cfgs, dataset)
        robust_err_total += err_robust
        natural_err_total += err_natural
    return natural_err_total, robust_err_total