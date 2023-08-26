import os
import torch
import torchvision
import torch.distributed as dist
from models import *
from datasets import *
from utils import *
from adversary import *
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torch.optim as optim
import datetime


def FATTrain(dataset,model):
    # 读取环境变量配置信息
    lr = 0.001
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    local_epoch = 4
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
    ddp_model = DDP(model,[local_rank])
    user_idx = local_rank + rank*1
    train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts = process_data(2, dataset)  # 获得切分好的数据集
    cls_num_list = get_cls_num_list(train_data_cls_counts)[user_idx]  # 将字典转化为列表
    train_data_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset,net_dataidx_map[user_idx]),batch_size=64,shuffle=True)
    weight_decay = 2e-3 if dataset == 'cifar10' else 2e-4
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr,momentum=0.9,weight_decay=weight_decay)  # 生成优化器
    criterion = maxMarginLoss(cls_num_list).cuda(local_rank)  # 生成损失函数
    criterion_kl = maxMarginLoss_kl(cls_num_list).cuda(local_rank)  # 生成kl损失函数s
    for _ in range(local_epoch):
        for batch_idx, (images, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()  # 清空优化器中的梯度
            data, target = images.cuda(local_rank), labels.cuda(local_rank)  # 将数据转移到GPU中
            x_adv = kl_adv(ddp_model,data,dataset,criterion_kl,local_rank)
            ddp_model.train()
            output = ddp_model(x_adv)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
    return ddp_model


def run():
    # os.environ["MASTER_ADDR"] = "192.168.1.100"
    # os.environ["MASTER_PORT"] = "29500"
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    dataset_name = 'mnist'
    world_size = env_dict['WORLD_SIZE']
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts = process_data(2, dataset_name)  # 获得切分好的数据集
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False)  # 制作DataLoader
    total = len(test_dataset)  # 计算测试集的长度
    bst_cln_acc = -1
    bst_rob_acc = -1
    if dataset_name == "cifar10":
        model = ModelCifar().cuda(local_rank)
    else:
        model = ModelMnist().cuda(local_rank)
        
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5))
    for epoch in tqdm(range(10)):
        local_model = FATTrain(dataset_name,model)
        local_params = [param.data.flatten() for param in local_model.parameters()]
        requires_grad_list = [param.requires_grad for param in local_model.parameters()]
        flatten_params = torch.cat(local_params)
        gathered_params = [torch.zeros_like(flatten_params) for _ in range(int(world_size))]
        dist.all_gather(gathered_params,flatten_params)
        gathered_params = torch.stack(gathered_params)
        mean_gathered_params = torch.mean(gathered_params, dim=0)
        global_model = model_reload(local_model,mean_gathered_params,requires_grad_list)
        natural_err_total, robust_err_total = eval_adv_test_whitebox(global_model, test_loader, dataset_name)
        cln_acc, rob_acc = (total - natural_err_total) * 1.0 / total, (total - robust_err_total) * 1.0 / total
        print("Epoch:{},\t cln_acc:{}, \t rob_acc:{}".format(epoch, cln_acc, rob_acc))
        print("Epoch:{},\t bst_cln_acc:{}, \t bst_rob_acc:{}".format(epoch, bst_cln_acc , bst_rob_acc))
        save_checkpoint({
            'state_dict': global_model.state_dict(),
            'epoch': epoch,
        }, cln_acc > bst_cln_acc, "ModelSave/"+str(dataset_name)+"/BestAcc.pth")
        save_checkpoint({
        'state_dict': global_model.state_dict(),
        'epoch': epoch,
        }, rob_acc > bst_rob_acc, "ModelSave/"+str(dataset_name)+"/BestRob.pth")
        model = global_model
        bst_cln_acc = max(bst_cln_acc, cln_acc)
        bst_rob_acc = max(bst_rob_acc, rob_acc)

    dist.destroy_process_group()


if __name__ == "__main__":
    setup_seed(2023)
    print(1)
    run()
