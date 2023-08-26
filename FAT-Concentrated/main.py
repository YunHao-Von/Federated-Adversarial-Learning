
from utils import *
from tqdm import tqdm
import torch.optim as optim
from advTraining import *
import copy

args = {
    'partition':"dirichlet",
    'beta':0.4,
    'num_users':10,
    'dataset':'cifar10',
    'comment':'ModelSave',
    'model':'cnn',
    'save':'ce',
    'epochs':70,
    'local_bs':256,
    'lr':0.01,
    'local_ep':10,
}
sets = {"mnist": dict(step_size=0.01, epsilon=0.3),
        "fmnist": dict(step_size=0.01, epsilon=0.3),
        "cifar10": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        "svhn": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        "cifar100": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        }

cfgs = sets[args['dataset']]
args['cfgs'] = cfgs


setup_seed(2021)
train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts = process_data(args['num_users'], args['dataset'], args['partition'])  # 获得切分好的数据集
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False, num_workers = 4)  # 制作DataLoader
model = get_model(args['dataset'])  # 获取模型
description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"  # 生成表述语言
cls_num_list = get_cls_num_list(train_data_cls_counts,args)  # 将字典转化为列表
epochs = args['epochs']  # 设定epochs的总数
total = len(test_dataset)  # 计算测试集的长度
bst_cln_acc = -1
bst_rob_acc = -1
accurate = list()
robust = list()

for epoch in tqdm(range(epochs)):
    local_weights = []  # 客户端参数
    for idx in range(args['num_users']):
        client_weight = client_train(idx,args,model,cls_num_list,train_dataset,net_dataidx_map)
        local_weights.append(copy.deepcopy(client_weight))
    global_weight = average_weights(local_weights)
    global_model = model
    global_model.load_state_dict(global_weight)
    print('================================================================')
    natural_err_total, robust_err_total = eval_adv_test_whitebox(global_model, test_loader, cfgs, args['dataset'])
    cln_acc, rob_acc = (total - natural_err_total) * 1.0 / total, (total - robust_err_total) * 1.0 / total
    print("Epoch:{},\t cln_acc:{}, \t rob_acc:{}".format(epoch, cln_acc, rob_acc))
    print("Epoch:{},\t bst_cln_acc:{}, \t bst_rob_acc:{}".format(epoch, bst_cln_acc , bst_rob_acc))
    save_checkpoint({
            'state_dict': global_model.state_dict(),
            'epoch': epoch,
        }, cln_acc > bst_cln_acc, "ModelSave/"+str(args['dataset'])+"/BestAcc.pth")

    save_checkpoint({
        'state_dict': global_model.state_dict(),
        'epoch': epoch,
    }, rob_acc > bst_rob_acc, "ModelSave/"+str(args['dataset'])+"/BestRob.pth")
    bst_cln_acc = max(bst_cln_acc, cln_acc)
    bst_rob_acc = max(bst_rob_acc, rob_acc)
    accurate.append(bst_cln_acc)
    robust.append(bst_rob_acc)


save_checkpoint({
            'state_dict': global_model.state_dict(),
            'epoch': epoch,
        }, 1 > 0, "ModelSave/"+str(args['dataset'])+"/Epoch-{}-last.pth".format(epoch))


import matplotlib.pyplot as plt
accurate_plot = torch.tensor(accurate).cpu()
robust_plot = torch.tensor(robust).cpu()
plt.plot(accurate_plot,label = 'accurate')
robust_plot = torch.tensor(robust).cpu()
plt.plot(robust_plot,label = 'robust')
plt.title(args['dataset'])
plt.legend()


