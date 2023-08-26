import os
import torch.distributed as dist
from datasets import *
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.backends.cudnn as cudnn



def setup_seed(seed):
    """Setting the seed of random numbers"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministuc = True


def record_net_data_stats(y_train, net_dataidx_map):
    """Return a dict for the net_dataidx_map"""
    net_cls_counts = {}
    num_class = np.unique(y_train).shape[0]
    for net_i, dataidx in net_dataidx_map.items():  # label:sets
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        for i in range(num_class):
            if i in tmp.keys():
                continue
            else:
                tmp[i] = 1  # 5

        net_cls_counts[net_i] = tmp

    return net_cls_counts


def process_data(n_user,dataset,partition='iid',beta=0.4):
    """处理数据集使得其分布在联邦学习的客户端上

    Args:
        n_user (int): 客户端的数量
        dataset (String): 数据集的名称
        partitin (String): 数据集的切分方式
        beta (float, optional): 狄利克雷的参数,Defaults to 0.4.
    """
    n_parties = n_user  # 将数据分为n份
    X_train,y_train,X_test,y_test,train_dataset,test_dataset = load_data(dataset)  # 载入数据
    data_size = y_train.shape[0]  # 数据的量
    if partition == 'iid':  # 如果数据呈现独立同分布
        idxs = np.random.permutation(data_size)  # Generate a permutation for the list[0,...,data_size - 1]
        batch_idxs = np.array_split(idxs, n_parties)  # Split the permutation into n_parties 
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}  # Generate a dict for the generated lists.{i:[]}
    
    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(n_parties)]
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(  # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    else:
        raise Exception('Invalid Partition')
    
    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts

def get_cls_num_list(traindata_cls_counts):
    """Transfer the dict into the list"""
    cls_num_list = []
    num_class = 10
    for key, val in traindata_cls_counts.items():
        temp = [0] * num_class  
        for key_1, val_1 in val.items():
            temp[key_1] = val_1
        cls_num_list.append(temp)

    return cls_num_list

class DatasetSplit(Dataset):
    """Split the train_dataset into the idxs client."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image,label



class maxMarginLoss(nn.Module):
    def __init__(self, cls_num_list,weight = None,s=1):
        super(maxMarginLoss, self).__init__()
        self.m_list = torch.FloatTensor(cls_num_list).cuda()
        assert s > 0
        self.s = s
        self.weight = weight
    
    def forward(self,x,target):
        output = x + 0.1 * torch.log(self.m_list + 1e-7)  # 加上类别的信息
        return F.cross_entropy(self.s * output,target, weight=self.weight)


class maxMarginLoss_kl(nn.Module):
    def __init__(self, cls_num_list, weight=None, s=1):
        super(maxMarginLoss_kl, self).__init__()
        m_list = torch.FloatTensor(cls_num_list).cuda()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x):
        output = x + 0.1 * torch.log(self.m_list + 1e-7)
        return output

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)