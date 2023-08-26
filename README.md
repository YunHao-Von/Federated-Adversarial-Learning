# FAT_Distributed

## Introducion
This project is based on two hosts (each with a Titan graphics card) to produce distributed federated adversaral training implementation.  
We also provide an implementation of federated learning for single-machine simulation, see the FAT-Concentrated folder.
Adversarial training is introduced in federated learning to enhance model robustness.
The adversarial training code refers to the NIPS 2022 paper:  
#### [CalFAT: Calibrated Federated Adversarial Training with Label Skewness.](https://github.com/cc233/CalFAT)
Based on Pytorch and NCCL implementations.
Please refer to the construction process of this project:
[Zhihu](https://zhuanlan.zhihu.com/p/652537621)

### Direction

```
cd FAT-Distributed
```
Machine1:
```
torchrun --nproc_per_node=1 --nnode=2 --node_rank=0 --master_addr="192.168.1.100" --master_port=29690 main.py
```  
Machine2:  
```
torchrun --nproc_per_node=1 --nnode=2 --node_rank=1 --master_addr="192.168.1.100" --master_port=29690 main.py
```
Parameter description:
+ nproc_per_node: The number of GPUs on the current machine.
+ nnode: The number of compute nodes.
+ master_addr: The IP address of the master machine.
+ master_port: The communication port of the master machine.


## 介绍  
本项目是基于两台主机(各有一张Titan显卡)，来制作的分布式联邦对抗训练实现。  
同时我们还提供了关于单机模拟联邦学习的实现，参见FAT-Concentrated文件夹。
在联邦学习中引入了对抗训练来增强模型鲁棒性。  
对抗训练代码参考了NIPS 2022 论文:
#### [CalFAT: Calibrated Federated Adversarial Training with Label Skewness.](https://github.com/cc233/CalFAT)
基于Pytorch和NCCL实现。
本项目搭建过程参见:  
[知乎](https://zhuanlan.zhihu.com/p/652537621)
### 使用说明

```
cd FAT-Distributed
```
机器1:
```
torchrun --nproc_per_node=1 --nnode=2 --node_rank=0 --master_addr="192.168.1.100" --master_port=29690 main.py
```  
机器2:  
```
torchrun --nproc_per_node=1 --nnode=2 --node_rank=1 --master_addr="192.168.1.100" --master_port=29690 main.py
```
参数解释:  
+ nproc_per_node:当前机器的GPU数量。  
+ nnode:计算节点数量。  
+ master_addr:主机器IP地址。  
+ master_port:主机器通信端口。  
