import numpy as np
from torchvision import datasets,transforms

def load_data(dataset):
    """load the datasets"""
    data_dir = "Dataset"
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir,train=True,transform=transforms.Compose([transforms.ToTensor()]),download=True)  # load the trainset for mnist
        test_dataset = datasets.MNIST(data_dir,train=False,transform=transforms.Compose([transforms.ToTensor()]),download=True)  # load the testset for mnist
        X_train, y_train = train_dataset.data, train_dataset.targets
        X_test, y_test = test_dataset.data, test_dataset.targets
    
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir,train=True,transform=transforms.Compose([transforms.ToTensor()]),download=True)  # load the trainset for fashionmnist
        test_dataset = datasets.FashionMNIST(data_dir,train=False,transform=transforms.Compose([transforms.ToTensor()]),download=True)  # load the testset for fashionmnist
        X_train, y_train = train_dataset.data, train_dataset.targets
        X_test, y_test = test_dataset.data, test_dataset.targets
        
    elif dataset == "cifar10":
        data_dir = "Dataset/Cifar10"
        train_dataset = datasets.CIFAR10(data_dir,train=True,transform=transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]),download=True)  # load the trainset for cifar10
        test_dataset = datasets.CIFAR10(data_dir,train=False,transform=transforms.Compose([transforms.ToTensor()]),download=True)  # load the testset for cifar10
        X_train, y_train = train_dataset.data, train_dataset.targets
        X_test, y_test = test_dataset.data, test_dataset.targets
    else:
        print("Invalid Dataset!")
        raise NotImplementedError
    
    if 'cifar10' in dataset:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        X_train = X_train.data.numpy()
        y_train = y_train.data.numpy()
        X_test = X_test.data.numpy()
        y_test = y_test.data.numpy()
    
    return X_train,y_train,X_test,y_test,train_dataset,test_dataset