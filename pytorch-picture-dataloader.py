import numpy as np
import random
import torch
import torch.utils.data
import torch.autograd.variable as Variable
import torchvision
import torchvision.models
import torchvision.transforms
import torch.nn as nn
import matplotlib.pyplot as plt 
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# 数据集读取
class Dataset:
    def __init__(self, config,train_transforms=None,test_transforms=None):
        self.config = config  # 保存参数的字典

        # 设定数据集保存地址
        dataset_rootdir = './Dataset'
        self.dataset_dir = dataset_rootdir + '/' + config['dataset_name']

        # 配置数据集预处理方法
        if train_transforms is None:
            self._train_transforms = [torchvision.transforms.ToTensor()]
        else:
            self._train_transforms = train_transforms
        if test_transforms is None:
            self._test_transforms = [torchvision.transforms.ToTensor()]
        else:
            self._test_transforms = test_transforms

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

        # 读取数据集的Dataset对象
        self.dataset_name = config['dataset_name']
        self.train_dataset, self.test_dataset = self.get_datasets()

    def get_datasets(self):
        # 提取dataset
        train_dataset = getattr(torchvision.datasets, self.dataset_name)(
            self.dataset_dir,
            train=True,
            transform=self.train_transform,
            download=True)
        test_dataset = getattr(torchvision.datasets, self.dataset_name)(
            self.dataset_dir,
            train=False,
            transform=self.test_transform,
            download=True)
        print('train dataset size:' + str(len(train_dataset)) + ' test dataset size:' + str(len(test_dataset)))
        return train_dataset, test_dataset

    def get_dataloader(self):
        # 用Dataser对象去创建Dataloader对象
        # Dataloader是把数据集按batch_size分割的迭代器
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,  # 是否打乱数据顺序
            drop_last=True,  # 最后一个batch数据数量可能会小于batch_size，为True则丢弃这一部分
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, test_loader

    def _get_train_transform(self):
        # 对训练数据进行一些预处理操作
        # 这里返回的是一个transforms对象，可以查阅资料自行添加一些其他的处理
        #self._train_transforms.append(torchvision.transforms.RandomCrop(32, padding=4))
        #self._train_transforms.append(torchvision.transforms.RandomHorizontalFlip())
        #self._train_transforms.append(torchvision.transforms.ToTensor())
        #self._train_transforms.append(torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        return torchvision.transforms.Compose(self._train_transforms)

    def _get_test_transform(self):
        # 对测试集数据的预处理
        # 这里返回的是一个transforms对象，可以查阅资料自行添加一些其他的处理
        #self._test_transforms.append(torchvision.transforms.ToTensor())
        #self._test_transforms.append(torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        return torchvision.transforms.Compose(self._test_transforms)

    def show_img(self, mode='train', index=0):
        # mode: 显示图片在训练集还是测试集
        # index: 显示第几张图片
        if mode == 'train':  # 判断是显示训练集中的还是测试集中的
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
        if index < 0 or index >= len(dataset):  # 判断index有没有越界
            print('invaild index')
            return

        # 图片显示
        pylab.rcParams['figure.figsize'] = (2, 2)  # 设定显示大小
        img = dataset[index][0].numpy()  # tensor转换成numpy.array
        if img.shape[0] == 1:  # 单通道灰度图
            plt.imshow(img.reshape(img.shape[1], img.shape[2]), cmap='gray')
        else:  # RGB图片
            img = img.transpose(1, 2, 0)  # 数据集中数据格式是(C,W,H),显示时要转换成(W,H,C)
            plt.imshow(img)

        plt.show()  # 显示图片
        label = dataset[index][1]
        if config['dataset_name'] not in ['CIFAR10', 'CIFAR100']:
            label = label.item()
        print('label:' + str(label))  # 输出图片的index和标签信息


def train(model, dataset, config):  # 训练部分
    # 设置device,torch.cuda.is_available()会返回GPU是否可用，若可用设定device为config['gpu']号GPU，否则为cpu
    device = torch.device(('cuda:' + config['gpu']) if torch.cuda.is_available() else "cpu")
    # 设置random seed
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    random.seed(config['random_seed'])
    # 把模型加载到device上
    model.to(device)

    # 设置loss和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(), config['lr'], momentum=config['momentum'],
                                weight_decay=config['weight_decay'])  # 设置优化器为随机梯度下降
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.3) 
    print('开始训练')
    for epoch in range(config['epochs']):  # 多个epochs循环
        # 模型设定到训练模式
        model.train()
        scheduler.step()
        train_loader, test_loader = dataset.get_dataloader()  # 取dataloader
        for i, (input, target) in enumerate(train_loader):  # 迭代train_loader中的batch
            optimizer.zero_grad()  # 每次迭代要先把梯度清0
            input_var = Variable(input).float().to(device)  # 输入数据，注意这里数据的device要与模型一致
            target = torch.LongTensor(target)
            target_var = Variable(target).long().to(device)  # 标签数据
            output = model(input_var)  # 模型输出结果
            loss = criterion(output, target_var)  # 计算损失
            loss.backward()  # 梯度反向传递
            optimizer.step()  # 参数更新
           

            if i % 100 == 0:  # 每100个batch输出loss信息
                line = 'Epoch:' + str(epoch) + '/' + str(i) + ' loss:' + str(loss.item() / config['batch_size'])
                print(line)
        torch.save(model.state_dict(), './checkpoints_gyc/' + str(epoch) + '.pth')  # 一个epoch结束保存模型参数
        acc, wrong_samples = test(model, test_loader, config)  # 每个epoch进行一次测试
        print('Epoch:' + str(epoch) + ' acc:' + str(acc))  # 打印当前模型在测试集上准确率
    print('训练结束')
    return acc


def test(model, test_loader, config):  # 测试部分
    model.eval()  # 测试模式
    device = torch.device(('cuda:' + config['gpu']) if torch.cuda.is_available() else "cpu")  # 设置device
    T = 0  # 正确分类样本数
    count = 0  # 总样本数
    wrong_samples = []  # 保存错误样本的(index,预测标签,真实标签)
    with torch.no_grad():  # 测试时不需要计算梯度信息
        for i, (input, target) in enumerate(test_loader):  # 迭代测试集
            input_var = Variable(input).float().to(device)  # 输入数据
            count += 1  # 总样本数加1
            output = model(input_var)  # 模型输出
            soft_output = torch.softmax(output, dim=-1)  # softmax转换成概率值
            _, predicted = torch.max(soft_output.data, 1)  # 将概率最高的类作为预测类
            predicted = predicted.to('cpu').detach().numpy()
            label = target.to('cpu').detach().numpy()
            if predicted == label:  # 预测正确
                T += 1
            else:  # 预测错误
                wrong_samples.append((i, predicted[0], label[0]))
    acc = float(T) / count  # 计算正确率
    return acc, wrong_samples


def show_wrong_samples(dataset, wrong_samples, index=0):  # 显示错误样本
    pylab.rcParams['figure.figsize'] = (2, 2)  # 设定显示大小
    if index < 0 or index >= len(wrong_samples):  # 判断index是否越界
        print('invaild index')
        return
    sample = wrong_samples[index]  # 第index个错误样本
    index = sample[0]  # 错误样本在数据集中的索引
    pre_label = sample[1]  # 预测标签
    true_label = sample[2]  # 真实标签
    img = dataset[index][0].numpy()  # tensor转换成numpy.array
    if img.shape[0] == 1:  # 单通道灰度图
        plt.imshow(img.reshape(img.shape[1], img.shape[2]), cmap='gray')
    else:  # RGB图片
        img = img.transpose(1, 2, 0)  # 数据集中数据格式是(C,W,H),显示时要转换成(W,H,C)
        plt.imshow(img)
    plt.show()  # 显示图片
    print('index:' + str(index) + ' pre_label:' + str(pre_label) + ' true_label:' + str(true_label))  # 输出预测标签和真实标签
