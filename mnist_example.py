# 训练模型
import torch
import torch.nn as nn
import torch.nn.functional as F

# create LeNet class
class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x) # C1: Nx6x28x28  N-->一批数据样本的个数batch_size
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # S2: Nx6x14x14
        x = self.conv2(x) # C3: Nx16x10x10
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # S4: Nx16x5x5
        x = x.flatten(start_dim=1) # Nx400
        x = self.fc1(x) # F5: Nx120
        x = F.relu(x)
        x = self.fc2(x) # F6: Nx84
        x = F.relu(x)

        output = self.fc3(x) # Nx10
        return output
model = LeNet() # 构造LeNet实例
model.cuda() # 使用GPU
print(model)

# 第三步，使用pytorch提供的SGD优化器
# torch.optim 提供了各种优化算法的实现
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adadelta(model.parameters(), lr = 1)
# step_size 在每一轮训练之后，对学习率做一个衰减调整
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

print(scheduler.get_last_lr())
# 准备 MNIST 数据集
# 在 PyTorch 的辅助工具箱 Torchvision 中，提供了包括MNIST在内的多个常用数据集的接口，以及常用的数据预处理和增广算法
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

# 使用 torchvision MNIST 数据接口构建 Dataset
# 数据预处理，toTensor()。把原始图像类型变成模型可以处理的Tensor类型
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.3081, ))]
)

train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transform)

test_dataset = datasets.MNIST('./data',
                              train=False,
                              download=True,
                              transform=transform)
# 构建 DataLoader
train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          shuffle=True,
                          pin_memory=True)
test_loader = DataLoader(test_dataset,
                         batch_size=1000,
                         shuffle=False,
                         pin_memory=True)

print('Length of train_dataset:',len(train_dataset))
print('Length of test_dataset:',len(test_dataset))
print('Length of train_loader:',len(train_loader))
print('Length of test_loader:',len(test_loader))



# 训练一个 epoch 的过程
def train_epoch(model, loader, optimizer, epoch):
    # 将模型设为训练模式
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        # 从数据集中取出 1 个 batch
        data, target = data.cuda(), target.cuda()
        # 清除之前的梯度信息
        optimizer.zero_grad()
        # 前向传播 forward propagation
        output = model(data)
        # 计算损失函数
        loss = F.cross_entropy(output, target)
        # 反向传播 backward propagation
        loss.backward()
        # 优化器更新模型参数
        optimizer.step()

        # 显示训练信息
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']
            ))
# 测试模型的过程
def test(model, test_loader):
    # 将模型设为测试模式
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # 统计损失函数值和正确分类样本数
            test_loss += F.cross_entropy(
                output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True) # get the index of the prediction
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    # 显示平均损失函数值和分类正确率信息
    print(
        '\nTest set: Average loss: {:.4f}, Accuary: {}/{} ({:.0f}%\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

# 进行多轮训练，每轮训练后测试模型精度
total_epoch = 14

for epoch in range(total_epoch):
    train_epoch(model, train_loader, optimizer, epoch)
    test(model, test_loader)
    # 更新学习率
    scheduler.step()
            
torch.save(model.state_dict(), './lenet_mnist.pth')