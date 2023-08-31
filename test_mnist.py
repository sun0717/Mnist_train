from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn

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
# 加载训练好的模型
model = LeNet()
model.load_state_dict(torch.load('lenet_mnist.pth'))
model.cuda()

print(model)
# 读取图像并显示
img = Image.open('input.jpg')

# 使用和训练时相同的预处理(数据增广除外)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.3081, ))])
data = transform(img).view(1, 1, 28, 28).cuda()

# 将数据送入模型，得到推理结果
pred = model(data)
pred = F.softmax(pred, dim=1)
for label, prob in enumerate(pred[0].cpu().tolist()):
    print(f'Predicted probability of {label} is: {prob:06f}')
