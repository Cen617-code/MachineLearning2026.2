import torch
import torch.nn as nn
import torch.optim as optim

class MicroMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(3, 2)
        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = MicroMLP()

X_test = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
prediction = model(X_test)
print("网络引擎随机打出的预知输出:", prediction)

y_true = torch.tensor([[5.0]], dtype=torch.float32)
# 1. 定义工业级官方误差标尺 (MSE Loss)
criterion = nn.MSELoss()
# 2. 接管这个神经网络盒子里所有的随机参数旋钮 (model.parameters())，约定步长 lr 为 0.05
optimizer = optim.SGD(model.parameters(), lr = 0.05)

print("\n--- 开始训练 ---")
# 3. 开启神圣的 Epoch 迭代闭环
for epoch in range(20):
    # a. 正向预测
    prediction = model(X_test)
    # b. 拿去跟真实传感器值比对标尺，计算物理差距
    loss = criterion(prediction, y_true)
    print(f"Epoch {epoch}, Loss:{loss.item():.4f}")
    optimizer.zero_grad()# 板斧1：清空中枢旧梯度，防止污染打架
    loss.backward()# 板斧2：全自动链路反向查杀，算满全图所有的微分偏导数
    optimizer.step()# 板斧3：让包工头去按算好的斜率，强行拧动底盘所有的 W 和 B 旋钮！

print("\n--- 训练结束，开始闭卷考试 ---")
X_val = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)

with torch.no_grad():
    val_prediction = model(X_val)
    print("模型面对全新数据的盲猜输出:", val_prediction)