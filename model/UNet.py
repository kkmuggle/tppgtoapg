import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, length=1024, n_channel=3):
        super(UNet, self).__init__()
        x = 64

        # 下采样
        self.enc1 = self.conv_block(n_channel, x)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = self.conv_block(x, x * 2)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = self.conv_block(x * 2, x * 4)
        self.pool3 = nn.MaxPool1d(2)

        self.enc4 = self.conv_block(x * 4, x * 8)
        self.pool4 = nn.MaxPool1d(2)

        self.enc5 = self.conv_block(x * 8, x * 16)

        # 上采样
        self.up6 = nn.ConvTranspose1d(x * 16, x * 8, 2, stride=2)
        self.dec6 = self.conv_block(x * 16, x * 8)

        self.up7 = nn.ConvTranspose1d(x * 8, x * 4, 2, stride=2)
        self.dec7 = self.conv_block(x * 8, x * 4)

        self.up8 = nn.ConvTranspose1d(x * 4, x * 2, 2, stride=2)
        self.dec8 = self.conv_block(x * 4, x * 2)

        self.up9 = nn.ConvTranspose1d(x * 2, x, 2, stride=2)
        self.dec9 = self.conv_block(x * 2, x)

        # 输出层
        self.out = nn.Conv1d(x, 1, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        # print(enc1.shape)
        pool1 = self.pool1(enc1)
        # print(pool1.shape)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)

        enc5 = self.enc5(pool4)

        up6 = self.up6(enc5)
        merge6 = torch.cat([up6, enc4], dim=1)
        dec6 = self.dec6(merge6)

        up7 = self.up7(dec6)
        merge7 = torch.cat([up7, enc3], dim=1)
        dec7 = self.dec7(merge7)

        up8 = self.up8(dec7)
        merge8 = torch.cat([up8, enc2], dim=1)
        dec8 = self.dec8(merge8)

        up9 = self.up9(dec8)
        merge9 = torch.cat([up9, enc1], dim=1)
        dec9 = self.dec9(merge9)

        out = self.out(dec9)
        return out

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

model = UNet()

# input = torch.randn(1, 3, 1024)
# output = model(input)

# print(output.shape)
# 打印模型的总参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# 打印每个层的参数数量
for name, parameter in model.named_parameters():
    print(f"{name}: {parameter.numel()} parameters")

# 可选：打印模型的概要（如果您有 torchsummary 库）
from torchsummary import summary
summary(model, input_size=(3, 1024))
