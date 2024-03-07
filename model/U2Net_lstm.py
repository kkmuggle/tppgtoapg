from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class REBNCONV(nn.Module):
    def __init__(self,in_ch=1,out_ch=1,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv1d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm1d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.interpolate(src, size=tar.shape[2:], mode='linear', align_corners=False)

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=1):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool1d(2)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool1d(2)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool1d(2)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool1d(2)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool1d(2)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=1):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool1d(2)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool1d(2)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool1d(2)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool1d(2)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=1):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool1d(2)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool1d(2)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool1d(2)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=1):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool1d(2)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool1d(2)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=1):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool1d(2)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool1d(2)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool1d(2)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool1d(2)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool1d(2)

        self.stage6 = RSU4F(512,256,512)

        self.bilstm_list = nn.ModuleList([
            nn.LSTM(input_size=512, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True),
            nn.LSTM(input_size=512, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True),
            nn.LSTM(input_size=256, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True),
            nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True),
            nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        ])

        self.conv1x1_list = nn.ModuleList([
            nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1),
            nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        ])

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv1d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv1d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv1d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv1d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv1d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv1d(512,out_ch,3,padding=1)

        # Add new BiLSTM layers for the decoder
        self.decoder_bilstm_list = nn.ModuleList([
            nn.LSTM(input_size=512, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True),
            nn.LSTM(input_size=256, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True),
            nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True),
            nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True),
            nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        ])

        self.decoder_conv1x1_list = nn.ModuleList([
            nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        ])

        self.outconv = nn.Conv1d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x
        # print('hx',hx.shape)
        #stage 1
        hx1 = self.stage1(hx)
        # print('hx1', hx1.shape)
        hx = self.pool12(hx1)
        # print('hx1pool', hx.shape)
        #stage 2
        hx2 = self.stage2(hx)
        # print('hx2',hx2.shape)
        hx = self.pool23(hx2)
        # print('hx2pool', hx.shape)

        #stage 3
        hx3 = self.stage3(hx)
        # print('hx3',hx3.shape)
        hx = self.pool34(hx3)
        # print("hx3",hx.shape)
        #stage 4
        hx4 = self.stage4(hx)
        # print('hx4',hx4.shape)
        hx = self.pool45(hx4)
        # print('hx4pool',hx.shape)
        #stage 5
        hx5 = self.stage5(hx)
        # print('hx5',hx5.shape)
        hx = self.pool56(hx5)
        # print('hx5pool', hx.shape)
        #stage 6
        hx6 = self.stage6(hx)
        # print('hx6',hx6.shape)
        # print('hx1',hx1.shape)

        hx_list = [hx5, hx4, hx3, hx2, hx1]

        # Apply BiLSTM layers and 1x1 convolutions to all encoder outputs
        for i in range(len(hx_list)):
            hx_list[i], _ = self.bilstm_list[i](hx_list[i].transpose(1, 2))
            hx_list[i] = hx_list[i].transpose(1, 2)
            hx_list[i] = self.conv1x1_list[i](hx_list[i])

        hx5, hx4, hx3, hx2, hx1 = hx_list

        hx6up = _upsample_like(hx6,hx5)
        # print('hx6up',hx6up.shape)

        #-------------------- decoder --------------------
        # hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        # hx5dup = _upsample_like(hx5d,hx4)
        #
        # hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        # hx4dup = _upsample_like(hx4d,hx3)
        #
        # hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        # hx3dup = _upsample_like(hx3d,hx2)
        #
        # hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        # hx2dup = _upsample_like(hx2d,hx1)
        #
        # hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        # print('hx5d',hx5d.shape)
        hx5d, _ = self.decoder_bilstm_list[0](hx5d.transpose(1, 2))
        # print('hx5d=lstm',hx5d.shape)
        hx5d = hx5d.transpose(1, 2)
        # print('hx5d',hx5d.shape)
        hx5d = self.decoder_conv1x1_list[0](hx5d)
        # print('hx5d',hx5d.shape)

        hx5dup = _upsample_like(hx5d, hx4)
        # print(hx5dup.shape)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4d, _ = self.decoder_bilstm_list[1](hx4d.transpose(1, 2))
        hx4d = hx4d.transpose(1, 2)

        hx4d = self.decoder_conv1x1_list[1](hx4d)

        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3d, _ = self.decoder_bilstm_list[2](hx3d.transpose(1, 2))
        hx3d = hx3d.transpose(1, 2)
        hx3d = self.decoder_conv1x1_list[2](hx3d)

        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2d, _ = self.decoder_bilstm_list[3](hx2d.transpose(1, 2))
        hx2d = hx2d.transpose(1, 2)
        hx2d = self.decoder_conv1x1_list[3](hx2d)

        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        hx1d, _ = self.decoder_bilstm_list[4](hx1d.transpose(1, 2))
        hx1d = hx1d.transpose(1, 2)
        hx1d = self.decoder_conv1x1_list[4](hx1d)




        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)


model = U2NET()
# input = torch.randn(1, 3, 1024)
# output = model(input)
# print(output)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")