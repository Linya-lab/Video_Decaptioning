import math
import torch
import functools
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.nn.init import kaiming_normal_, constant_

class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((2,2)),
            DoubleConv2D(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.conv = DoubleConv2D(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)
        
class MaskUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MaskUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv2D(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        b,t,c,h,w=x.size()
        inc=x.contiguous().view(b*t,c,h,w)
        x1 = self.inc(inc)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits=logits.contiguous().view(b,t,1,h,w)
        return logits

class Gatedconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1,bias=True,activation=nn.LeakyReLU(0.2, inplace=True)):
        super(Gatedconv2d,self).__init__()
        self.gatingConv=nn.Conv2d(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups, bias)
        self.featureConv=nn.Conv2d(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups, bias)
        self.sigmoid = nn.Sigmoid()
        self.activation=activation

    def forward(self, xs):
        gating = self.gatingConv(xs)
        feature = self.featureConv(xs)
        if self.activation is not None:
            feature = self.activation(feature)
        out=self.sigmoid(gating)*feature
        return out

class GatedDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True,activation=nn.LeakyReLU(0.2, inplace=True),scale_factor=2):
        super(GatedDeconv2d,self).__init__()
        self.unsample=torch.nn.Upsample(scale_factor=scale_factor)
        self.conv = Gatedconv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,groups=groups,bias=bias,activation=activation)
    def forward(self,xs):
        xs=self.unsample(xs)
        return self.conv(xs)
      
class Frames_patch(nn.Module):
    def __init__(self,psize):
        super(Frames_patch, self).__init__()
        self.psize=psize
        self.softmax=nn.Softmax(dim=-1)
        self.linear=nn.Sequential(nn.Conv2d(256*len(psize),256,3,1,1),
                                nn.LeakyReLU(0.2, inplace=True))
        self.convg=Gatedconv2d(512,256,3,1,1)

    def forward(self,x):
        x,xs,ms=x['x'],x['xs'],x['ms']
        b,c,h,w=x.size()
        t=xs.size(0)//b
        output=[]
        for (p_h,p_w) in self.psize:
            out_h,out_w=h//p_h,w//p_w
            mm=ms.view(b,t,1,out_h,p_h,out_w,p_w)
            mm=mm.permute(0,1,3,5,2,4,6).contiguous().view(b,t*out_h*out_w,p_h*p_w)
            mm=(torch.mean(mm,dim=-1,keepdim=False)>0.5).unsqueeze(1).repeat(1,out_h*out_w,1)
            query=x.view(b,c,out_h,p_h,out_w,p_w)
            query=query.permute(0,2,4,1,3,5).contiguous().view(b,out_h*out_w,c*p_h*p_w)
            key=xs.view(b,t,c,out_h,p_h,out_w,p_w)
            key=key.permute(0,1,3,5,2,4,6).contiguous().view(b,t*out_h*out_w,c*p_h*p_w)
            value=key
            scores=torch.matmul(query,key.transpose(-2, -1))/math.sqrt(query.size(-1))
            scores.masked_fill(mm,-1e9)
            atten=self.softmax(scores)
            val=torch.matmul(atten,value)
            y=val.view(b,out_h,out_w,c,p_h,p_w)
            y=y.permute(0,3,1,4,2,5).contiguous().view(b,c,h,w)
            output.append(y)
        output=torch.cat(output,dim=1)
        output=self.linear(output)
        x=self.convg(torch.cat([x,output],dim=1))
        return {'x':x,'xs':xs,'ms':ms}

class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.convg1=Gatedconv2d(4,64,3,1,1)
        self.convg2=Gatedconv2d(64,128,3,2,1)
        self.convg3=Gatedconv2d(128,256,3,2,1)
        self.convg4=Gatedconv2d(256,256,3,2,1)

        self.frames_atten=nn.Sequential(Frames_patch([(1,1),(2,2),(4,4),(8,8)]),
                                        Frames_patch([(1,1),(2,2),(4,4),(8,8)]),
                                        Frames_patch([(1,1),(2,2),(4,4),(8,8)]),
                                        Frames_patch([(1,1),(2,2),(4,4),(8,8)]))

        self.deconv1=nn.Sequential(Gatedconv2d(512,256,3,1,1),
                                    GatedDeconv2d(256,256,3,1,1))
        self.deconv2=nn.Sequential(Gatedconv2d(512,256,3,1,1),
                                    GatedDeconv2d(256,128,3,1,1))
        self.deconv3=nn.Sequential(Gatedconv2d(256,128,3,1,1),
                                    GatedDeconv2d(128,64,3,1,1))
        self.deconv4=nn.Sequential(Gatedconv2d(128,64,3,1,1),
                                    nn.Conv2d(64,3,3,1,1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self,imgs,masks):
        inc=torch.cat([imgs,masks],dim=2)
        b,t,c,h,w=inc.size()
        inc=inc.contiguous().view(b*t,c,h,w)
        ms=masks.contiguous().view(b*t,1,h,w)
        ms=(F.interpolate(ms,scale_factor=1/8,recompute_scale_factor=False)>0).float() 

        xe1=self.convg1(inc)
        bt,c,h,w=xe1.size()
        x1=xe1.view(b,t,c,h,w)
        x1=x1[:,t//2]
        xe2=self.convg2(xe1)
        bt,c,h,w=xe2.size()
        x2=xe2.view(b,t,c,h,w)
        x2=x2[:,t//2]
        xe3=self.convg3(xe2)
        bt,c,h,w=xe3.size()
        x3=xe3.view(b,t,c,h,w)
        x3=x3[:,t//2]
        xe4=self.convg4(xe3)
        bt,c,h,w=xe4.size()
        x4=xe4.view(b,t,c,h,w)
        x4=x4[:,t//2]

        xa=self.frames_atten({'x':x4,'xs':xe4,'ms':ms})['x']
        y1=self.deconv1(torch.cat([xa,x4],dim=1))
        y2=self.deconv2(torch.cat([y1,x3],dim=1))
        y3=self.deconv3(torch.cat([y2,x2],dim=1))
        y=self.deconv4(torch.cat([y3,x1],dim=1))
        y=torch.clamp(y,-1.,1.)
        return y