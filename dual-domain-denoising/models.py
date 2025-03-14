import numpy as np
import pickle as pk
import torch
import torch.nn as nn
from constants import *
from torchsummary import summary

np.random.seed(SEED)

# Us module in wNet
class UNet_kdata(nn.Module):
    def __init__(self):
        super().__init__()
        
        # encoder
        self.e11 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.relu11 = nn.LeakyReLU(inplace=True)
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu12 = nn.LeakyReLU(inplace=True)
        self.e13 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu13 = nn.LeakyReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.e21 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu21 = nn.LeakyReLU(inplace=True)
        self.e22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu22 = nn.LeakyReLU(inplace=True)
        self.e23 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu23 = nn.LeakyReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.e31 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu31 = nn.LeakyReLU(inplace=True)
        self.e32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu32 = nn.LeakyReLU(inplace=True)
        self.e33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu33 = nn.LeakyReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.e41 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu41 = nn.LeakyReLU(inplace=True)
        self.e42 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu42 = nn.LeakyReLU(inplace=True)
        self.e43 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu43 = nn.LeakyReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.brelu1 = nn.LeakyReLU(inplace=True)
        self.bottleneck2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.brelu2 = nn.LeakyReLU(inplace=True)

        # decoder
        self.d11 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.drelu11 = nn.LeakyReLU(inplace=True)

        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.drelu21 = nn.LeakyReLU(inplace=True)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.drelu22 = nn.LeakyReLU(inplace=True)
        self.d23 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.drelu23 = nn.LeakyReLU(inplace=True)
        self.d24 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d25 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.drelu25 = nn.LeakyReLU(inplace=True)

        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.drelu31 = nn.LeakyReLU(inplace=True)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.drelu32 = nn.LeakyReLU(inplace=True)
        self.d33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.drelu33 = nn.LeakyReLU(inplace=True)
        self.d34 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d35 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.drelu35 = nn.LeakyReLU(inplace=True)

        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.drelu41 = nn.LeakyReLU(inplace=True)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.drelu42 = nn.LeakyReLU(inplace=True)
        self.d43 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.drelu43 = nn.LeakyReLU(inplace=True)
        self.d44 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d45 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.drelu45 = nn.LeakyReLU(inplace=True)

        self.out1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.outrelu1 = nn.LeakyReLU(inplace=True)
        self.out2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.outrelu2 = nn.LeakyReLU(inplace=True)
        self.out3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.outrelu3 = nn.LeakyReLU(inplace=True)
        self.out4 = nn.Conv2d(32, 2, kernel_size=1)


    def forward(self, x):
        # encoder
        xe11 = self.e11(x)
        xrelu11 = self.relu11(xe11)
        xe12 = self.e12(xrelu11)
        xrelu12 = self.relu12(xe12)
        xe13 = self.e13(xrelu12)
        xrelu13 = self.relu13(xe13)
        xpool1 = self.pool1(xrelu13)

        xe21 = self.e21(xpool1)
        xrelu21 = self.relu21(xe21)
        xe22 = self.e22(xrelu21)
        xrelu22 = self.relu22(xe22)
        xe23 = self.e23(xrelu22)
        xrelu23 = self.relu23(xe23)
        xpool2 = self.pool2(xrelu23)

        xe31 = self.e31(xpool2)
        xrelu31 = self.relu31(xe31)
        xe32 = self.e32(xrelu31)
        xrelu32 = self.relu32(xe32)
        xe33 = self.e33(xrelu32)
        xrelu33 = self.relu33(xe33)
        xpool3 = self.pool3(xrelu33)

        xe41 = self.e41(xpool3)
        xrelu41 = self.relu41(xe41)
        xe42 = self.e42(xrelu41)
        xrelu42 = self.relu42(xe42)
        xe43 = self.e43(xrelu42)
        xrelu43 = self.relu43(xe43)
        xpool4 = self.pool4(xrelu43)

        xbottleneck1 = self.bottleneck1(xpool4)
        xbrelu1 = self.brelu1(xbottleneck1)
        xbottleneck2 = self.bottleneck2(xbrelu1)
        xbrelu2 = self.brelu2(xbottleneck2)

        # decoder
        xd11 = self.d11(xbrelu2)
        xdrelu11 = self.drelu11(xd11)

        xd21 = self.d21(torch.cat([xrelu43, xdrelu11], dim=1))
        xdrelu21 = self.drelu21(xd21)
        xd22 = self.d22(xdrelu21)
        xdrelu22 = self.drelu22(xd22)
        xd23 = self.d23(xdrelu22)
        xdrelu23 = self.drelu23(xd23)
        xd24 = self.d24(xdrelu23)
        xd25 = self.d25(xd24)
        xdrelu25 = self.drelu25(xd25)

        xd31 = self.d31(torch.cat([xrelu33, xdrelu25], dim=1))
        xdrelu31 = self.drelu31(xd31)
        xd32 = self.d32(xdrelu31)
        xdrelu32 = self.drelu32(xd32)
        xd33 = self.d33(xdrelu32)
        xdrelu33 = self.drelu33(xd33)
        xd34 = self.d34(xdrelu33)
        xd35 = self.d35(xd34)
        xdrelu35 = self.drelu35(xd35)

        xd41 = self.d41(torch.cat([xrelu23, xdrelu35], dim=1))
        xdrelu41 = self.drelu41(xd41)
        xd42 = self.d42(xdrelu41)
        xdrelu42 = self.drelu42(xd42)
        xd43 = self.d43(xdrelu42)
        xdrelu43 = self.drelu43(xd43)
        xd44 = self.d44(xdrelu43)
        xd45 = self.d45(xd44)
        xdrelu45 = self.drelu45(xd45)

        xout1 = self.out1(torch.cat([xrelu13, xdrelu45], dim=1))
        xoutrelu1 = self.outrelu1(xout1)
        xout2 = self.out2(xoutrelu1)
        xoutrelu2 = self.outrelu2(xout2)
        xout3 = self.out3(xoutrelu2)
        xoutrelu3 = self.outrelu3(xout3)
        xout4 = self.out4(xoutrelu3)

        return xout4


# Ui module in wNet
class UNet_image(nn.Module):
    def __init__(self):
        super().__init__()
        
        # encoder
        self.e11 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.e21 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu21 = nn.ReLU(inplace=True)
        self.e22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu22 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.e31 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu31 = nn.ReLU(inplace=True)
        self.e32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu32 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.e41 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu41 = nn.ReLU(inplace=True)
        self.e42 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu42 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.brelu1 = nn.ReLU(inplace=True)
        self.bottleneck2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.brelu2 = nn.ReLU(inplace=True)

        # decoder
        self.d11 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.drelu11 = nn.LeakyReLU(inplace=True)

        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.drelu21 = nn.ReLU(inplace=True)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.drelu22 = nn.ReLU(inplace=True)
        self.d24 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d25 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.drelu25 = nn.LeakyReLU(inplace=True)

        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.drelu31 = nn.ReLU(inplace=True)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.drelu32 = nn.ReLU(inplace=True)
        self.d34 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d35 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.drelu35 = nn.LeakyReLU(inplace=True)

        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.drelu41 = nn.ReLU(inplace=True)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.drelu42 = nn.ReLU(inplace=True)
        self.d44 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d45 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.drelu45 = nn.LeakyReLU(inplace=True)

        self.out1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.outrelu1 = nn.ReLU(inplace=True)
        self.out2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.outrelu2 = nn.ReLU(inplace=True)
        self.out4 = nn.Conv2d(32, 1, kernel_size=1)


    def forward(self, x):
        # encoder
        xe11 = self.e11(x)
        xrelu11 = self.relu11(xe11)
        xe12 = self.e12(xrelu11)
        xrelu12 = self.relu12(xe12)
        xpool1 = self.pool1(xrelu12)

        xe21 = self.e21(xpool1)
        xrelu21 = self.relu21(xe21)
        xe22 = self.e22(xrelu21)
        xrelu22 = self.relu22(xe22)
        xpool2 = self.pool2(xrelu22)

        xe31 = self.e31(xpool2)
        xrelu31 = self.relu31(xe31)
        xe32 = self.e32(xrelu31)
        xrelu32 = self.relu32(xe32)
        xpool3 = self.pool3(xrelu32)

        xe41 = self.e41(xpool3)
        xrelu41 = self.relu41(xe41)
        xe42 = self.e42(xrelu41)
        xrelu42 = self.relu42(xe42)
        xpool4 = self.pool4(xrelu42)

        xbottleneck1 = self.bottleneck1(xpool4)
        xbrelu1 = self.brelu1(xbottleneck1)
        xbottleneck2 = self.bottleneck2(xbrelu1)
        xbrelu2 = self.brelu2(xbottleneck2)

        # decoder
        xd11 = self.d11(xbrelu2)
        xdrelu11 = self.drelu11(xd11)

        xd21 = self.d21(torch.cat([xrelu42, xdrelu11], dim=1))
        xdrelu21 = self.drelu21(xd21)
        xd22 = self.d22(xdrelu21)
        xdrelu22 = self.drelu22(xd22)
        xd24 = self.d24(xdrelu22)
        xd25 = self.d25(xd24)
        xdrelu25 = self.drelu25(xd25)

        xd31 = self.d31(torch.cat([xrelu32, xdrelu25], dim=1))
        xdrelu31 = self.drelu31(xd31)
        xd32 = self.d32(xdrelu31)
        xdrelu32 = self.drelu32(xd32)
        xd34 = self.d34(xdrelu32)
        xd35 = self.d35(xd34)
        xdrelu35 = self.drelu35(xd35)

        xd41 = self.d41(torch.cat([xrelu22, xdrelu35], dim=1))
        xdrelu41 = self.drelu41(xd41)
        xd42 = self.d42(xdrelu41)
        xdrelu42 = self.drelu42(xd42)
        xd44 = self.d44(xdrelu42)
        xd45 = self.d45(xd44)
        xdrelu45 = self.drelu45(xd45)

        xout1 = self.out1(torch.cat([xrelu12, xdrelu45], dim=1))
        xoutrelu1 = self.outrelu1(xout1)
        xout2 = self.out2(xoutrelu1)
        xoutrelu2 = self.outrelu2(xout2)
        xout4 = self.out4(xoutrelu2)

        return xout4
    

def main():
    model_k = UNet_kdata()
    summary(model_k, (2, 64, 64))

    model_i = UNet_image()
    summary(model_i, (1, 64, 64))


if __name__ == '__main__':
    main()
