import torch.nn as nn
import torch
import torchvision.models as models
from modules_resunet import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)

vgg19 = models.vgg19_bn( pretrained=True,progress=True)
vgg19.features[0]=nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[1]=nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[3]=nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[4]=nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[6]=nn.Sequential(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True))
vgg19.features[7]=nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[8]=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[10]=nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[11]=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[14]=nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[15]=nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[17]=nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[18]=nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[20]=nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[21]=nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[23]=nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[24]=nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[27]=nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[28]=nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[30]=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[31]=nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[33]=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[34]=nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[36]=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg19.features[37]=nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
vgg19.features[40]=nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg1=vgg19.features[0:7]
vgg2=vgg19.features[7:14]
vgg3=vgg19.features[14:27]
vgg4=vgg19.features[27:40] 
vgg5=vgg19.features[40::]

resnet = models.resnet34(pretrained=True)
resnet.conv1=nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet.bn1=nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# resnet.layer1[0].conv1=nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
print(resnet)
class attention_block(nn.Module):
    def __init__(self, in_channels, out_channels,batchszie1):
        super(attention_block,self).__init__() 
        self.A=torch.ones(batchszie1,1,3,3).cuda()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.attention1=nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels//4,
                                      kernel_size=(1,3), stride=(1,1), padding=(0,1)),
                                      nn.BatchNorm2d(self.out_channels//4),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.out_channels//4, self.out_channels,
                                      kernel_size=(1,3), stride=(1,1), padding=(0,1)),
                                      nn.BatchNorm2d(self.out_channels),
                                      nn.ReLU(inplace=True))
    def forward(self, x1,x2,x3):
        self.A_attention=self.attention1(self.A).cuda()
        # self.A_attention_gap= self.A_attention
        self.A_attention_gap=torch.softmax(self.A_attention,dim=3)
        self.A11=self.A_attention_gap[:,:,0,0].unsqueeze(2).unsqueeze(3)
        self.A12=self.A_attention_gap[:,:,0,1].unsqueeze(2).unsqueeze(3)
        self.A13=self.A_attention_gap[:,:,0,2].unsqueeze(2).unsqueeze(3)
        self.A21=self.A_attention_gap[:,:,1,0].unsqueeze(2).unsqueeze(3)
        self.A22=self.A_attention_gap[:,:,1,1].unsqueeze(2).unsqueeze(3)
        self.A23=self.A_attention_gap[:,:,1,2].unsqueeze(2).unsqueeze(3)
        self.A31=self.A_attention_gap[:,:,2,0].unsqueeze(2).unsqueeze(3)
        self.A32=self.A_attention_gap[:,:,2,1].unsqueeze(2).unsqueeze(3)
        self.A33=self.A_attention_gap[:,:,2,2].unsqueeze(2).unsqueeze(3)
        attention_x1= self.A11*x1+self.A12*x2
        attention_x2= self.A21*x1+self.A22*x2+self.A23*x3
        attention_x3= self.A32*x2+self.A33*x3
        return attention_x1,attention_x2,attention_x3



class ResUnetPlusPlus(nn.Module):
    def __init__(self, batchsize1,channel=1, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()
        self.batchsize1=batchsize1
        self.SelfAttentionBlock1=attention_block(1,64,batchsize1)
        self.SelfAttentionBlock2=attention_block(1,128,batchsize1)
        self.SelfAttentionBlock3=attention_block(1,256,batchsize1)
        self.SelfAttentionBlock4=attention_block(1,512,batchsize1)
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),     
                        resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
       
        self.encoder12 = vgg1
        self.encoder22 = vgg2
        self.encoder32 = vgg3
        self.encoder42 = vgg4
        self.encoder52 = vgg5
        
        self.convation=nn.Sequential(nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1 ,1), padding=(1, 1), bias=False),
                              nn.BatchNorm2d(32),
                              nn.ReLU(inplace=True))
        
        '''回归模型'''
        self.l1=nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2 ,2), padding=(1, 1), bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(inplace=True))
        self.l2=nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(inplace=True))
        self.l3=nn.Sequential(nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                              nn.BatchNorm2d(64),
                              nn.ReLU(inplace=True))
        self.l4=nn.Sequential(nn.Linear(1024, 512),  
                              nn.ReLU(inplace=True),  
                              nn.Dropout(p=0.1),  
                              nn.Linear(512, 256),
                              nn.ReLU(inplace=True),
                              nn.Dropout(p=0.1),  
                              nn.Linear(256, 128),
                              nn.ReLU(inplace=True),
                              nn.Linear(128, 64),
                              nn.ReLU(inplace=True),
                              nn.Linear(64, 2))
        self.oneconv1=nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.oneconv2=nn.Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.oneconv=nn.Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x11 = self.encoder1_conv(x)  
        x11 = self.encoder1_bn(x11)
        x11 = self.encoder1_relu(x11) 
        
        x111 = self.encoder12(x)  





        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)
        x22=self.encoder2(x11)
        x222 = self.encoder22(x111)  
        
        x2,x22,x222=self.SelfAttentionBlock1(x2,x22,x222)
        x2=torch.cat([x2,x22,x222],dim=1)
        x2=self.oneconv1(x2)
        


        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)
        x33=self.encoder3(x22)
        x333 = self.encoder32(x222)  
        
        x3,x33,x333=self.SelfAttentionBlock2(x3,x33,x333)
        x3=torch.cat([x3,x33,x333],dim=1)
        x3=self.oneconv2(x3)




        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)
        x44=self.encoder4(x33)
        x444 = self.encoder42(x333)  
        
        x4,x44,x444=self.SelfAttentionBlock3(x4,x44,x444)
        x41=torch.cat([x4,x44,x444],dim=1)
        x41=self.oneconv(x41)

        x5 = self.aspp_bridge(x41)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)
        # x9 = self.convation(x8)
        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        
        '''生成标签元素个数与背景元素个数'''
        q1=self.l1(x4)
        q2=self.l2(q1)
        q3=self.l3(q2)
        q4=q3.view(self.batchsize1, -1)
        q4=self.l4(q4)

        return out,q4