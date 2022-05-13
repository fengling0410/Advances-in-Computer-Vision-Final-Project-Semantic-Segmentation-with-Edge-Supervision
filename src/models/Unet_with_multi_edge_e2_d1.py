from torch import cat, nn, squeeze
import torch
import os

import cv2
import numpy as np

import torchvision

import warnings
warnings.filterwarnings("ignore")

def get_encoder(model, pretrained=True):
    if model == "resnet18":
        encoder = torchvision.models.resnet18(pretrained=pretrained)
    elif model == "resnet34":
        encoder = torchvision.models.resnet34(pretrained=pretrained)
    elif model == "resnet50":
        encoder = torchvision.models.resnet50(pretrained=pretrained)
    elif model == "resnext50":
        encoder = torchvision.models.resnext50_32x4d(pretrained=pretrained)
    elif model == "resnext101":
        encoder = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        
    if model in ["resnet18", "resnet34"]: 
        model = "resnet18-34"
    else: 
        model = "resnet50-101"
        
    filters_dict = {
        "resnet18-34": [512, 512, 256, 128, 64],
        "resnet50-101": [2048, 2048, 1024, 512, 256]
    }

    return encoder, filters_dict[model]


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int, activate=True, batch_norm=False):
        super(ConvRelu, self).__init__()
        self.activate = activate
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(out)
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.activate:
            x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, num_filters: int, batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(num_filters)
        self.activation = nn.ReLU(inplace=True)
        self.conv_block = ConvRelu(in_channels, num_filters, activate=True, batch_norm=True)
        self.conv_block_na = ConvRelu(in_channels, num_filters, activate=False, batch_norm=True)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, inp):
        x = self.conv_block(inp)
        x = self.conv_block_na(x)
        if self.batch_norm:
            x = self.bn(x)
        x = x.add(inp)
        x = self.activation(x)
        return x

class DecoderBlockResNet(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    https://distill.pub/2016/deconv-checkerboard/
    About residual blocks:  
    http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, in_channels, middle_channels, out_channels, res_blocks_dec=False):
        super(DecoderBlockResNet, self).__init__()
        self.in_channels = in_channels
        self.res_blocks_dec = res_blocks_dec

        layers_list = [ConvRelu(in_channels, middle_channels, activate=True, batch_norm=False)]
        
        if self.res_blocks_dec:
            layers_list.append(ResidualBlock(middle_channels, middle_channels, batch_norm=True))
        
        layers_list.append(nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1))
        if not self.res_blocks_dec:
            layers_list.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.block(x)

class UnetResNetEdge(nn.Module):

    def __init__(self, input_channels=3, num_classes=1, num_filters=32, res_blocks_dec=False,
                 Dropout=.2, encoder_name="resnet34", edge_encoder_name = "resnet18", pretrained=True):
        
        super().__init__()

        self.encoder, self.filters_dict = get_encoder(encoder_name, pretrained)
        self.edge_encoder, self.filters_dict_edge = get_encoder(edge_encoder_name, pretrained)
        self.num_classes = num_classes
        self.Dropout = Dropout
        self.res_blocks_dec = res_blocks_dec
        self.input_channels = input_channels
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.channel_tuner = nn.Conv2d(1, 3, kernel_size=1)
        
        # build encoder for image
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        # build encoder for edge
        self.edge_conv1 = nn.Sequential(self.edge_encoder.conv1, self.edge_encoder.bn1, self.edge_encoder.relu, self.pool)
        self.edge_conv2 = self.edge_encoder.layer1
        self.edge_conv3 = self.edge_encoder.layer2
        self.edge_conv4 = self.edge_encoder.layer3
        self.edge_conv5 = self.edge_encoder.layer4

        self.concat_img_edge = nn.Conv2d(512, 256, kernel_size=1)
        
        # build decoder blocks
        self.center = DecoderBlockResNet(self.filters_dict[0], num_filters * 8 * 2, 
                                         num_filters * 8, res_blocks_dec=False)
        self.dec5 = DecoderBlockResNet(self.filters_dict[1] + num_filters * 8, 
                                       num_filters * 8 * 2, num_filters * 8, res_blocks_dec=self.res_blocks_dec)    
        self.dec4 = DecoderBlockResNet(self.filters_dict[2] + num_filters * 8, 
                                       num_filters * 8 * 2, num_filters * 8, res_blocks_dec=self.res_blocks_dec)
        self.dec3 = DecoderBlockResNet(self.filters_dict[3] + num_filters * 8, 
                                       num_filters * 4 * 2, num_filters * 2, res_blocks_dec=self.res_blocks_dec)
        self.dec2 = DecoderBlockResNet(self.filters_dict[4] + num_filters * 2, 
                                       num_filters * 2 * 2, num_filters * 2 * 2, res_blocks_dec=self.res_blocks_dec)
        
        self.dec1 = DecoderBlockResNet(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, res_blocks_dec=False)
        self.dec0 = ConvRelu(num_filters, num_filters)

        # dropout layers
        self.dropout_2d = nn.Dropout2d(p=self.Dropout)
        self.edge_dropout_2d = nn.Dropout2d(p=self.Dropout)

        # the output layer for seg
        self.final_seg = nn.Conv2d(num_filters, num_classes, kernel_size=3, padding=1)
        
        # the output layer for edge
        self.final_edge = nn.Conv2d(num_filters, num_classes, kernel_size=3, padding=1)

        # the final output layer
        self.final = nn.Conv2d(num_classes + num_classes, num_classes, kernel_size = 3, padding = 1)
        

    def forward(self, x, z=None):
        
        # encode image
        conv1 = self.conv1(x)
        # print("conv1: ", conv1.shape) conv1:  torch.Size([2, 64, 128, 128])
        conv2 = self.dropout_2d(self.conv2(conv1))
        # print("conv2: ", conv2.shape) conv2:  torch.Size([2, 64, 128, 128])
        conv3 = self.dropout_2d(self.conv3(conv2))
        # print("conv3: ", conv3.shape) conv3:  torch.Size([2, 128, 64, 64])
        conv4 = self.dropout_2d(self.conv4(conv3))
        # print("conv4: ", conv4.shape) conv4:  torch.Size([2, 256, 32, 32])
        conv5 = self.dropout_2d(self.conv5(conv4))
        # print("conv5: ", conv5.shape) conv5:  torch.Size([2, 512, 16, 16])
        center = self.center(self.pool(conv5))
        # print("center: ", center.shape) center:  torch.Size([2, 256, 16, 16])

        # reduce data storage for memory limit and computation speed
        # conv1_edge = self.edge_conv1(x)
        # # print(conv1_edge.shape)
        # conv2_edge = self.edge_dropout_2d(self.edge_conv2(conv1_edge))
        # conv3_edge = self.edge_dropout_2d(self.edge_conv3(conv2_edge))
        # conv4_edge = self.edge_dropout_2d(self.edge_conv4(conv3_edge))
        # conv5_edge = self.edge_dropout_2d(self.edge_conv5(conv4_edge))
        # center_edge = self.center(self.pool(conv5_edge))

        # encode edge
        # the seg and edge share the same center convolution weights
        center_edge = self.center(self.pool(self.edge_dropout_2d(self.edge_conv5(self.edge_dropout_2d(self.edge_conv4(self.edge_dropout_2d(self.edge_conv3(self.edge_dropout_2d(self.edge_conv2(self.edge_conv1(x)))))))))))

        # concatenate image and edge features into the bottleneck layer
        # f_repeat = torch.cat([torch.empty((1,1,)+(center.size()[2:])).fill_(center_edge[i]) for i in range(z.size()[0])])
        center = torch.cat([center, center_edge], 1)
        center = self.concat_img_edge(center)
        

        # decoder 
        dec5 = self.dec5(cat([center, conv5], 1))
        dec4 = self.dec4(cat([dec5, conv4], 1))
        dec3 = self.dec3(cat([dec4, conv3], 1))
        dec2 = self.dec2(cat([dec3, conv2], 1))
        dec2 = self.dropout_2d(dec2)

        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1) 
        # print("dec0: ", dec0.shape) dec0:  torch.Size([2, 32, 512, 512])

        # output edge prediction
        edge_pred = self.final_edge(dec0)
        # print("edge_pred: ", edge_pred.shape) edge_pred:  torch.Size([bs, 151, 512, 512])

        # concatenate seg prediction and edge prediction
        combined_cat = torch.cat([edge_pred, self.final_seg(dec0)], 1)
        # print("combined_cat: ", combined_cat.shape) combined_cat:  torch.Size([bs, 232, 512, 512])

        return self.final(combined_cat), edge_pred # (torch.Size([bs, 151, 512, 512]), torch.Size([bs, 151, 512, 512]))





