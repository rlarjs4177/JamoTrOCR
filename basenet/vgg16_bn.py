# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# 📘 Source:
#   https://github.com/clovaai/CRAFT-pytorch/blob/master/basenet/vgg16_bn.py
# 📄 License: MIT (NAVER Clova AI Research)
# -------------------------------------------------------------
# Description:
#   - VGG16 backbone network with Batch Normalization (BN)
#   - Used in CRAFT (Character Region Awareness for Text Detection)
#   - Extracts hierarchical convolutional feature maps from an input image
#   - These feature maps are later fused by U-Net–style upsampling in craft.py
# -------------------------------------------------------------

from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models

# -------------------------------------------------------------
# 사전학습된 VGG16 모델 가중치 URL
# -------------------------------------------------------------
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

# -------------------------------------------------------------
# 가중치 초기화 함수
# -------------------------------------------------------------
def init_weights(modules):
    """
    네트워크 내 레이어들의 가중치를 초기화.
    - Conv2d: Xavier uniform 초기화
    - BatchNorm2d: gamma=1, beta=0
    - Linear: Gaussian(0, 0.01)
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

# -------------------------------------------------------------
# VGG16-BN Backbone 클래스 정의
# -------------------------------------------------------------
class vgg16_bn(torch.nn.Module):
    """
    CRAFT의 백본 네트워크로 사용되는 VGG16 + BatchNorm 모델
    - conv2_2, conv3_3, conv4_3, conv5_3 단계의 feature map을 추출
    - 추가적으로 dilated convolution(fc6, fc7 역할)을 포함
    """

    def __init__(self, pretrained=True, freeze=True):
        """
        Args:
            pretrained (bool): torchvision의 ImageNet 사전학습 가중치 사용 여부
            freeze (bool): 초기 conv layer를 고정할지 여부
        """
        super(vgg16_bn, self).__init__()

        # HTTPS → HTTP 변환 (일부 환경에서 HTTPS 다운로드 오류 방지)
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')

        # torchvision에서 pretrained VGG16-BN 모델의 feature extractor 불러오기
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        # -------------------------------------------------------------
        # VGG16을 다섯 개의 블록(slice)으로 분할
        # 각 블록은 서로 다른 계층의 feature map을 추출
        # -------------------------------------------------------------
        self.slice1 = torch.nn.Sequential()  # conv2_2
        self.slice2 = torch.nn.Sequential()  # conv3_3
        self.slice3 = torch.nn.Sequential()  # conv4_3
        self.slice4 = torch.nn.Sequential()  # conv5_3
        self.slice5 = torch.nn.Sequential()  # fc6, fc7 대체 (dilated conv)

        # conv1_1 ~ conv2_2 (index 0~11)
        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        # conv3_1 ~ conv3_3 (index 12~18)
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        # conv4_1 ~ conv4_3 (index 19~28)
        for x in range(19, 29):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        # conv5_1 ~ conv5_3 (index 29~38)
        for x in range(29, 39):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # -------------------------------------------------------------
        # 추가 계층 (FC6, FC7 역할을 하는 Dilated Convolution)
        # -------------------------------------------------------------
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),  # dilated conv
            nn.Conv2d(1024, 1024, kernel_size=1)  # 1x1 conv
        )

        # -------------------------------------------------------------
        # 사전학습을 사용하지 않는 경우, 가중치 수동 초기화
        # -------------------------------------------------------------
        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        # fc6, fc7 대체 부분은 항상 새로 초기화
        init_weights(self.slice5.modules())

        # -------------------------------------------------------------
        # freeze=True인 경우, slice1 (초기 conv layer) 고정
        # -------------------------------------------------------------
        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False

    # -------------------------------------------------------------
    # Forward 연산 정의
    # -------------------------------------------------------------
    def forward(self, X):
        """
        입력 이미지를 VGG16 구조를 통해 다단계 feature map으로 변환.
        반환되는 feature들은 craft.py의 U-Net 업샘플링 경로에 전달된다.
        """
        # 각 블록별 특징 추출
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h

        # 여러 단계의 feature를 튜플 형태로 반환
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
