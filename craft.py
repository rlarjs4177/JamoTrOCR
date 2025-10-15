"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# CRAFT (Character Region Awareness for Text Detection)
# - 문자 영역과 문자 간 연결 관계를 예측하는 텍스트 탐지 모델
# - VGG16-BN을 백본으로 사용하며, U-Net 형태의 업샘플링 구조로 구성
# - 출력: (1) 문자 영역 점수맵, (2) 링크 점수맵 (문자 간 연결)
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from basenet.vgg16_bn import vgg16_bn, init_weights


# -------------------------------------------------------------
# double_conv 모듈
# - 업샘플 단계에서 사용되는 기본 Conv 블록
# - (입력채널 + skip 연결채널) → mid_ch → out_ch
# -------------------------------------------------------------
class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            # 1x1 convolution: 채널 수를 중간 크기(mid_ch)로 조정
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            # 3x3 convolution: 공간 정보를 통합하며 출력 채널 생성
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 입력 feature에 conv 블록 적용
        x = self.conv(x)
        return x


# -------------------------------------------------------------
# CRAFT 모델 본체
# -------------------------------------------------------------
class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ 
        Base network (VGG16-BN backbone) 
        - 입력 이미지를 다양한 해상도의 feature map으로 변환
        - sources: [stage1, stage2, stage3, stage4, stage5]
        """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ 
        U-Net 업샘플링 경로 구성 
        - 각 단계에서 상위 feature를 업샘플하여 하위 feature와 결합
        """
        self.upconv1 = double_conv(1024, 512, 256)  # stage1 + stage2
        self.upconv2 = double_conv(512, 256, 128)   # stage2 + stage3
        self.upconv3 = double_conv(256, 128, 64)    # stage3 + stage4
        self.upconv4 = double_conv(128, 64, 32)     # stage4 + stage5

        """ 
        Classification branch 
        - 최종 feature로부터 문자/링크 영역 점수맵 예측
        - num_class = 2 (문자 영역, 링크 영역)
        """
        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),  # 최종 출력 (2채널)
        )

        # 가중치 초기화
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """ 
        순전파 과정 (Forward)
        1. 입력 이미지 x를 VGG16-BN 백본에 통과시켜 multi-scale feature 추출
        2. 업샘플 및 skip 연결을 통해 고해상도 feature 복원
        3. 문자/링크 영역 score map 출력
        """
        # -----------------------------------------------------
        # ① Base network
        # -----------------------------------------------------
        sources = self.basenet(x)  # [f1, f2, f3, f4, f5]

        # -----------------------------------------------------
        # ② U-Net 업샘플링 경로
        # -----------------------------------------------------

        # Stage 1: 가장 깊은 feature 결합
        y = torch.cat([sources[0], sources[1]], dim=1)  # 채널 방향 병합
        y = self.upconv1(y)

        # Stage 2: 업샘플 후 다음 레벨 feature와 결합
        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        # Stage 3: 업샘플 및 병합 반복
        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        # Stage 4: 최종 업샘플 단계
        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)  # 최종 feature map (고해상도)

        # -----------------------------------------------------
        # ③ Classification layer (문자 / 링크 영역 예측)
        # -----------------------------------------------------
        y = self.conv_cls(feature)

        # -----------------------------------------------------
        # ④ 출력 정리
        # permute: (B, C, H, W) → (B, H, W, C)
        # feature: 후속 refinement 또는 loss 계산용 feature map
        # -----------------------------------------------------
        return y.permute(0, 2, 3, 1), feature


# -------------------------------------------------------------
# 실행 테스트
# -------------------------------------------------------------
if __name__ == '__main__':
    # 모델 초기화 (사전학습된 가중치 사용)
    model = CRAFT(pretrained=True).cuda()

    # 더미 입력 (1 x 3 x 768 x 768)
    dummy = torch.randn(1, 3, 768, 768).cuda()

    # Forward 수행
    output, feature = model(dummy)

    # 출력 형태 확인: (batch, height, width, num_class)
    print(output.shape)
