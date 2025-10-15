# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# CRAFT (Character Region Awareness for Text Detection) 유틸 함수
# - 모델 출력(score map)으로부터 텍스트 영역 박스(box) 생성
# - connectedComponents를 이용해 연결된 텍스트 영역을 탐지
# -------------------------------------------------------------

import cv2
import numpy as np


# -------------------------------------------------------------
# getDetBoxes
# -------------------------------------------------------------
# Args:
#   textmap (ndarray): CRAFT 모델의 텍스트 점수맵 (float32)
#   text_threshold (float): 텍스트 픽셀 판단 임계값
#   link_threshold (float): 링크(문자 간 연결) 임계값 (사용하지 않음)
#   low_text (float): 낮은 텍스트 점수 픽셀 필터링 기준
#
# Returns:
#   boxes (list): 탐지된 텍스트 영역의 바운딩 박스 리스트
#   None (placeholder, 호환성 유지용)
# -------------------------------------------------------------
def getDetBoxes(textmap, text_threshold, link_threshold, low_text):
    boxes = []

    # ---------------------------------------------------------
    # (1) 텍스트 점수맵에서 low_text 이상인 영역을 이진화
    #     → 연결된 픽셀 영역을 라벨링
    # ---------------------------------------------------------
    # OpenCV 4.x: connectedComponents는 (num_labels, labels) 반환
    num_labels, labels = cv2.connectedComponents((textmap > low_text).astype(np.uint8))

    # ---------------------------------------------------------
    # (2) 각 연결된 영역에 대해 bounding box 계산
    # ---------------------------------------------------------
    for k in range(1, num_labels):  # 0번 라벨은 배경이므로 제외
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255

        # 영역의 외곽 박스 계산 (x, y, w, h)
        x, y, w, h = cv2.boundingRect(segmap)

        # 4개의 꼭짓점 좌표로 변환
        box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        boxes.append(box)

    # ---------------------------------------------------------
    # (3) CRAFT 구조와의 호환성을 위해 두 번째 반환값 None 유지
    # ---------------------------------------------------------
    return boxes, None
