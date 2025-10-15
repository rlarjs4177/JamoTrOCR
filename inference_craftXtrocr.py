# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# CRAFT + JamoTrOCR 통합 추론 코드
# -------------------------------------------------------------
# 1. CRAFT: 텍스트 영역 탐지 (문자 박스 좌표 추출)
# 2. TrOCR: 탐지된 영역을 crop하여 OCR 수행 (자모 단위 인식)
# 3. PIL로 원본 이미지에 텍스트 박스 및 인식 결과 시각화
# -------------------------------------------------------------
# 출력:
#   ./results/ 폴더에 텍스트 박스 + 인식 텍스트가 포함된 이미지 저장
# -------------------------------------------------------------

import os
import glob
import time
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict
import unicodedata
import torch.nn as nn
from torchvision import transforms
from model.trocr_model import JamoTrOCR
from tokenizer.jamo_tokenizer import JamoTokenizer


# -------------------------------------------------------------
# 자모 결합 함수 (초성, 중성, 종성을 완성형 문자로 조합)
# -------------------------------------------------------------
def join_jamos(text):
    return unicodedata.normalize("NFC", text)


# -------------------------------------------------------------
# state_dict의 "module." prefix 제거 (DataParallel 저장 호환용)
# -------------------------------------------------------------
def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # "module." 제거
            new_state_dict[name] = v
        return new_state_dict
    return state_dict


# -------------------------------------------------------------
# 입력 이미지를 비율 유지하며 리사이즈
# -------------------------------------------------------------
def resize_aspect_ratio(img, max_size=1280):
    h, w, _ = img.shape
    ratio = max_size / max(h, w)
    target_h, target_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(img, (target_w, target_h))
    return resized, ratio, ratio


# -------------------------------------------------------------
# 이미지 정규화 및 텐서 변환
# -------------------------------------------------------------
def normalize_image(img):
    img = img.astype(np.float32) / 255.               # [0,1] 정규화
    img -= (0.485, 0.456, 0.406)                      # ImageNet 평균값
    img /= (0.229, 0.224, 0.225)                      # 표준편차로 정규화
    img = img.transpose(2, 0, 1)                      # (H, W, C) → (C, H, W)
    img = torch.from_numpy(img).unsqueeze(0)          # 배치 차원 추가 (1, C, H, W)
    return img


# -------------------------------------------------------------
# CRAFT 예측 결과에서 텍스트 박스 추출
# - score_text, score_link: 모델 출력 점수맵
# -------------------------------------------------------------
def get_text_boxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
    _, text_score = cv2.threshold(score_text, low_text, 1, 0)
    _, link_score = cv2.threshold(score_link, link_threshold, 1, 0)
    combined = np.clip(text_score + link_score, 0, 1)

    nLabels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined.astype(np.uint8), connectivity=4
    )

    boxes = []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue
        where = labels == k
        if np.max(score_text[where]) < text_threshold:
            continue

        segmap = np.zeros(score_text.shape, dtype=np.uint8)
        segmap[where] = 255
        contours, _ = cv2.findContours(segmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        x, y, w, h = cv2.boundingRect(contours[0])
        box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        boxes.append(box)
    return boxes


# -------------------------------------------------------------
# 텍스트 박스 확장 함수
# - 인식 안정성을 위해 박스를 약간 확장
# -------------------------------------------------------------
def expand_box_xy(box, expand_ratio_x=0.1, expand_ratio_y=0.2, image_shape=None):
    center = box.mean(axis=0)
    vecs = box - center
    expanded_box = box.copy()
    expanded_box[:, 0] = center[0] + vecs[:, 0] * (1 + expand_ratio_x)
    expanded_box[:, 1] = center[1] + vecs[:, 1] * (1 + expand_ratio_y)

    # 이미지 경계 밖으로 나가지 않도록 클리핑
    if image_shape is not None:
        h, w = image_shape[:2]
        expanded_box[:, 0] = np.clip(expanded_box[:, 0], 0, w - 1)
        expanded_box[:, 1] = np.clip(expanded_box[:, 1], 0, h - 1)

    return expanded_box.astype(np.int32)


# -------------------------------------------------------------
# TrOCR 벡터화 추론 (배치 입력)
# - 여러 텍스트 박스를 한 번에 디코딩
# -------------------------------------------------------------
def generate_batch(model, image_tensors, tokenizer, max_length=128):
    """
    Args:
        image_tensors: (B, C, H, W)
    Returns:
        outputs: 각 이미지에 대한 예측 토큰 리스트
    """
    device = image_tensors.device
    B = image_tensors.size(0)

    with torch.no_grad():
        # 인코더 수행
        enc_outputs = model.encoder(pixel_values=image_tensors)
        memory = model.enc_to_dec(enc_outputs.last_hidden_state)
        memory = memory.transpose(0, 1)  # (seq, batch, dim)

        # BOS 토큰으로 초기화
        input_ids = torch.full((B, 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs = [[] for _ in range(B)]

        # 토큰을 순차적으로 생성 (Greedy)
        for _ in range(max_length):
            tgt_emb = model.embedding(input_ids)
            pos_emb = model.pos_embedding[:tgt_emb.size(1), :].unsqueeze(0)
            tgt_emb = tgt_emb + pos_emb
            tgt_emb = tgt_emb.transpose(0, 1)

            # 미래 토큰 차단용 mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(device)
            dec_out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            dec_out = dec_out.transpose(0, 1)

            logits = model.fc_out(dec_out[:, -1, :])  # (B, vocab_size)
            next_tokens = logits.argmax(dim=-1)

            # EOS 토큰이 나올 때까지 반복
            for i in range(B):
                if not finished[i]:
                    if next_tokens[i].item() == tokenizer.eos_token_id:
                        finished[i] = True
                    else:
                        outputs[i].append(next_tokens[i].item())

            # 다음 step 입력으로 이어붙이기
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            if finished.all():
                break
    return outputs


# -------------------------------------------------------------
# 메인 함수: CRAFT + TrOCR 추론 파이프라인
# -------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = "image"           # 입력 이미지 폴더
    result_dir = "results"        # 결과 저장 폴더
    os.makedirs(result_dir, exist_ok=True)

    # -------------------- CRAFT 모델 --------------------
    from craft import CRAFT
    craft_model = CRAFT()
    ckpt = torch.load("craft_checkpoints/CRAFT_clr_amp_5000.pth", map_location=device)
    craft_model.load_state_dict(copy_state_dict(ckpt['craft']))
    craft_model.to(device).eval()

    # -------------------- TrOCR 모델 --------------------
    tokenizer = JamoTokenizer()
    vocab_size = tokenizer.vocab_size
    model = JamoTrOCR(vocab_size=vocab_size, d_model=768)
    checkpoint = torch.load("checkpoints/epoch_130.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    # -------------------- 이미지 전처리 --------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 한글 폰트 설정
    font = ImageFont.truetype(font_path, 15)

    # -------------------- 이미지 목록 불러오기 --------------------
    image_paths = glob.glob(os.path.join(image_dir, "*.*"))
    total_times, craft_times, trocr_times = [], [], []

    # -------------------- 이미지별 처리 루프 --------------------
    for image_path in image_paths:
        print(f"\nProcessing {image_path} ...")
        total_start = time.time()
        image_orig = cv2.imread(image_path)
        if image_orig is None:
            continue

        # 이미지 크기 조정
        img_resized, ratio_h, ratio_w = resize_aspect_ratio(image_orig)
        x = normalize_image(img_resized).to(device)

        # ---- CRAFT 실행 (텍스트 박스 탐지) ----
        craft_start = time.time()
        with torch.no_grad():
            y, _ = craft_model(x)
        score_text = y[0, :, :, 0].cpu().numpy()
        score_link = y[0, :, :, 1].cpu().numpy()
        boxes = get_text_boxes(score_text, score_link)
        craft_end = time.time()
        craft_elapsed = craft_end - craft_start

        # ---- 결과 시각화를 위한 PIL 객체 생성 ----
        image_pil = Image.fromarray(cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        # ---- TrOCR 배치 인식 ----
        trocr_start = time.time()
        crops, box_list = [], []
        for box in boxes:
            # 박스 확장 및 좌표 보정
            box = box / ratio_w * 2
            box = expand_box_xy(box, image_shape=image_orig.shape)
            xmin = max(int(box[:, 0].min()), 0)
            xmax = min(int(box[:, 0].max()), image_orig.shape[1] - 1)
            ymin = max(int(box[:, 1].min()), 0)
            ymax = min(int(box[:, 1].max()), image_orig.shape[0] - 1)
            crop_img = image_orig[ymin:ymax, xmin:xmax]
            if crop_img.size == 0:
                continue

            # crop된 이미지를 PIL로 변환 후 전처리
            image_crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            crops.append(transform(image_crop_pil))
            box_list.append(box)

        # ---- 배치 단위 OCR 수행 ----
        if len(crops) > 0:
            image_batch = torch.stack(crops, dim=0).to(device)
            token_ids_list = generate_batch(model, image_batch, tokenizer)
            for tokens, box in zip(token_ids_list, box_list):
                final_text = join_jamos(tokenizer.decode(tokens))  # 자모 → 완성형 변환
                draw.polygon([tuple(pt) for pt in box], outline="red", width=3)
                text_x, text_y = int(box[0][0]), int(box[0][1]) - 20
                draw.text((text_x, text_y), final_text, font=font, fill=(255, 255, 0))

        trocr_end = time.time()
        trocr_elapsed = trocr_end - trocr_start

        # ---- 결과 저장 ----
        filename = os.path.basename(image_path)
        save_path = os.path.join(result_dir, filename)
        image_pil.save(save_path)

        # ---- 실행 시간 기록 ----
        total_end = time.time()
        total_elapsed = total_end - total_start
        total_times.append(total_elapsed)
        craft_times.append(craft_elapsed)
        trocr_times.append(trocr_elapsed)

        print(f"CRAFT time: {craft_elapsed:.2f} s | TrOCR time: {trocr_elapsed:.2f} s | Total: {total_elapsed:.2f} s")
        print(f"Saved result to {save_path}")

    # ---- 평균 수행 시간 출력 ----
    if total_times:
        print(f"\n✅ Average per image:")
        print(f"    CRAFT: {sum(craft_times)/len(craft_times):.2f} s")
        print(f"    TrOCR: {sum(trocr_times)/len(trocr_times):.2f} s")
        print(f"    Total: {sum(total_times)/len(total_times):.2f} s")


# -------------------------------------------------------------
# 프로그램 실행
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
