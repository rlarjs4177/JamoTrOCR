# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# 자모 기반 TrOCR(JamoTrOCR) 모델의 추론(inference) 코드
# - crop 폴더 내의 이미지들을 순차적으로 불러와 인식 수행
# - Greedy Decoding 방식으로 문장 생성
# - 결과를 pred.txt 파일로 저장
# -------------------------------------------------------------

import os
import torch
from PIL import Image
from torchvision import transforms
from model.trocr_model import JamoTrOCR
from tokenizer.jamo_tokenizer import JamoTokenizer
from tqdm import tqdm
import torch.nn as nn
import re

# -------------------------------------------------------------
# 1. 디바이스 설정
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CUDA 사용 가능 시 GPU 사용, 아니면 CPU로 처리

# -------------------------------------------------------------
# 2. 자모 토크나이저 로드
# -------------------------------------------------------------
tokenizer = JamoTokenizer()
vocab_size = tokenizer.vocab_size          # 전체 자모 토큰 개수
pad_token_id = tokenizer.pad_token_id      # PAD 토큰 ID

# -------------------------------------------------------------
# 3. 모델 로드
# -------------------------------------------------------------
model = JamoTrOCR(vocab_size=vocab_size, d_model=768)  # TrOCR 모델 초기화
checkpoint_path = "checkpoints/epoch_130.pt"           # 가중치 경로
# 학습된 파라미터 불러오기
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()  # 추론 모드 설정 (Dropout/BN 비활성화)

# -------------------------------------------------------------
# 4. 이미지 전처리 파이프라인
# -------------------------------------------------------------
# 모델 입력 크기에 맞춰 224x224로 리사이즈하고 텐서 변환 및 정규화 수행
transform = transforms.Compose([
    transforms.Resize((224, 224)),         # 입력 크기 조정
    transforms.ToTensor(),                 # Tensor 변환 (0~1)
    transforms.Normalize(mean=[0.5], std=[0.5]),  # 정규화 (-1~1)
])

# -------------------------------------------------------------
# 5. Greedy Decoding 함수
# -------------------------------------------------------------
# 한 이미지에서 토큰을 순차적으로 생성하며 가장 높은 확률 토큰 선택
def generate(model, image_tensor, tokenizer, max_length=128):
    """
    Args:
        model: 학습된 JamoTrOCR 모델
        image_tensor: 전처리된 이미지 텐서 (1, 3, 224, 224)
        tokenizer: 자모 기반 토크나이저
        max_length: 최대 생성 길이
    Returns:
        outputs: 예측된 토큰 ID 시퀀스
    """
    with torch.no_grad():  # 추론 중 그래디언트 비활성화
        # (1) 인코더 처리
        enc_outputs = model.encoder(pixel_values=image_tensor)
        memory = model.enc_to_dec(enc_outputs.last_hidden_state)  # 인코더 출력 차원 맞춤
        memory = memory.transpose(0, 1)  # (src_len, batch, hidden)

        # (2) 디코더 시작: <BOS> 토큰으로 시작
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        outputs = []

        # (3) 한 토큰씩 순차적으로 예측
        for _ in range(max_length):
            # 디코더 입력 임베딩 + 위치 임베딩
            tgt_emb = model.embedding(input_ids)
            pos_emb = model.pos_embedding[:tgt_emb.size(1), :].unsqueeze(0)
            tgt_emb = tgt_emb + pos_emb
            tgt_emb = tgt_emb.transpose(0, 1)  # (tgt_len, batch, dim)

            # causal mask 생성 (미래 토큰 참조 방지)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(device)

            # 디코더 실행
            dec_out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            dec_out = dec_out.transpose(0, 1)  # (batch, tgt_len, dim)

            # 마지막 토큰의 로짓(logit)에서 argmax로 다음 토큰 선택
            logits = model.fc_out(dec_out[:, -1, :])
            next_token = logits.argmax(dim=-1).item()

            # <EOS> 토큰 예측 시 종료
            if next_token == tokenizer.eos_token_id:
                break

            # 출력 시퀀스에 토큰 추가
            outputs.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

    return outputs


# -------------------------------------------------------------
# 6. 파일명에 포함된 숫자를 기준으로 정렬하기 위한 함수
# -------------------------------------------------------------
def extract_number(path):
    """
    파일 이름에서 숫자를 추출하여 정렬용 기준으로 반환
    예: img_12.jpg → 12
    """
    filename = os.path.splitext(os.path.basename(path))[0]
    number = re.findall(r'\d+', filename)
    return int(number[0]) if number else -1  # 숫자 없으면 맨 앞으로 정렬


# -------------------------------------------------------------
# 7. 이미지 경로 수집 및 숫자 기준 정렬
# -------------------------------------------------------------
image_folder = "crop"
# crop 폴더 내 모든 이미지 파일 경로 수집
image_paths = sorted([
    os.path.join(image_folder, fname)
    for fname in os.listdir(image_folder)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
], key=extract_number)


# -------------------------------------------------------------
# 8. 예측 수행 및 결과 저장
# -------------------------------------------------------------
output_path = "pred.txt"
with open(output_path, "w", encoding="utf-8") as f:
    # tqdm으로 진행 상황 표시
    for img_path in tqdm(image_paths, desc="Predicting"):
        try:
            # 이미지 로드 및 RGB 변환
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ 이미지 열기 실패: {img_path}, 오류: {e}")
            continue

        # 이미지 전처리 후 배치 차원 추가
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Greedy decoding 수행
        token_ids = generate(model, image_tensor, tokenizer)

        # 토큰 ID → 문자열 디코딩
        pred_text = tokenizer.decode(token_ids)
        filename = os.path.basename(img_path)

        # 결과 파일에 "파일명\t예측문장" 형식으로 저장
        f.write(f"{filename}\t{pred_text}\n")
