# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# Jamo 기반 TrOCR 학습 스크립트
# - ViT 인코더 + Transformer 디코더 구조를 가진 JamoTrOCR 학습
# - 자모 단위 토크나이저를 사용하여 한국어 문자 인식 수행
# - 학습 데이터셋과 검증 데이터셋을 분리하여 반복 학습
# - 오답 재학습(incorrect sample retraining) 전략 포함
# -------------------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from model.trocr_model import JamoTrOCR
from tokenizer.jamo_tokenizer import JamoTokenizer
import torch.optim as optim
import os
import glob
import random
import gc
import re


# -------------------------------------------------------------
# OCRDataset 클래스
# -------------------------------------------------------------
# - 이미지 파일과 텍스트 라벨을 한 쌍으로 구성
# - 토크나이저로 텍스트를 자모 단위 토큰 시퀀스로 변환
# - 디코더 입력([BOS] 제외)과 디코더 타깃([EOS] 제외)을 분리 반환
# -------------------------------------------------------------
class OCRDataset(Dataset):
    def __init__(self, img_paths, labels, tokenizer, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 이미지 로드 및 RGB 변환
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # 텍스트 라벨 인코딩 (자모 단위)
        label_text = self.labels[idx]
        label_ids = self.tokenizer.encode(label_text)

        # 디코더 입력 / 타깃 분리
        decoder_input_ids = label_ids[:-1]  # [BOS] 제외
        labels = label_ids[1:]              # [EOS] 제외

        return img, torch.tensor(decoder_input_ids), torch.tensor(labels)


# -------------------------------------------------------------
# collate_fn
# -------------------------------------------------------------
# - DataLoader에서 batch 단위로 데이터 병합 시 호출
# - 각 시퀀스 길이를 맞추기 위해 패딩(padding) 처리 수행
# -------------------------------------------------------------
def collate_fn(batch):
    imgs, dec_inp_ids, labels = zip(*batch)

    # 이미지 텐서 병합
    imgs = torch.stack(imgs)

    # 토큰 시퀀스 패딩
    dec_inp_ids = torch.nn.utils.rnn.pad_sequence(dec_inp_ids, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    # CrossEntropyLoss에서 무시해야 할 [BOS] 토큰 처리
    BOS_TOKEN_ID = tokenizer.bos_token_id
    padded_labels[padded_labels == BOS_TOKEN_ID] = -100

    return imgs, dec_inp_ids, padded_labels


# -------------------------------------------------------------
# load_data 함수
# -------------------------------------------------------------
# - 데이터 폴더에서 이미지와 라벨 파일을 불러옴
# - 파일명과 라벨 매칭 확인 및 정렬 수행
# -------------------------------------------------------------
def load_data(data_dir, tokenizer):
    label_dict = {}
    label_path = os.path.join(data_dir, 'labels.txt')

    # 라벨 파일 파싱
    with open(label_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                print(f"⚠️ 문제 있는 라인 {i}: '{line}'")
                continue
            filename, label = parts[0], parts[1]
            label_dict[filename] = label

    # 이미지 경로 수집
    img_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        img_paths.extend(glob.glob(os.path.join(data_dir, 'images', ext)))

    # 파일명 숫자 기준 정렬
    def numerical_sort_key(s):
        numbers = re.findall(r'\d+', os.path.basename(s))
        return int(numbers[0]) if numbers else -1

    img_paths = sorted(img_paths, key=numerical_sort_key)

    img_paths_sorted, labels_sorted = [], []
    for path in img_paths:
        filename = os.path.basename(path)
        if filename in label_dict:
            img_paths_sorted.append(path)
            labels_sorted.append(label_dict[filename])
        else:
            print(f"⚠️ '{filename}'에 해당하는 라벨이 없습니다.")

    assert len(img_paths_sorted) == len(labels_sorted), f"이미지 수({len(img_paths_sorted)})와 라벨 수({len(labels_sorted)})가 다릅니다."
    return img_paths_sorted, labels_sorted


# -------------------------------------------------------------
# train_step 함수
# -------------------------------------------------------------
# - 모델 학습용 단일 배치 처리
# - forward → loss 계산 → backward → optimizer 업데이트
# -------------------------------------------------------------
def train_step(model, optimizer, loss_fn, imgs, dec_inp_ids, labels, device, pad_token_id):
    model.train()
    imgs, dec_inp_ids, labels = imgs.to(device), dec_inp_ids.to(device), labels.to(device)
    optimizer.zero_grad()

    # 모델 forward
    logits = model(imgs, dec_inp_ids, pad_token_id)

    # 손실 계산
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item(), logits


# -------------------------------------------------------------
# eval_step 함수
# -------------------------------------------------------------
# - 검증(Validation) 단계에서 손실 계산
# -------------------------------------------------------------
def eval_step(model, loss_fn, imgs, dec_inp_ids, labels, device, pad_token_id):
    model.eval()
    imgs, dec_inp_ids, labels = imgs.to(device), dec_inp_ids.to(device), labels.to(device)
    with torch.no_grad():
        logits = model(imgs, dec_inp_ids, pad_token_id)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss.item()


# -------------------------------------------------------------
# main 함수 (전체 학습 파이프라인)
# -------------------------------------------------------------
def main():
    # GPU 캐시 정리
    torch.cuda.empty_cache()
    gc.collect()

    # 로그 기록 함수
    global tokenizer
    log_file = open("log.txt", "a", encoding="utf-8")
    def log(message):
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    # ------------------------- 데이터 로드 -------------------------
    data_dir = './data'
    tokenizer = JamoTokenizer()
    img_paths, labels = load_data(data_dir, tokenizer)

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # 데이터셋 및 분할
    dataset = OCRDataset(img_paths, labels, tokenizer, transform)
    total_size = len(dataset)
    indices = list(range(total_size))
    random.seed(42)
    random.shuffle(indices)
    split = int(total_size * 0.8)
    train_dataset = Subset(dataset, indices[:split])
    val_dataset = Subset(dataset, indices[split:])

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # ------------------------- 모델 설정 -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JamoTrOCR(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # 체크포인트 디렉토리 생성
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 최신 체크포인트 불러오기
    latest_checkpoint_path = os.path.join(save_dir, "latest.pt")
    if os.path.exists(latest_checkpoint_path):
        model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))
        log("✅ 학습된 모델을 불러왔습니다. 이어서 학습을 진행합니다.")
    else:
        log("🚀 새 모델로 학습을 시작합니다.")

    # ------------------------- 학습 루프 -------------------------
    epochs = 200
    for epoch in range(epochs):
        log(f"\nEpoch {epoch+1}/{epochs}")

        model.train()
        train_loss = 0.0
        wrong_samples = []

        # -------- Training --------
        for step, (imgs, dec_inp_ids, labels) in enumerate(train_loader):
            loss, logits = train_step(model, optimizer, loss_fn, imgs, dec_inp_ids, labels, device, tokenizer.pad_token_id)
            train_loss += loss

            # 오답 샘플 추적
            with torch.no_grad():
                preds = logits.argmax(dim=-1).to(device)
                labels = labels.to(device)
                mask = (labels != -100).to(device)
                wrong_mask = (preds != labels) & mask
                batch_wrong = wrong_mask.any(dim=1).cpu()
                for i in range(len(batch_wrong)):
                    if batch_wrong[i]:
                        wrong_samples.append((imgs[i].cpu(), dec_inp_ids[i].cpu(), labels[i].cpu()))

            if step % 1000 == 0:
                log(f"[Train Step {step}/{len(train_loader)}] Loss: {loss:.4f}")
        train_loss /= len(train_loader)

        # -------- 오답 재학습 --------
        if wrong_samples:
            log(f"\n오답 샘플 수: {len(wrong_samples)} → 재학습 진행")
            wrong_loader = DataLoader(wrong_samples, batch_size=8, shuffle=True, collate_fn=collate_fn)
            for imgs, dec_inp_ids, labels in wrong_loader:
                train_step(model, optimizer, loss_fn, imgs, dec_inp_ids, labels, device, tokenizer.pad_token_id)

            # 메모리 정리
            wrong_samples.clear()
            del wrong_loader
            torch.cuda.empty_cache()
            gc.collect()

        # -------- Validation --------
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, dec_inp_ids, labels in val_loader:
                loss = eval_step(model, loss_fn, imgs, dec_inp_ids, labels, device, tokenizer.pad_token_id)
                val_loss += loss
        val_loss /= len(val_loader)

        log(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # -------- Checkpoint 저장 --------
        epoch_checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        torch.save(model.state_dict(), latest_checkpoint_path)
        log(f"💾 모델 저장 완료: {epoch_checkpoint_path}")

    log_file.close()


# -------------------------------------------------------------
# 프로그램 실행
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
