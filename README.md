# 🏷️ JamoTrOCR: Hangul OCR with Vision Transformer

## 📘 개요 (Overview)
이 저장소는 한글 자모 단위 기반 OCR 시스템인 **JamoTrOCR**의 전체 코드와 모델 구조를 포함합니다. 모델은 **Vision Transformer (ViT)** 인코더와 **Transformer 디코더(8층)** 를 결합하여, 한글 문자를 초성·중성·종성 단위로 분해해 인식하는 방식을 사용합니다. 또한 **Additive Gated Bridge** 모듈을 도입하여 인코더와 디코더 간의 표현 불일치를 완화했습니다. 텍스트 영역 검출에는 **CRAFT** 모델을 결합하여 완전한 OCR 파이프라인을 구성합니다.

---

## ⚙️ 환경 설정 (Requirements)
아래 명령어로 필요한 라이브러리를 설치하세요.
```bash
pip install -r requirements.txt
```

---

## 🧠 모델 구조 (Model Architecture)
```
[Image] 
   ↓
Vision Transformer (Encoder)
   ↓
Additive Gated Bridge
   ↓
Transformer Decoder (8 layers)
   ↓
[Jamo Token Sequence → Hangul Reconstruction]
```

---

## 🚀 학습 (Training)
학습용 데이터(`data/images/`, `data/labels.txt`)를 준비한 후 다음 명령어를 실행합니다.
```bash
python train.py
```
- 학습 시 80:20 비율로 데이터셋을 자동 분할합니다.  
- 학습 도중 오답 재학습 전략이 적용되어 성능 수렴 속도를 높입니다.  
- 학습된 모델은 `checkpoints/` 폴더에 `.pt` 파일로 저장됩니다.  

---

## 🔍 추론 (Inference)
### 1️⃣ TrOCR 단독 인식
`crop/` 폴더에 잘린 텍스트 이미지를 넣은 후 실행합니다.
```bash
python inference.py
```
### 2️⃣ CRAFT + TrOCR 통합 인식
전체 송장 이미지에서 텍스트 영역을 탐지하고 OCR을 수행합니다.
```bash
python inference_craftxtrocr.py
```
결과 이미지는 `results/` 폴더에 저장됩니다.

---

## 🧩 주요 특징 (Key Features)
- 한글 자모 단위 토크나이징 (BOS/EOS/NUL/SPC/UNK 지원)  
- ViT 인코더 + Transformer 디코더 기반 OCR  
- Additive Gated Bridge 모듈로 인코더-디코더 융합 강화  
- 오답 재학습(Incorrect Sample Retraining) 전략  
- CRAFT와의 통합 파이프라인을 통한 송장 단위 인식  

---

## 📜 라이선스 (License)
본 프로젝트는 **MIT License**를 따릅니다.  
CRAFT 모델의 일부 코드는 **NAVER Clova AI Research (MIT License)** 를 기반으로 합니다.
```
MIT License  
Copyright (c) 2025 Geon Kim
```

---

## ✍️ 인용 (Citation)
```
@article{JamoTrOCR2025,
  title={JamoTrOCR: Hangul OCR with Vision Transformer and Additive Gated Bridge},
  author={Kim, Geon},
  year={2025}
}
```

---

## 📩 문의 (Contact)
For research collaboration or questions, please contact:  
📧 **rlarjs4177@naver.com**
