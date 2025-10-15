# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# TrOCR 모델 구현 파일 (JamoTrOCR)
# - Transformer 기반 OCR 구조로, ViT 인코더 + Transformer 디코더 구성
# - 한글 자모 단위 인식을 위해 설계됨
# -------------------------------------------------------------

import torch
import torch.nn as nn
from transformers import ViTModel


# ----------------------------------------------------------
# Additive Gated Bridge
# 인코더 출력(hidden state)을 디코더 입력 차원에 맞추며
# 게이트(gate)를 통해 정보 중요도를 동적으로 조절
# ----------------------------------------------------------
class AdditiveGatedBridge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 인코더 hidden 크기를 디코더 hidden 크기로 투영
        self.proj = nn.Linear(in_dim, out_dim)
        # 게이트: 각 feature의 중요도를 0~1 사이로 조절
        self.gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        proj_x = self.proj(x)             # 선형 변환 (feature 차원 맞춤)
        gate = self.gate(x)               # 게이트 값 계산 (0~1 범위)
        return proj_x * gate + x * (1 - gate)  # 게이트 가중 합 (정보 융합)


# ----------------------------------------------------------
# Transformer Decoder Layer
# - 자기어텐션(Self-Attention)
# - 인코더-디코더 어텐션(Cross-Attention)
# - FFN(Feed-Forward Network)
# ----------------------------------------------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 자기어텐션 (디코더 내부 시퀀스 간 의존성 학습)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 인코더-디코더 어텐션 (인코더의 정보 활용)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 피드포워드 네트워크 (비선형 변환)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 정규화 및 드롭아웃
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        # ① 자기어텐션 (디코더 내 토큰 간 관계 학습)
        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)   # 잔차 연결(residual connection)
        tgt = self.norm1(tgt)             # 정규화

        # ② 인코더-디코더 어텐션 (인코더 feature 참조)
        tgt2 = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ③ FFN (비선형 변환)
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# ----------------------------------------------------------
# Transformer Decoder
# - 여러 개의 TransformerDecoderLayer를 쌓아 구성
# ----------------------------------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        # 동일한 디코더 레이어를 num_layers개 반복
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.self_attn.embed_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        # 순차적으로 모든 디코더 레이어 통과
        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        output = self.norm(output)  # 최종 정규화
        return output


# ----------------------------------------------------------
# JamoTrOCR
# - ViT 인코더 + Additive Gated Bridge + Transformer 디코더 구조
# - 한글 자모 단위 예측을 수행
# ----------------------------------------------------------
class JamoTrOCR(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_decoder_layers=8,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # ① ViT 인코더 (사전학습된 비전 트랜스포머)
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # ② Additive Gated Bridge (인코더 → 디코더 정보 연결)
        self.enc_to_dec = AdditiveGatedBridge(self.encoder.config.hidden_size, d_model)

        # ③ 디코더 입력 임베딩 (자모 토큰 ID → 벡터)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # ④ 위치 임베딩 (문장 내 토큰 순서 정보 추가)
        self.pos_embedding = nn.Parameter(torch.randn(500, d_model))

        # ⑤ Transformer 디코더 정의
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # ⑥ 출력층: 디코더 출력 → vocab 크기 차원으로 변환 (로짓)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, pixel_values, tgt_input_ids, pad_token_id):
        # ---------------------------
        # (1) 인코더 처리
        # ---------------------------
        enc_outputs = self.encoder(pixel_values=pixel_values)
        enc_hidden_states = enc_outputs.last_hidden_state  # (batch, src_seq_len, hidden)

        # Additive Gated Bridge로 차원 맞추기 + 게이트 통합
        enc_hidden_states = self.enc_to_dec(enc_hidden_states)  # (batch, src_seq_len, d_model)

        # ---------------------------
        # (2) 디코더 입력 준비
        # ---------------------------
        tgt_emb = self.embedding(tgt_input_ids)  # (batch, tgt_seq_len, d_model)

        # 위치 임베딩 추가
        seq_len = tgt_emb.size(1)
        tgt_emb = tgt_emb + self.pos_embedding[:seq_len, :].unsqueeze(0)

        # PyTorch MultiheadAttention은 (seq_len, batch, dim) 형태 요구
        tgt_emb = tgt_emb.transpose(0, 1)
        enc_hidden_states = enc_hidden_states.transpose(0, 1)

        # ---------------------------
        # (3) 마스크 생성
        # ---------------------------
        # 미래 토큰을 참조하지 않도록 사각 마스크(causal mask) 생성
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt_emb.device)

        # 패딩 위치(True) → attention에서 무시
        tgt_key_padding_mask = (tgt_input_ids == pad_token_id)

        # ---------------------------
        # (4) 디코더 수행
        # ---------------------------
        dec_outputs = self.decoder(
            tgt_emb,
            enc_hidden_states,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # (tgt_len, batch, d_model)

        # ---------------------------
        # (5) 출력 변환
        # ---------------------------
        dec_outputs = dec_outputs.transpose(0, 1)  # (batch, tgt_len, d_model)
        logits = self.fc_out(dec_outputs)           # (batch, tgt_len, vocab_size)

        # logits: 각 자모 토큰에 대한 확률 예측 값
        return logits
