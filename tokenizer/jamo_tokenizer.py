# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# 한글 자모 단위 토크나이저 (JamoTokenizer)
# - 한글 완성형 문자를 초성/중성/종성(자모) 단위로 분해 및 재조합
# - BOS, EOS, PAD 등 특수 토큰 포함
# - 숫자, 알파벳, 특수문자 등도 추가 토큰으로 포함
# - encode(): 텍스트 → 토큰 ID 시퀀스 변환
# - decode(): 토큰 ID 시퀀스 → 텍스트 복원
# -------------------------------------------------------------

from collections import OrderedDict

# -------------------------------------------------------------
# 초성(Chosung) 리스트 (19개)
# -------------------------------------------------------------
CHOSUNG_LIST = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

# -------------------------------------------------------------
# 중성(Jungsung) 리스트 (21개)
# -------------------------------------------------------------
JUNGSUNG_LIST = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ',
    'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ',
    'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]

# -------------------------------------------------------------
# 종성(Jongsung) 리스트 (28개, 공백 포함)
# -------------------------------------------------------------
JONGSUNG_LIST = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ',
    'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

# -------------------------------------------------------------
# 특수 토큰 정의
# -------------------------------------------------------------
BOS_TOKEN = '[BOS]'   # 문장 시작 (Beginning Of Sequence)
EOS_TOKEN = '[EOS]'   # 문장 종료 (End Of Sequence)
PAD_TOKEN = '[PAD]'   # 패딩 토큰 (길이 맞추기용)
SPC_TOKEN = '[SPC]'   # 공백(space) 토큰
NUL_TOKEN = '[NUL]'   # 종성이 없는 글자 표시
UNK_TOKEN = '[UNK]'   # 미등록(Unknown) 토큰

# -------------------------------------------------------------
# JamoTokenizer 클래스
# -------------------------------------------------------------
class JamoTokenizer:
    def __init__(self):
        # 특수 토큰 리스트
        self.special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SPC_TOKEN, NUL_TOKEN, UNK_TOKEN]

        # 숫자, 알파벳, 특수문자 등 추가 토큰 (OCR에서 자주 등장)
        self.extra_tokens = (
            [str(i) for i in range(10)] +  # 0~9 숫자
            list("abcdefghijklmnopqrstuvwxyz") +  # 소문자 알파벳
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +  # 대문자 알파벳
            list(".,-!?()[]:;\"'/\\@#$%^&*_+=~`<>|{}※☆★●◆■▲◁◀▷▶♠♡♥♣☎☞")  # 특수기호
        )

        # -----------------------------------------------------
        # 전체 토큰을 순서대로 등록 (중복 제거 위해 OrderedDict 사용)
        # -----------------------------------------------------
        seen = OrderedDict()
        for token in (
            self.special_tokens +
            CHOSUNG_LIST +
            JUNGSUNG_LIST +
            JONGSUNG_LIST[1:] +  # 공백('') 제외
            self.extra_tokens
        ):
            if token not in seen:
                seen[token] = len(seen)

        # 최종 어휘 사전 생성
        self.vocab = list(seen.keys())
        self.token2id = dict(seen)
        self.id2token = {i: t for t, i in self.token2id.items()}

        # -----------------------------------------------------
        # 주요 토큰 ID 속성화
        # -----------------------------------------------------
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.spc_token = SPC_TOKEN
        self.nul_token = NUL_TOKEN
        self.unk_token = UNK_TOKEN

        self.bos_token_id = self.token2id[BOS_TOKEN]
        self.eos_token_id = self.token2id[EOS_TOKEN]
        self.pad_token_id = self.token2id[PAD_TOKEN]
        self.spc_token_id = self.token2id[SPC_TOKEN]
        self.nul_token_id = self.token2id[NUL_TOKEN]
        self.unk_token_id = self.token2id[UNK_TOKEN]

    # ---------------------------------------------------------
    # 어휘 사전 크기 반환
    # ---------------------------------------------------------
    @property
    def vocab_size(self):
        return len(self.vocab)

    # ---------------------------------------------------------
    # _decompose_hangul
    # ---------------------------------------------------------
    # 한글 완성형 문자를 초성·중성·종성(자모)으로 분리
    # 유니코드 범위(0xAC00~0xD7A3)를 이용한 수학적 분해 방식 사용
    # ---------------------------------------------------------
    def _decompose_hangul(self, syllable):
        code = ord(syllable)

        # 한글 완성형 범위가 아니면 그대로 반환
        if not (0xAC00 <= code <= 0xD7A3):
            return [syllable] if syllable in self.token2id else [UNK_TOKEN]

        # 초성, 중성, 종성 인덱스 계산
        base = code - 0xAC00
        jong_idx = base % 28
        jung_idx = ((base - jong_idx) // 28) % 21
        cho_idx = ((base - jong_idx) // 28) // 21

        cho = CHOSUNG_LIST[cho_idx]
        jung = JUNGSUNG_LIST[jung_idx]
        jong = JONGSUNG_LIST[jong_idx]

        # 종성이 없을 경우 [NUL] 추가
        return [cho, jung, NUL_TOKEN] if jong == '' else [cho, jung, jong]

    # ---------------------------------------------------------
    # encode
    # ---------------------------------------------------------
    # 텍스트 → 토큰 ID 시퀀스 변환
    # - 한글은 자모 단위로 분해
    # - 공백은 [SPC], 미등록 문자는 [UNK]
    # - BOS, EOS 토큰 자동 추가
    # ---------------------------------------------------------
    def encode(self, text, add_special_tokens=True):
        tokens = []

        if add_special_tokens:
            tokens.append(BOS_TOKEN)

        for char in text:
            if char == ' ':
                tokens.append(SPC_TOKEN)
            elif char in self.token2id:
                tokens.append(char)
            else:
                decomposed = self._decompose_hangul(char)
                tokens.extend(decomposed if decomposed else [UNK_TOKEN])

        if add_special_tokens:
            tokens.append(EOS_TOKEN)

        # 토큰을 ID 시퀀스로 변환 (없는 토큰은 UNK로 대체)
        return [self.token2id.get(t, self.unk_token_id) for t in tokens]

    # ---------------------------------------------------------
    # decode
    # ---------------------------------------------------------
    # 토큰 ID 시퀀스 → 텍스트 복원
    # - [BOS], [EOS], [PAD] 제거
    # - [SPC] → 공백
    # - 자모 3개 단위로 한글 결합
    # ---------------------------------------------------------
    def decode(self, token_ids):
        tokens = [self.id2token.get(i, UNK_TOKEN) for i in token_ids]

        # 불필요한 특수 토큰 제거 및 변환
        filtered = []
        for t in tokens:
            if t in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
                continue
            elif t == SPC_TOKEN:
                filtered.append(' ')
            elif t == UNK_TOKEN:
                filtered.append('�')  # UNK 문자 표시
            else:
                filtered.append(t)

        # -----------------------------------------------------
        # 내부 함수: 자모 3개를 결합하여 완성형 문자로 복원
        # -----------------------------------------------------
        def compose(jamos):
            try:
                cho = CHOSUNG_LIST.index(jamos[0])
                jung = JUNGSUNG_LIST.index(jamos[1])
                jong = 0 if jamos[2] == NUL_TOKEN else JONGSUNG_LIST.index(jamos[2])
                return chr(0xAC00 + (cho * 21 * 28) + (jung * 28) + jong)
            except:
                return ''.join(jamos)  # 예외 시 원형 반환

        # -----------------------------------------------------
        # 자모를 모아 한글 복원 (buffer 방식)
        # -----------------------------------------------------
        result = []
        buffer = []

        for t in filtered:
            if t == ' ':
                # 공백 전 버퍼 처리 후 공백 삽입
                if len(buffer) == 3:
                    result.append(compose(buffer))
                    buffer = []
                elif buffer:
                    result.extend(buffer)
                    buffer = []
                result.append(' ')
            elif t in CHOSUNG_LIST or t in JUNGSUNG_LIST or t in JONGSUNG_LIST or t == NUL_TOKEN:
                # 자모 버퍼에 저장
                buffer.append(t)
                if len(buffer) == 3:
                    result.append(compose(buffer))
                    buffer = []
            else:
                # 자모 외의 문자 처리 (숫자, 알파벳 등)
                if buffer:
                    if len(buffer) == 3:
                        result.append(compose(buffer))
                    else:
                        result.extend(buffer)
                    buffer = []
                result.append(t)

        # 남은 버퍼 처리
        if buffer:
            if len(buffer) == 3:
                result.append(compose(buffer))
            else:
                result.extend(buffer)

        return ''.join(result)
