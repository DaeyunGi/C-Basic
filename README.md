# 이미지 검사 프로그램 (양/불 분류)

폴더에 정리해 둔 이미지를 학습해서, 새 이미지가 **양품(ok) / 불량(ng)** 중
어디에 속하는지 판별하는 간단한 이미지 검사기입니다.
양/불 2종류뿐 아니라 **폴더만 추가하면 여러 종류로도 분류**할 수 있습니다.

GPU 없이 CPU만으로 동작하며, 딥러닝 프레임워크 대신 가벼운
scikit-learn(랜덤포레스트)을 사용합니다.

## 1. 설치

```bash
pip install -r requirements.txt
```

## 2. 바로 해보기 (샘플 데이터)

실제 이미지가 없어도 데모용 이미지를 만들어 전체 과정을 시험해 볼 수 있습니다.

```bash
# (1) 샘플 이미지 생성  -> data/train, data/test 폴더가 만들어짐
python make_sample_data.py

# (2) 학습
python -m inspector train data/train -o model.joblib

# (3) 판별
python -m inspector predict model.joblib data/test/ng/ng_000.png
```

출력 예시:

```
data/test/ng/ng_000.png  ->  [ng]   (ng 74.0%, ok 26.0%)
```

## 3. 내 이미지로 사용하기

학습 폴더 안에 **분류 이름으로 하위 폴더**를 만들고, 그 안에 사진을 넣습니다.
**하위 폴더 이름이 그대로 분류 라벨**이 됩니다.

```
data/train/
├── ok/          <- 양품 사진들
│   ├── 1.jpg
│   └── 2.jpg
└── ng/          <- 불량 사진들
    ├── 3.jpg
    └── 4.jpg
```

> 3종류 이상으로 분류하고 싶으면 `scratch/`, `dent/` 처럼 폴더를 더 만들면 됩니다.

그다음 학습하고 판별합니다.

```bash
python -m inspector train data/train -o model.joblib
python -m inspector predict model.joblib 검사할이미지.jpg
```

한 번에 여러 장도 가능합니다.

```bash
python -m inspector predict model.joblib a.jpg b.jpg c.jpg
```

## 4. 명령어 옵션

| 명령 | 설명 |
|------|------|
| `train <폴더> -o <모델경로>` | 이미지 폴더로 모델 학습 후 저장 |
| `train ... --size 96` | 이미지 축소 크기 조절 (클수록 정확·느림, 기본 64) |
| `train ... --trees 300` | 랜덤포레스트 트리 수 (기본 200) |
| `predict <모델> <이미지...>` | 학습한 모델로 이미지 판별 |

## 5. 파이썬 코드에서 직접 쓰기

```python
from inspector import ImageInspector

# 학습
insp = ImageInspector()
insp.train("data/train")
insp.save("model.joblib")

# 불러와서 판별
insp = ImageInspector.load("model.joblib")
label, confidences = insp.predict("검사할이미지.jpg")
print(label, confidences)   # 예: ng {'ng': 0.74, 'ok': 0.26}
```

## 6. 동작 방식

1. **특징 추출** (`inspector/dataset.py`) — 이미지를 숫자 벡터로 변환
   - 흑백 축소 픽셀(모양) + 색상 히스토그램(색 분포) + 밝기 통계
2. **학습/판별** (`inspector/model.py`) — 랜덤포레스트 분류기
3. **명령줄** (`inspector/__main__.py`) — `train` / `predict`

## 7. 폴더 구조

```
.
├── inspector/
│   ├── __init__.py
│   ├── dataset.py      # 이미지 로딩 · 특징 추출
│   ├── model.py        # 학습 · 저장 · 판별
│   └── __main__.py     # 명령줄 인터페이스
├── make_sample_data.py # 데모용 샘플 이미지 생성
├── requirements.txt
└── README.md
```

## 8. 정확도를 높이려면

- 각 분류마다 이미지를 **많이**(수십~수백 장) 넣을수록 좋아집니다.
- 촬영 각도·밝기·배경을 일정하게 유지하면 훨씬 안정적입니다.
- `--size` 를 키우면 세밀한 흠집을 더 잘 잡지만 느려집니다.
- 더 어려운 문제라면 이후 CNN(딥러닝) 방식으로 확장할 수 있습니다.
