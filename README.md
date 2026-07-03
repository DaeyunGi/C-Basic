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

## 화면(GUI)으로 쓰기 — 제일 쉬움 ⭐

명령어가 어렵다면 **창(GUI)** 으로 바로 검사할 수 있습니다.
파이썬에 기본 포함된 Tkinter 를 쓰므로 추가 설치가 거의 없습니다.

```bash
python gui.py
```

창이 열리면:

1. **[1. 모델 학습]** 클릭 → 학습 폴더(`data/train`) 선택 → 자동 학습·저장
   (또는 이미 만든 `model.joblib` 이 있으면 **[모델 불러오기]**)
2. **[2. 이미지 열기]** 클릭 → 검사할 이미지 선택
3. 화면에 **이미지 + 판정 결과(양품/불량)** 와 각 분류의 **확신도 막대**가 표시됩니다.
   (양품=초록, 불량=빨강)
4. **[3. 폴더 일괄 검사]** 클릭 → 폴더 선택 → 폴더 안의 모든 이미지를 한꺼번에 검사하고
   결과가 표로 나옵니다. **[결과 CSV 저장]** 으로 엑셀용 파일로 저장할 수 있습니다.
5. **[4. 불량 모으기+히트맵]** 클릭 → 검사할 폴더와 저장할 폴더 선택 →
   **불량 이미지를 따로 모으고**, **불량 판정 근거를 표시한 히트맵**까지 만들어 저장합니다.

> 처음이라면 먼저 `python make_sample_data.py` 로 샘플 이미지를 만든 뒤
> GUI 에서 `data/train` 폴더로 학습해 보세요.

아래는 명령줄(터미널)로 쓰는 방법입니다.

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

### 폴더 통째로 일괄 검사 + 결과 저장

폴더 안의 모든 이미지(하위 폴더 포함)를 한 번에 검사하고, 결과를 CSV 로 저장합니다.

```bash
python -m inspector scan model.joblib data/test --csv 검사결과.csv
```

출력 예시:

```
  data/test/ng/ng_000.png  ->  [ng] (74.0%)
  ...
[요약] 총 16장  |  ng 8장, ok 8장
[저장] 결과 CSV: 검사결과.csv
```

저장된 `검사결과.csv` 는 엑셀에서 바로 열립니다(한글 깨짐 방지). 열 구성:
`파일명, 경로, 판정, 분류별 확신도(%), 오류`

> 하위 폴더는 빼고 검사하려면 `--no-recursive` 를 붙이세요.

### 불량 이미지 모으기 + 히트맵 (결과 폴더 정리)

폴더를 검사해서 **불량 이미지를 따로 모으고**, 각 불량이 **왜 불량인지**를
빨갛게 표시한 **히트맵**까지 한 번에 만들어 결과 폴더에 정리합니다.

```bash
python -m inspector export model.joblib data/test -o 검사결과
```

만들어지는 폴더 구조:

```
검사결과/
├── 불량/          ← 불량으로 판정된 원본 이미지 복사본
├── 히트맵/        ← 불량 판정 근거를 빨갛게 표시한 히트맵 이미지
└── 검사결과.csv   ← 전체 결과 요약표
```

- 히트맵은 **occlusion(가림) 민감도** 방식입니다. 이미지를 조금씩 가려 보며
  불량 확신도가 크게 떨어지는 부위(=불량의 근거)를 찾아 빨갛게 칠합니다.
- 옵션:
  - `--heatmap-all` : 불량뿐 아니라 **모든 이미지** 히트맵 생성
  - `--no-heatmap` : 히트맵 없이 불량 수집 + CSV 만
  - `--defect-class ng` : 불량으로 볼 분류 이름 직접 지정(자동 판단이 안 될 때)
  - `--grid 14` : 히트맵 세밀도(클수록 정밀·느림, 기본 10)

## 4. 명령어 옵션

| 명령 | 설명 |
|------|------|
| `train <폴더> -o <모델경로>` | 이미지 폴더로 모델 학습 후 저장 |
| `train ... --size 96` | 이미지 축소 크기 조절 (클수록 정확·느림, 기본 64) |
| `train ... --trees 300` | 랜덤포레스트 트리 수 (기본 200) |
| `predict <모델> <이미지...>` | 학습한 모델로 이미지 판별 |
| `scan <모델> <폴더> [--csv 파일]` | 폴더 안 이미지 일괄 검사 + CSV 저장 |
| `scan ... --no-recursive` | 하위 폴더는 검사하지 않음 |
| `export <모델> <폴더> -o <결과폴더>` | 불량 이미지 모으기 + 히트맵 + CSV |
| `export ... --heatmap-all` | 모든 이미지 히트맵 생성 |
| `export ... --defect-class ng` | 불량으로 볼 분류 직접 지정 |

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
3. **일괄 검사/저장** (`inspector/report.py`) — 폴더 검사 · 불량 수집 · CSV
4. **히트맵** (`inspector/heatmap.py`) — occlusion 민감도로 불량 근거 시각화
5. **명령줄** (`inspector/__main__.py`) — `train` / `predict` / `scan` / `export`
6. **그래픽 화면** (`gui.py`) — Tkinter GUI

## 7. 폴더 구조

```
.
├── inspector/
│   ├── __init__.py
│   ├── dataset.py      # 이미지 로딩 · 특징 추출
│   ├── model.py        # 학습 · 저장 · 판별
│   ├── report.py       # 폴더 일괄 검사 · 불량 수집 · CSV
│   ├── heatmap.py      # 불량 근거 히트맵 생성
│   └── __main__.py     # 명령줄 인터페이스 (train/predict/scan/export)
├── gui.py              # 그래픽 화면(GUI)
├── make_sample_data.py # 데모용 샘플 이미지 생성
├── requirements.txt
└── README.md
```

## 8. 정확도를 높이려면

- 각 분류마다 이미지를 **많이**(수십~수백 장) 넣을수록 좋아집니다.
- 촬영 각도·밝기·배경을 일정하게 유지하면 훨씬 안정적입니다.
- `--size` 를 키우면 세밀한 흠집을 더 잘 잡지만 느려집니다.
- 더 어려운 문제라면 이후 CNN(딥러닝) 방식으로 확장할 수 있습니다.
