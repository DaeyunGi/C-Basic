"""이미지 로딩과 특징(feature) 추출.

이미지를 그대로 학습에 쓰기는 어렵기 때문에, 각 이미지를 고정 크기의
숫자 벡터(feature vector)로 바꿔 줍니다. 여기서는 세 가지를 이어 붙입니다.

1. 흑백으로 줄인 픽셀값     -> 전체적인 모양/구조
2. 색상 히스토그램          -> 색 분포 (얼룩, 변색 등)
3. 밝기 통계(평균/표준편차) -> 전반적인 밝기와 얼룩짐 정도

이렇게 만든 벡터를 scikit-learn 분류기에 넣어 학습/판별합니다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# 이미지를 줄일 기본 크기 (가로=세로). 값이 클수록 정확하지만 느려집니다.
DEFAULT_SIZE = 64

# 색상 히스토그램의 채널당 구간 수
HISTOGRAM_BINS = 16


def image_to_features(path_or_image, size: int = DEFAULT_SIZE) -> np.ndarray:
    """이미지 한 장을 1차원 특징 벡터로 변환한다.

    path_or_image : 이미지 파일 경로(str/Path) 또는 이미 열린 PIL.Image
    size          : 흑백 축소 시 한 변의 픽셀 수
    """
    if isinstance(path_or_image, (str, Path)):
        image = Image.open(path_or_image)
    else:
        image = path_or_image

    # 투명 배경(RGBA) 등도 안전하게 RGB 로 통일
    rgb = image.convert("RGB").resize((size, size))
    arr = np.asarray(rgb, dtype=np.float32) / 255.0  # 0~1 로 정규화

    # 1) 흑백 축소 픽셀 (구조 정보)
    gray = arr.mean(axis=2).reshape(-1)

    # 2) 색상 히스토그램 (색 분포 정보)
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=HISTOGRAM_BINS, range=(0.0, 1.0))
        hist_parts.append(hist.astype(np.float32) / (size * size))
    color_hist = np.concatenate(hist_parts)

    # 3) 밝기 통계 (평균, 표준편차)
    stats = np.array([arr.mean(), arr.std()], dtype=np.float32)

    return np.concatenate([gray, color_hist, stats])


def _iter_image_paths(folder: Path):
    """폴더 안의 이미지 파일 경로를 정렬해서 하나씩 돌려준다."""
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            yield entry


def load_dataset(root: str | Path, size: int = DEFAULT_SIZE):
    """학습 폴더를 읽어 (X, y, class_names) 를 만든다.

    폴더 구조 예시::

        root/
          ok/    <- 하위 폴더 이름이 곧 분류 라벨이 된다 (양품)
            a.png
            b.png
          ng/    <- 불량
            c.png

    반환값
      X           : (샘플수, 특징수) 형태의 numpy 배열
      y           : 각 샘플의 정수 라벨
      class_names : 정수 라벨 -> 폴더 이름 리스트
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"학습 폴더를 찾을 수 없습니다: {root}")

    class_names = [d.name for d in sorted(root.iterdir()) if d.is_dir()]
    if not class_names:
        raise ValueError(
            f"'{root}' 안에 분류용 하위 폴더가 없습니다. "
            f"예: {root}/ok, {root}/ng 처럼 만들어 주세요."
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    for label, name in enumerate(class_names):
        class_dir = root / name
        count = 0
        for image_path in _iter_image_paths(class_dir):
            try:
                features.append(image_to_features(image_path, size=size))
                labels.append(label)
                count += 1
            except Exception as exc:  # 깨진 이미지는 건너뛴다
                print(f"  [건너뜀] {image_path}: {exc}")
        if count == 0:
            raise ValueError(f"'{class_dir}' 안에 학습할 이미지가 없습니다.")

    X = np.vstack(features)
    y = np.array(labels, dtype=np.int64)
    return X, y, class_names
