"""검사 결과 히트맵 생성.

"이 이미지의 어느 부분 때문에 불량으로 판정했는가?" 를 색으로 보여 줍니다.

방법: **occlusion(가림) 민감도**
  이미지를 격자로 나눠, 한 칸씩 평균색으로 가려 본 뒤 다시 검사합니다.
  어떤 칸을 가렸더니 '불량' 확신도가 크게 떨어졌다면, 그 칸이 바로
  불량 판정에 크게 기여한 부위입니다. 그런 부위일수록 빨갛게 칠합니다.

이 방식은 딥러닝이 아니어도(랜덤포레스트 등) 모든 분류기에 적용됩니다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .model import ImageInspector

# 히트맵을 계산할 작업 해상도(너무 크면 느려짐)
WORK_SIZE = 192


def defect_heatmap(
    inspector: ImageInspector,
    image: str | Path | Image.Image,
    target_name: str | None = None,
    grid: int = 10,
    max_alpha: float = 0.6,
):
    """이미지 한 장의 히트맵 오버레이 이미지를 만든다.

    inspector   : 학습된 모델
    image       : 이미지 경로 또는 PIL 이미지
    target_name : 어떤 분류의 확신도를 설명할지(보통 '불량' 클래스).
                  None 이면 예측된 분류를 대상으로 한다.
    grid        : 가림 격자 칸 수(한 변). 클수록 세밀하지만 느림.
    max_alpha   : 빨간 오버레이의 최대 진하기(0~1)

    반환: (오버레이된 PIL 이미지, grid×grid 민감도 배열)
    """
    if isinstance(image, (str, Path)):
        original = Image.open(image).convert("RGB")
    else:
        original = image.convert("RGB")

    work = original.resize((WORK_SIZE, WORK_SIZE))
    work_arr = np.asarray(work, dtype=np.uint8)
    mean_color = work_arr.reshape(-1, 3).mean(axis=0).astype(np.uint8)

    # 기준 확신도
    label, base_conf = inspector.predict(work)
    if target_name is None or target_name not in base_conf:
        target_name = label
    baseline = base_conf.get(target_name, 0.0)

    # 격자별로 가려 보며 확신도 하락(=기여도) 측정
    cell = max(1, WORK_SIZE // grid)
    heat = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            occluded = work_arr.copy()
            y0, y1 = i * cell, min((i + 1) * cell, WORK_SIZE)
            x0, x1 = j * cell, min((j + 1) * cell, WORK_SIZE)
            occluded[y0:y1, x0:x1] = mean_color
            _, conf = inspector.predict(Image.fromarray(occluded))
            drop = baseline - conf.get(target_name, 0.0)
            heat[i, j] = max(0.0, drop)  # 양수만: 불량쪽으로 기여한 부위

    # 0~1 로 정규화
    if heat.max() > 0:
        heat = heat / heat.max()

    overlay_img = _overlay(original, heat, max_alpha=max_alpha)
    return overlay_img, heat


def _overlay(original: Image.Image, heat: np.ndarray, max_alpha: float) -> Image.Image:
    """격자 민감도(heat)를 원본 이미지 위에 부드러운 빨간 히트맵으로 얹는다."""
    # 격자 해상도를 원본 크기로 부드럽게 확대
    small = Image.fromarray((heat * 255).astype(np.uint8))
    heat_full = small.resize(original.size, Image.BICUBIC)
    heat_arr = np.asarray(heat_full, dtype=np.float32) / 255.0

    base = np.asarray(original, dtype=np.float32)
    alpha = (np.power(heat_arr, 0.8) * max_alpha)[..., None]  # 강한 곳일수록 진하게
    red = np.array([255.0, 0.0, 0.0])
    blended = base * (1 - alpha) + red * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
