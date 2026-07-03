"""데모용 샘플 이미지 생성기.

진짜 이미지가 없어도 프로그램을 바로 시험해 볼 수 있도록,
'양품(ok)' 과 '불량(ng)' 이미지를 자동으로 만들어 줍니다.

  - ok : 깨끗한 회색 판
  - ng : 회색 판 위에 흠집/얼룩(랜덤한 점과 선)이 있는 판

실제 사용 시에는 이 스크립트 대신 여러분의 실제 사진을
data/train/ok, data/train/ng 폴더에 넣으면 됩니다.

사용::

    python make_sample_data.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 128


def make_ok_image(seed: int) -> Image.Image:
    """깨끗한 판 (약간의 밝기 노이즈만 있음)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(150, 180)  # 판 밝기
    noise = rng.normal(0, 4, size=(IMG_SIZE, IMG_SIZE))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def make_ng_image(seed: int) -> Image.Image:
    """흠집/얼룩이 있는 불량 판."""
    image = make_ok_image(seed)
    draw = ImageDraw.Draw(image)
    rng = random.Random(seed)

    # 흠집 선 몇 개
    for _ in range(rng.randint(1, 3)):
        x1, y1 = rng.randint(0, IMG_SIZE), rng.randint(0, IMG_SIZE)
        x2, y2 = rng.randint(0, IMG_SIZE), rng.randint(0, IMG_SIZE)
        shade = rng.randint(40, 90)
        draw.line((x1, y1, x2, y2), fill=(shade, shade, shade), width=rng.randint(1, 3))

    # 얼룩 점 몇 개
    for _ in range(rng.randint(2, 5)):
        cx, cy = rng.randint(0, IMG_SIZE), rng.randint(0, IMG_SIZE)
        r = rng.randint(3, 10)
        shade = rng.randint(30, 80)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(shade, shade, shade))

    return image


def generate(root: str = "data", n_train: int = 40, n_test: int = 8) -> None:
    root_path = Path(root)
    plan = {
        "train": n_train,
        "test": n_test,
    }
    seed = 0
    for split, count in plan.items():
        for label, maker in (("ok", make_ok_image), ("ng", make_ng_image)):
            out_dir = root_path / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                seed += 1
                maker(seed).save(out_dir / f"{label}_{i:03d}.png")
    print(f"[완료] 샘플 이미지 생성: {root_path}/train, {root_path}/test")
    print(f"       train: 양품/불량 각 {n_train}장, test: 각 {n_test}장")


if __name__ == "__main__":
    generate()
