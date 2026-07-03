"""폴더 일괄 검사와 결과 저장(CSV).

여러 이미지를 한 번에 검사하고, 그 결과를 CSV 파일로 저장하는 기능을 모아
CLI 와 GUI 가 함께 사용합니다.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .dataset import find_images
from .model import ImageInspector


def inspect_folder(
    inspector: ImageInspector,
    folder: str | Path,
    recursive: bool = True,
) -> list[dict]:
    """폴더 안의 모든 이미지를 검사해 결과 목록을 돌려준다.

    각 결과는 다음 형태의 dict::

        {
          "path": "data/test/ng/ng_000.png",
          "label": "ng",             # 판정
          "confidences": {"ng": 0.74, "ok": 0.26},
          "error": None,             # 검사 실패 시 오류 메시지
        }
    """
    results: list[dict] = []
    for image_path in find_images(folder, recursive=recursive):
        row: dict = {"path": str(image_path), "label": None, "confidences": {}, "error": None}
        try:
            label, confidences = inspector.predict(image_path)
            row["label"] = label
            row["confidences"] = confidences
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
        results.append(row)
    return results


def summarize(results: list[dict]) -> dict:
    """판정 결과별 개수를 세어 돌려준다. 예: {'ok': 8, 'ng': 8}."""
    counts: dict[str, int] = {}
    errors = 0
    for row in results:
        if row["error"]:
            errors += 1
            continue
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    counts_sorted = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    return {"counts": counts_sorted, "errors": errors, "total": len(results)}


def save_csv(results: list[dict], csv_path: str | Path, class_names: list[str]) -> None:
    """검사 결과를 CSV 파일로 저장한다.

    엑셀에서 한글이 깨지지 않도록 UTF-8(BOM) 로 저장한다.
    열: 파일명, 경로, 판정, (분류별 확신도...), 오류
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["파일명", "경로", "판정"] + [f"{name}(%)" for name in class_names] + ["오류"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in results:
            conf = row["confidences"]
            confs = [f"{conf.get(name, 0) * 100:.1f}" for name in class_names]
            writer.writerow(
                [Path(row["path"]).name, row["path"], row["label"] or ""]
                + confs
                + [row["error"] or ""]
            )
