"""폴더 일괄 검사와 결과 저장(CSV).

여러 이미지를 한 번에 검사하고, 그 결과를 CSV 파일로 저장하는 기능을 모아
CLI 와 GUI 가 함께 사용합니다.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from .dataset import find_images
from .heatmap import defect_heatmap
from .model import ImageInspector

# 분류 이름으로 양호/불량을 자동 판단하기 위한 단어 모음
GOOD_WORDS = {"ok", "good", "pass", "양품", "정상", "합격"}
BAD_WORDS = {"ng", "bad", "fail", "defect", "불량", "결함", "불합격"}


def resolve_defect_classes(class_names: list[str], explicit: str | None = None) -> set[str]:
    """어떤 분류가 '불량'인지 결정한다.

    - explicit 이 주어지면 그 이름을 불량으로 본다.
    - 아니면 이름에 ng/불량 등이 들어간 분류를 불량으로 본다.
    - '양품' 계열만 있으면 나머지를 불량으로 본다.
    - 판단할 수 없으면 빈 집합(불량 구분 불가)을 돌려준다.
    """
    if explicit:
        return {explicit}
    bad = {c for c in class_names if c.lower() in BAD_WORDS}
    if bad:
        return bad
    good = {c for c in class_names if c.lower() in GOOD_WORDS}
    if good:
        return set(class_names) - good
    return set()


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


def export_results(
    inspector: ImageInspector,
    folder: str | Path,
    out_dir: str | Path,
    recursive: bool = True,
    make_heatmap: bool = True,
    heatmap_all: bool = False,
    defect_class: str | None = None,
    grid: int = 10,
    progress=None,
) -> dict:
    """폴더를 검사해 결과를 out_dir 아래에 정리해서 저장한다.

    만들어지는 구조::

        out_dir/
        ├── 불량/          <- 불량으로 판정된 원본 이미지 복사본
        ├── 히트맵/        <- 불량 판정 근거를 표시한 히트맵 이미지
        └── 검사결과.csv   <- 전체 검사 결과 요약표

    make_heatmap : 히트맵 생성 여부
    heatmap_all  : True 면 모든 이미지, False 면 불량 이미지만 히트맵 생성
    defect_class : 불량으로 볼 분류 이름(미지정 시 자동 판단)
    grid         : 히트맵 격자 세밀도
    progress     : 진행 알림 콜백 progress(done, total, message) (선택)

    반환: 요약 dict (counts, ng_copied, heatmaps, out_dir, csv 등)
    """
    out = Path(out_dir)
    ng_dir = out / "불량"
    heat_dir = out / "히트맵"

    results = inspect_folder(inspector, folder, recursive=recursive)
    defect_classes = resolve_defect_classes(inspector.class_names, defect_class)

    # 히트맵을 만들 대상 결정
    def is_defect(row: dict) -> bool:
        if row["error"]:
            return False
        if defect_classes:
            return row["label"] in defect_classes
        return False

    targets = [r for r in results if not r["error"] and (heatmap_all or is_defect(r))]

    total_steps = len(targets) + 1  # 히트맵들 + CSV 저장
    done = 0

    ng_copied = 0
    heatmaps = 0

    for row in results:
        if is_defect(row):
            ng_dir.mkdir(parents=True, exist_ok=True)
            src = Path(row["path"])
            shutil.copy2(src, ng_dir / src.name)
            ng_copied += 1

    if make_heatmap:
        for row in targets:
            src = Path(row["path"])
            # 히트맵이 설명할 대상: 불량 분류(있으면) 아니면 예측 분류
            target_name = row["label"]
            if defect_classes:
                # 불량 분류 중 확신도가 가장 높은 이름을 대상으로
                dc = [(c, row["confidences"].get(c, 0)) for c in defect_classes]
                target_name = max(dc, key=lambda kv: kv[1])[0]
            try:
                overlay, _ = defect_heatmap(inspector, src, target_name=target_name, grid=grid)
                heat_dir.mkdir(parents=True, exist_ok=True)
                overlay.save(heat_dir / f"{src.stem}_heatmap.png")
                heatmaps += 1
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"히트맵 실패: {exc}"
            done += 1
            if progress:
                progress(done, total_steps, f"히트맵 생성 {done}/{len(targets)}")

    # CSV 저장
    csv_path = out / "검사결과.csv"
    out.mkdir(parents=True, exist_ok=True)
    save_csv(results, csv_path, inspector.class_names)
    done += 1
    if progress:
        progress(done, total_steps, "CSV 저장 완료")

    stats = summarize(results)
    stats.update(
        {
            "out_dir": str(out),
            "ng_dir": str(ng_dir),
            "heat_dir": str(heat_dir),
            "csv": str(csv_path),
            "ng_copied": ng_copied,
            "heatmaps": heatmaps,
            "defect_classes": sorted(defect_classes),
            "results": results,
        }
    )
    return stats
