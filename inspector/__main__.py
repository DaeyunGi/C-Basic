"""명령줄 인터페이스.

사용 예::

    python -m inspector train data/train -o model.joblib
    python -m inspector predict model.joblib 어떤이미지.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model import ImageInspector
from .report import export_results, inspect_folder, save_csv, summarize


def _cmd_train(args: argparse.Namespace) -> int:
    print(f"[학습] 데이터 폴더: {args.data_dir}")
    inspector = ImageInspector(size=args.size, n_estimators=args.trees)
    report = inspector.train(args.data_dir)

    print(f"  - 학습 이미지 수 : {report['num_images']}장")
    print(f"  - 분류 종류      : {', '.join(report['class_names'])}")
    for name, count in report["per_class_counts"].items():
        print(f"      · {name}: {count}장")
    if report["cv_accuracy"] is not None:
        print(f"  - 교차검증 정확도: {report['cv_accuracy'] * 100:.1f}%")
    else:
        print("  - 교차검증 정확도: (데이터가 적어 생략)")

    inspector.save(args.output)
    print(f"[완료] 모델 저장: {args.output}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    if not Path(args.model).exists():
        print(f"모델 파일이 없습니다: {args.model}", file=sys.stderr)
        return 1
    inspector = ImageInspector.load(args.model)

    for image_path in args.images:
        if not Path(image_path).exists():
            print(f"[건너뜀] 이미지 없음: {image_path}", file=sys.stderr)
            continue
        label, confidences = inspector.predict(image_path)
        ranked = sorted(confidences.items(), key=lambda kv: kv[1], reverse=True)
        detail = ", ".join(f"{name} {p * 100:.1f}%" for name, p in ranked)
        print(f"{image_path}  ->  [{label}]   ({detail})")
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    if not Path(args.model).exists():
        print(f"모델 파일이 없습니다: {args.model}", file=sys.stderr)
        return 1
    inspector = ImageInspector.load(args.model)

    print(f"[일괄 검사] 폴더: {args.folder}")
    results = inspect_folder(inspector, args.folder, recursive=not args.no_recursive)
    if not results:
        print("검사할 이미지를 찾지 못했습니다.", file=sys.stderr)
        return 1

    for row in results:
        if row["error"]:
            print(f"  [오류] {row['path']}: {row['error']}")
        else:
            conf = row["confidences"].get(row["label"], 0)
            print(f"  {row['path']}  ->  [{row['label']}] ({conf * 100:.1f}%)")

    stats = summarize(results)
    summary = ", ".join(f"{name} {count}장" for name, count in stats["counts"].items())
    print(f"\n[요약] 총 {stats['total']}장  |  {summary}"
          + (f"  |  오류 {stats['errors']}장" if stats["errors"] else ""))

    if args.csv:
        save_csv(results, args.csv, inspector.class_names)
        print(f"[저장] 결과 CSV: {args.csv}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    if not Path(args.model).exists():
        print(f"모델 파일이 없습니다: {args.model}", file=sys.stderr)
        return 1
    inspector = ImageInspector.load(args.model)

    print(f"[결과 정리] 폴더: {args.folder}  ->  {args.output}")

    def progress(done, total, message):
        print(f"  ... {message}")

    stats = export_results(
        inspector,
        args.folder,
        args.output,
        recursive=not args.no_recursive,
        make_heatmap=not args.no_heatmap,
        heatmap_all=args.heatmap_all,
        defect_class=args.defect_class,
        grid=args.grid,
        progress=progress,
    )

    summary = ", ".join(f"{name} {count}장" for name, count in stats["counts"].items())
    print(f"\n[요약] 총 {stats['total']}장  |  {summary}")
    if stats["defect_classes"]:
        print(f"  - 불량 분류    : {', '.join(stats['defect_classes'])}")
    else:
        print("  - 불량 분류    : (자동 판단 불가 → --defect-class 로 지정하세요)")
    print(f"  - 불량 이미지  : {stats['ng_copied']}장  ->  {stats['ng_dir']}")
    print(f"  - 히트맵 생성  : {stats['heatmaps']}장  ->  {stats['heat_dir']}")
    print(f"  - 결과 CSV     : {stats['csv']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspector",
        description="이미지 검사(양/불 분류) 프로그램",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="이미지 폴더로 모델 학습")
    p_train.add_argument("data_dir", help="분류별 하위 폴더가 있는 학습 폴더 (예: data/train)")
    p_train.add_argument("-o", "--output", default="model.joblib", help="저장할 모델 경로")
    p_train.add_argument("--size", type=int, default=64, help="이미지 축소 크기 (기본 64)")
    p_train.add_argument("--trees", type=int, default=200, help="랜덤포레스트 트리 수 (기본 200)")
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="학습한 모델로 이미지 판별")
    p_pred.add_argument("model", help="학습한 모델 파일 (예: model.joblib)")
    p_pred.add_argument("images", nargs="+", help="판별할 이미지 파일들")
    p_pred.set_defaults(func=_cmd_predict)

    p_scan = sub.add_parser("scan", help="폴더 안 이미지를 통째로 일괄 검사")
    p_scan.add_argument("model", help="학습한 모델 파일 (예: model.joblib)")
    p_scan.add_argument("folder", help="검사할 이미지들이 들어있는 폴더")
    p_scan.add_argument("--csv", help="결과를 저장할 CSV 파일 경로 (예: 결과.csv)")
    p_scan.add_argument("--no-recursive", action="store_true", help="하위 폴더는 검사하지 않음")
    p_scan.set_defaults(func=_cmd_scan)

    p_export = sub.add_parser(
        "export", help="폴더 검사 후 불량 이미지 수집 + 히트맵 + CSV 를 결과 폴더에 저장"
    )
    p_export.add_argument("model", help="학습한 모델 파일 (예: model.joblib)")
    p_export.add_argument("folder", help="검사할 이미지 폴더")
    p_export.add_argument("-o", "--output", default="검사결과", help="결과를 저장할 폴더")
    p_export.add_argument("--no-heatmap", action="store_true", help="히트맵을 만들지 않음")
    p_export.add_argument("--heatmap-all", action="store_true", help="불량뿐 아니라 모든 이미지 히트맵 생성")
    p_export.add_argument("--defect-class", help="불량으로 볼 분류 이름 (미지정 시 자동 판단)")
    p_export.add_argument("--grid", type=int, default=10, help="히트맵 격자 세밀도 (기본 10)")
    p_export.add_argument("--no-recursive", action="store_true", help="하위 폴더는 검사하지 않음")
    p_export.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
