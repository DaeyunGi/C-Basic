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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
