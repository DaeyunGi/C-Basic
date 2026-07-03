"""이미지 검사 모델의 학습 / 저장 / 불러오기 / 판별.

scikit-learn 의 RandomForest 분류기를 사용합니다. 트리 기반이라
픽셀값처럼 스케일이 제각각인 특징도 별도 정규화 없이 잘 다루고,
데이터가 적어도 비교적 안정적으로 동작합니다.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from .dataset import DEFAULT_SIZE, image_to_features, load_dataset


class ImageInspector:
    """이미지 검사기.

    학습(train)으로 모델을 만들고, 예측(predict)으로 새 이미지를 판별합니다.
    save/load 로 학습한 모델을 파일에 저장하고 다시 불러올 수 있습니다.
    """

    def __init__(self, size: int = DEFAULT_SIZE, n_estimators: int = 200):
        self.size = size
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self.class_names: list[str] = []

    # ------------------------------------------------------------------ #
    # 학습
    # ------------------------------------------------------------------ #
    def train(self, data_dir: str | Path):
        """학습 폴더로 모델을 학습하고 간단한 성능 지표를 반환한다."""
        X, y, class_names = load_dataset(data_dir, size=self.size)
        self.class_names = class_names

        # 데이터가 적을 때를 대비해 교차검증 fold 수를 자동 조절
        n_per_class = np.bincount(y).min()
        cv = int(min(5, n_per_class))

        accuracy = None
        if cv >= 2:
            scores = cross_val_score(self.clf, X, y, cv=cv)
            accuracy = float(scores.mean())

        self.clf.fit(X, y)

        return {
            "num_images": int(len(y)),
            "class_names": class_names,
            "per_class_counts": {class_names[i]: int(c) for i, c in enumerate(np.bincount(y))},
            "cv_accuracy": accuracy,  # 교차검증 정확도 (None 이면 데이터 부족)
        }

    # ------------------------------------------------------------------ #
    # 판별
    # ------------------------------------------------------------------ #
    def predict(self, image_path: str | Path):
        """이미지 한 장을 판별해 (예측 라벨, 확률 dict) 를 반환한다."""
        if not self.class_names:
            raise RuntimeError("학습되지 않은 모델입니다. 먼저 train() 하거나 load() 하세요.")

        features = image_to_features(image_path, size=self.size).reshape(1, -1)
        proba = self.clf.predict_proba(features)[0]
        best = int(np.argmax(proba))
        confidences = {self.class_names[i]: float(p) for i, p in enumerate(proba)}
        return self.class_names[best], confidences

    # ------------------------------------------------------------------ #
    # 저장 / 불러오기
    # ------------------------------------------------------------------ #
    def save(self, model_path: str | Path):
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"clf": self.clf, "class_names": self.class_names, "size": self.size},
            model_path,
        )

    @classmethod
    def load(cls, model_path: str | Path) -> "ImageInspector":
        data = joblib.load(model_path)
        inspector = cls(size=data["size"])
        inspector.clf = data["clf"]
        inspector.class_names = data["class_names"]
        return inspector
