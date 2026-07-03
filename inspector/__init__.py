"""이미지 검사(양/불 분류) 프로그램.

폴더에 정리된 이미지를 학습해서 새 이미지가 어떤 분류에 속하는지
(예: 양품/불량) 판별하는 간단한 이미지 검사기입니다.
"""

from .dataset import load_dataset, image_to_features
from .model import ImageInspector

__all__ = ["load_dataset", "image_to_features", "ImageInspector"]
__version__ = "0.1.0"
