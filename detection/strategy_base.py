"""
检测策略模块 - 使用策略模式支持多种YOLO推理引擎
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class BoundingBox:
    """统一的边界框数据结构"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionResult:
    """统一的检测结果数据结构"""
    image_path: str
    image_shape: Tuple[int, int, int]  # (height, width, channels)
    boxes: List[BoundingBox]
    model_info: Dict[str, Any]
    detection_params: Dict[str, float]

    def extract_visualization_data(self) -> List[Dict]:
        """
        提取用于绘图的数据，方便后续绘图函数使用。
        返回每个 box 的基本信息，含坐标、颜色、标签等。
        """

        unique_class_ids = list({box.class_id for box in self.boxes})
        np.random.seed(42)
        color_map = {
            cls_id: tuple(int(c) for c in np.random.randint(0, 240, 3))
            for cls_id in unique_class_ids
        }

        result = []
        for box in self.boxes:
            result.append({
                "box": (int(box.x1), int(box.y1), int(box.x2), int(box.y2)),
                "label": f"{box.class_name} {box.confidence:.2f}",
                "color": color_map[box.class_id]
            })
        return result


class DetectionStrategy(ABC):
    """检测策略抽象基类"""

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """加载模型"""
        pass

    @abstractmethod
    def predict(self, image_path: str, conf_threshold: float, iou_threshold: float) -> DetectionResult:
        """执行预测"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式"""
        pass

