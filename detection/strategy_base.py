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

