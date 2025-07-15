from typing import List, Dict, Any, Optional
from pathlib import Path
from detection.onnx_strategy import ONNXDetectionStrategy
from detection.strategy_base import DetectionStrategy, DetectionResult
from detection.torch_strategy import PyTorchDetectionStrategy


class DetectionStrategyFactory:
    """检测策略工厂"""

    @staticmethod
    def create_strategy(model_path: str) -> DetectionStrategy:
        """根据模型文件扩展名创建相应的策略"""
        extension = Path(model_path).suffix.lower()

        if extension == '.pt':
            return PyTorchDetectionStrategy()
        elif extension == '.onnx':
            return ONNXDetectionStrategy()
        else:
            raise ValueError(f"不支持的模型格式: {extension}")

    @staticmethod
    def get_supported_formats() -> List[str]:
        """获取所有支持的模型格式"""
        return ['.pt', '.onnx']



class DetectionContext:
    """检测上下文类"""

    def __init__(self):
        self.strategy: Optional[DetectionStrategy] = None
        self.model_path: Optional[str] = None

    def load_model(self, model_path: str) -> bool:
        """加载模型并设置相应策略"""
        try:
            self.strategy = DetectionStrategyFactory.create_strategy(model_path)
            success = self.strategy.load_model(model_path)
            if success:
                self.model_path = model_path
            return success
        except Exception as e:
            self.strategy = None
            self.model_path = None
            raise e

    def predict(self, image_path: str, conf_threshold: float, iou_threshold: float) -> DetectionResult:
        """执行预测"""
        if self.strategy is None:
            raise RuntimeError("未加载模型")

        return self.strategy.predict(image_path, conf_threshold, iou_threshold)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.strategy is None:
            return {"status": "未加载"}

        return self.strategy.get_model_info()

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.strategy is not None
