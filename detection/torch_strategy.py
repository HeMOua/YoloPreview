import cv2
from typing import List, Dict, Any
from ultralytics import YOLO
from detection.strategy_base import DetectionStrategy, DetectionResult, BoundingBox


class PyTorchDetectionStrategy(DetectionStrategy):
    """PyTorch YOLO检测策略"""

    def __init__(self):
        self.model = None
        self.model_path = None

    def load_model(self, model_path: str) -> bool:
        """加载PyTorch模型"""
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            return True
        except Exception as e:
            raise RuntimeError(f"PyTorch模型加载失败: {str(e)}")

    def predict(self, image_path: str, conf_threshold: float, iou_threshold: float) -> DetectionResult:
        """使用PyTorch模型执行预测"""
        if self.model is None:
            raise RuntimeError("模型未加载")

        # 读取图像信息
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"无法读取图像: {image_path}")

        image_shape = image.shape

        # 执行推理
        results = self.model(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        result = results[0]
        boxes = []

        # 转换检测结果
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_np = result.boxes.xyxy.cpu().numpy()
            cls_np = result.boxes.cls.cpu().numpy()
            conf_np = result.boxes.conf.cpu().numpy()

            for i in range(len(boxes_np)):
                x1, y1, x2, y2 = boxes_np[i]
                class_id = int(cls_np[i])
                confidence = float(conf_np[i])
                class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"

                bbox = BoundingBox(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    confidence=confidence, class_id=class_id, class_name=class_name
                )
                boxes.append(bbox)

        return DetectionResult(
            image_path=image_path,
            image_shape=image_shape,
            boxes=boxes,
            model_info=self.get_model_info(),
            detection_params={
                'confidence_threshold': conf_threshold,
                'iou_threshold': iou_threshold
            }
        )

    def get_model_info(self) -> Dict[str, Any]:
        """获取PyTorch模型信息"""
        if self.model is None:
            return {"status": "未加载"}

        info = {
            "model_path": self.model_path,
            "model_type": "PyTorch (.pt)",
            "engine": "ultralytics"
        }

        if hasattr(self.model, 'names'):
            info["class_count"] = len(self.model.names)
            info["classes"] = list(self.model.names.values())

        return info

    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式"""
        return ['.pt']
