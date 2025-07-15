import cv2
from typing import Dict, Any, List

from detection.strategy_base import DetectionStrategy, DetectionResult, BoundingBox
from detection.yolo_onnx import YOLO11


class ONNXDetectionStrategy(DetectionStrategy):
    """ONNX YOLO检测策略"""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.class_names = None

    def load_model(self, model_path: str, class_names=None) -> bool:
        """
        加载ONNX模型

        Args:
            model_path: ONNX模型路径
            class_names: 类别名称列表或字典，可选

        Returns:
            bool: 加载是否成功
        """
        try:
            # 使用优化后的YOLO11类
            self.model = YOLO11(
                onnx_model=model_path,
                class_names=class_names
            )
            self.model_path = model_path
            self.class_names = self.model.classes
            return True
        except Exception as e:
            raise RuntimeError(f"ONNX模型加载失败: {str(e)}")

    def predict(self, image_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45) -> DetectionResult:
        """
        使用ONNX模型执行预测

        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值

        Returns:
            DetectionResult: 检测结果
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 读取图像信息
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"无法读取图像: {image_path}")

        image_shape = image.shape

        # 更新模型阈值参数
        self.model.update_thresholds(
            confidence_thres=conf_threshold,
            iou_thres=iou_threshold
        )

        # 执行推理 - 使用新的predict方法
        result_image, detections = self.model.predict(image_path)

        # 转换检测结果为统一格式
        boxes = []
        for detection in detections:
            bbox_info = detection["bbox"]

            # 从新的bbox格式中提取坐标
            x1 = float(bbox_info["x1"])
            y1 = float(bbox_info["y1"])
            x2 = float(bbox_info["x2"])
            y2 = float(bbox_info["y2"])

            bbox = BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(detection["confidence"]),
                class_id=int(detection["class_id"]),
                class_name=detection["class_name"]
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

    def predict_batch(self, image_paths: List[str], conf_threshold: float = 0.5, iou_threshold: float = 0.45) -> List[
        DetectionResult]:
        """
        批量预测多张图像

        Args:
            image_paths: 图像路径列表
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值

        Returns:
            List[DetectionResult]: 检测结果列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, conf_threshold, iou_threshold)
                results.append(result)
            except Exception as e:
                # 记录错误但继续处理其他图像
                print(f"处理图像 {image_path} 时出错: {str(e)}")
                # 创建一个空的结果
                empty_result = DetectionResult(
                    image_path=image_path,
                    image_shape=None,
                    boxes=[],
                    model_info=self.get_model_info(),
                    detection_params={
                        'confidence_threshold': conf_threshold,
                        'iou_threshold': iou_threshold,
                        'error': str(e)
                    }
                )
                results.append(empty_result)

        return results

    def predict_with_visualization(self, image_path: str, conf_threshold: float = 0.5,
                                   iou_threshold: float = 0.45) -> tuple:
        """
        执行预测并返回可视化结果

        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值

        Returns:
            tuple: (DetectionResult, 可视化图像)
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 读取图像信息
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"无法读取图像: {image_path}")

        image_shape = image.shape

        # 更新模型阈值参数
        self.model.update_thresholds(
            confidence_thres=conf_threshold,
            iou_thres=iou_threshold
        )

        # 执行推理并获取可视化结果
        result_image, detections = self.model.predict(image_path)

        # 转换检测结果
        boxes = []
        for detection in detections:
            bbox_info = detection["bbox"]

            x1 = float(bbox_info["x1"])
            y1 = float(bbox_info["y1"])
            x2 = float(bbox_info["x2"])
            y2 = float(bbox_info["y2"])

            bbox = BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(detection["confidence"]),
                class_id=int(detection["class_id"]),
                class_name=detection["class_name"]
            )
            boxes.append(bbox)

        detection_result = DetectionResult(
            image_path=image_path,
            image_shape=image_shape,
            boxes=boxes,
            model_info=self.get_model_info(),
            detection_params={
                'confidence_threshold': conf_threshold,
                'iou_threshold': iou_threshold
            }
        )

        return detection_result, result_image

    def get_model_info(self) -> Dict[str, Any]:
        """获取ONNX模型信息"""
        if self.model is None:
            return {"status": "未加载"}

        info = {
            "model_path": self.model_path,
            "model_type": "ONNX (.onnx)",
            "engine": "ONNXRuntime",
            "input_size": f"{self.model.input_width}x{self.model.input_height}",
            "status": "已加载"
        }

        if self.class_names:
            info["class_count"] = len(self.class_names)
            info["classes"] = list(self.class_names.values()) if isinstance(self.class_names,
                                                                            dict) else self.class_names

        return info

    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式"""
        return ['.onnx']

    def get_detection_summary(self) -> Dict[str, Any]:
        """获取最近一次检测的摘要信息"""
        if self.model is None:
            return {"error": "模型未加载"}

        return self.model.get_detection_summary()

    def export_last_results(self, output_path: str = None) -> str:
        """
        导出最近一次检测结果为JSON

        Args:
            output_path: 输出路径，如果为None则自动生成

        Returns:
            str: 输出文件路径
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        return self.model.export_results_to_json(output_path)

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def unload_model(self):
        """卸载模型，释放资源"""
        self.model = None
        self.model_path = None
        self.class_names = None

    def get_class_names(self) -> Dict[int, str]:
        """获取类别名称字典"""
        if self.model is None:
            return {}
        return self.class_names or {}

    def set_font_size(self, font_size: int):
        """设置可视化字体大小"""
        if self.model is not None:
            self.model.font_size = font_size
            # 重新初始化字体
            self.model._init_attributes()
