from importlib.metadata import metadata

import cv2
import numpy as np
import onnxruntime as ort
import json
import ast
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple

from settings import ROOT


class YOLO11:
    """YOLO11 目标检测模型类，用于处理推理和可视化。"""

    def __init__(self, onnx_model: str, class_names: Optional[Union[List[str], Dict[int, str]]] = None,
                 confidence_thres: float = 0.5, iou_thres: float = 0.45, font_size: int = 20):
        """
        初始化 YOLO11 类的实例。

        Args:
            onnx_model: ONNX 模型的路径
            class_names: 类别名称列表或字典，如果为None则从模型中读取
            confidence_thres: 置信度阈值
            iou_thres: NMS的IoU阈值
            font_size: 标签字体大小
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.font_size = font_size

        # 初始化ONNX会话
        self._init_session()

        # 处理类别名称
        self._init_class_names(class_names)

        # 初始化其他属性
        self._init_attributes()

    def _init_session(self):
        """初始化ONNX推理会话"""
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else [
            "CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_model, providers=providers)

        # 获取模型输入输出信息
        self.model_inputs = self.session.get_inputs()
        self.input_shape = self.model_inputs[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]

        logger.info(f"模型加载成功：{self.onnx_model}")
        logger.info(f"模型输入尺寸：{self.input_width}x{self.input_height}")

    def _init_class_names(self, class_names: Optional[Union[List[str], Dict[int, str]]]):
        """初始化类别名称"""
        metadata = None
        if class_names is None:
            # 尝试从模型元数据中获取类别名称
            try:
                metadata = self.session.get_modelmeta().custom_metadata_map.get('names')
                if metadata:
                    # 如果元数据是JSON字符串，解析它
                    if isinstance(metadata, str):
                        self.classes = json.loads(metadata)
                    else:
                        self.classes = metadata
                else:
                    # 如果没有元数据，使用默认类别名称
                    num_classes = self.input_shape[1] - 4  # 假设输出格式为 [batch, 4+classes, height, width]
                    self.classes = {i: f"class_{i}" for i in range(num_classes)}
            except Exception as _:
                try:
                    self.classes = ast.literal_eval(metadata)
                except Exception as e:
                    logger.warning(f"无法从模型获取类别信息：{e}")
                    # 使用默认类别
                    output_shape = self.session.get_outputs()[0].shape
                    num_classes = output_shape[1] - 4
                    self.classes = {i: f"class_{i}" for i in range(num_classes)}
        else:
            if isinstance(class_names, list):
                self.classes = {i: name for i, name in enumerate(class_names)}
            else:
                self.classes = class_names

        # 验证类别数量
        self._validate_class_count()

    def _validate_class_count(self):
        """验证模型类别数与提供的类别名称数量是否一致"""
        try:
            output_shape = self.session.get_outputs()[0].shape
            model_num_classes = output_shape[1] - 4
            provided_num_classes = len(self.classes)

            if model_num_classes != provided_num_classes:
                logger.warning(f"模型类别数({model_num_classes})与提供的类别数({provided_num_classes})不一致")
        except Exception as e:
            logger.warning(f"无法验证类别数量：{e}")

    def _init_attributes(self):
        """初始化其他属性"""
        # 为每个类别生成颜色调色板
        max_class_id = max(self.classes.keys()) if self.classes else 0
        np.random.seed(42)  # 固定随机种子，确保颜色一致性
        self.color_palette = np.random.uniform(0, 255, size=(max_class_id + 1, 3))

        # 初始化字体
        font_path = ROOT / 'fonts/simhei.ttf'
        try:
            self.font = ImageFont.truetype(str(font_path), self.font_size)
        except (OSError, IOError):
            logger.warning(f"无法加载字体文件：{font_path}，使用默认字体")
            self.font = ImageFont.load_default()

        # 重置检测结果
        self.detection_results = []

    def preprocess(self, image: Union[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        预处理输入图像

        Args:
            image: 输入图像路径或numpy数组

        Returns:
            Tuple[np.ndarray, Dict]: 预处理后的图像数据和元数据
        """
        # 读取图像
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法读取图像：{image}")
        else:
            img = image.copy()

        # 获取原始图像尺寸
        img_height, img_width = img.shape[:2]

        # 颜色空间转换
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Letterbox处理
        letterboxed_img, ratio, (dw, dh) = self._letterbox(img_rgb, (self.input_height, self.input_width))

        # 归一化和维度调整
        image_data = np.array(letterboxed_img, dtype=np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # HWC -> CHW
        image_data = np.expand_dims(image_data, axis=0)  # 添加batch维度

        # 保存预处理元数据
        metadata = {
            'original_shape': (img_height, img_width),
            'ratio': ratio,
            'pad': (dw, dh),
            'original_image': img
        }

        return image_data, metadata

    def _letterbox(self, img: np.ndarray, new_shape: Tuple[int, int],
                   color: Tuple[int, int, int] = (114, 114, 114), scaleup: bool = True) -> Tuple[
        np.ndarray, Tuple[float, float], Tuple[float, float]]:
        """
        Letterbox填充，保持宽高比

        Args:
            img: 输入图像
            new_shape: 目标尺寸 (height, width)
            color: 填充颜色
            scaleup: 是否允许放大

        Returns:
            Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]: 处理后的图像、缩放比例、填充偏移
        """
        shape = img.shape[:2]  # (height, width)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        # 计算新的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (width, height)

        # 计算填充
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]

        # 缩放图像
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 计算填充位置
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        # 添加边框
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img, (r, r), (dw / 2, dh / 2)

    def postprocess(self, outputs: List[np.ndarray], metadata: Dict) -> List[Dict]:
        """
        后处理模型输出

        Args:
            outputs: 模型输出
            metadata: 预处理元数据

        Returns:
            List[Dict]: 检测结果列表
        """
        # 获取预处理元数据
        original_shape = metadata['original_shape']
        ratio = metadata['ratio']
        dw, dh = metadata['pad']

        # 处理模型输出
        output = outputs[0]
        if len(output.shape) == 3:
            output = output[0]  # 移除batch维度

        output = output.T  # 转置为 (num_detections, 4+num_classes)

        # 提取边界框和置信度
        boxes = output[:, :4]
        scores = output[:, 4:]

        # 找到每个检测的最高置信度和对应类别
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # 过滤低置信度检测
        valid_indices = max_scores >= self.confidence_thres

        if not np.any(valid_indices):
            return []

        valid_boxes = boxes[valid_indices]
        valid_scores = max_scores[valid_indices]
        valid_class_ids = class_ids[valid_indices]

        # 坐标转换：从模型输出坐标转换为原图坐标
        valid_boxes = self._scale_boxes(valid_boxes, ratio, dw, dh, original_shape)

        # NMS
        nms_indices = self._apply_nms(valid_boxes, valid_scores)

        # 构建最终结果
        detections = []
        for idx in nms_indices:
            x1, y1, x2, y2 = valid_boxes[idx]
            detections.append({
                "class_id": int(valid_class_ids[idx]),
                "class_name": self.classes.get(int(valid_class_ids[idx]), f"class_{int(valid_class_ids[idx])}"),
                "confidence": float(valid_scores[idx]),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                }
            })

        self.detection_results = detections
        return detections

    def _scale_boxes(self, boxes: np.ndarray, ratio: Tuple[float, float],
                     dw: float, dh: float, original_shape: Tuple[int, int]) -> np.ndarray:
        """将边界框从模型坐标系转换到原图坐标系"""
        # 从中心点格式转换为角点格式
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # 移除填充
        x1 -= dw
        y1 -= dh
        x2 -= dw
        y2 -= dh

        # 缩放到原图尺寸
        x1 /= ratio[0]
        y1 /= ratio[1]
        x2 /= ratio[0]
        y2 /= ratio[1]

        # 限制在图像边界内
        img_h, img_w = original_shape
        x1 = np.clip(x1, 0, img_w - 1)
        y1 = np.clip(y1, 0, img_h - 1)
        x2 = np.clip(x2, 0, img_w - 1)
        y2 = np.clip(y2, 0, img_h - 1)

        return np.column_stack([x1, y1, x2, y2])

    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """应用非极大值抑制"""
        # 转换为OpenCV NMS需要的格式 (x, y, w, h)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cv_boxes = np.column_stack([x1, y1, x2 - x1, y2 - y1])

        indices = cv2.dnn.NMSBoxes(
            cv_boxes.tolist(),
            scores.tolist(),
            self.confidence_thres,
            self.iou_thres
        )

        if indices is not None:
            return indices.flatten().tolist() if hasattr(indices, 'flatten') else [indices]
        return []

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 输入图像
            detections: 检测结果列表

        Returns:
            np.ndarray: 绘制了检测结果的图像
        """
        if not detections:
            return image

        # 转换为PIL图像以支持中文显示
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        for detection in detections:
            bbox = detection["bbox"]
            class_id = detection["class_id"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]

            # 获取边界框坐标
            x1, y1 = bbox["x1"], bbox["y1"]
            x2, y2 = bbox["x2"], bbox["y2"]

            # 获取颜色
            color = tuple(map(int, self.color_palette[class_id % len(self.color_palette)]))

            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            self._draw_label(draw, label, x1, y1, color)

        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _draw_label(self, draw: ImageDraw.Draw, label: str, x: int, y: int, color: Tuple[int, int, int]):
        """绘制标签文本"""
        # 获取文本尺寸
        bbox = draw.textbbox((0, 0), label, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 计算标签位置
        label_y = max(0, y - text_height - 5)

        # 绘制背景
        draw.rectangle([x, label_y, x + text_width + 4, label_y + text_height + 4], fill=color)

        # 绘制文本
        draw.text((x + 2, label_y + 2), label, fill=(0, 0, 0), font=self.font)

    def predict(self, image: Union[str, np.ndarray], export_json: bool = False,
                json_output_path: Optional[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        执行目标检测预测

        Args:
            image: 输入图像路径或numpy数组
            export_json: 是否导出JSON结果
            json_output_path: JSON输出路径

        Returns:
            Tuple[np.ndarray, List[Dict]]: 带检测结果的图像和检测结果列表
        """
        # 预处理
        image_data, metadata = self.preprocess(image)

        # 推理
        outputs = self.session.run(None, {self.model_inputs[0].name: image_data})

        # 后处理
        detections = self.postprocess(outputs, metadata)

        # 绘制结果
        result_image = self.draw_detections(metadata['original_image'], detections)

        # 打印摘要
        self._print_summary(detections)

        # 导出JSON
        if export_json:
            self.export_results_to_json(image, json_output_path)

        return result_image, detections

    def _print_summary(self, detections: List[Dict]):
        """打印检测摘要"""
        if not detections:
            logger.info("未检测到任何目标")
            return

        class_counts = {}
        total_confidence = 0

        for detection in detections:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += confidence

        avg_confidence = total_confidence / len(detections)

        logger.info(f"检测完成！共检测到 {len(detections)} 个目标")
        logger.info(f"检测到的类别: {', '.join(class_counts.keys())}")
        logger.info(f"各类别数量: {class_counts}")
        logger.info(f"平均置信度: {avg_confidence:.3f}")

    def export_results_to_json(self, image_input: Union[str, np.ndarray], output_path: Optional[str] = None) -> str:
        """
        导出检测结果为JSON格式

        Args:
            image_input: 原始图像输入
            output_path: 输出路径

        Returns:
            str: 输出文件路径
        """
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"detection_results_{timestamp}.json"

        # 准备输出数据
        output_data = {
            "model_info": {
                "model_path": self.onnx_model,
                "input_size": f"{self.input_width}x{self.input_height}",
                "class_names": self.classes
            },
            "image_info": {
                "image_path": image_input if isinstance(image_input, str) else "numpy_array",
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "detection_parameters": {
                "confidence_threshold": self.confidence_thres,
                "iou_threshold": self.iou_thres
            },
            "detections": self.detection_results,
            "summary": {
                "total_detections": len(self.detection_results),
                "classes_detected": list(set([det["class_name"] for det in self.detection_results]))
            }
        }

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"检测结果已导出到: {output_path}")
        return output_path

    def get_detection_summary(self) -> Dict:
        """获取检测结果摘要"""
        if not self.detection_results:
            return {"total_detections": 0, "classes_detected": []}

        class_counts = {}
        total_confidence = 0

        for detection in self.detection_results:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += confidence

        return {
            "total_detections": len(self.detection_results),
            "class_counts": class_counts,
            "classes_detected": list(class_counts.keys()),
            "average_confidence": total_confidence / len(self.detection_results) if self.detection_results else 0
        }

    def update_thresholds(self, confidence_thres: Optional[float] = None, iou_thres: Optional[float] = None):
        """更新阈值参数"""
        if confidence_thres is not None:
            self.confidence_thres = confidence_thres
        if iou_thres is not None:
            self.iou_thres = iou_thres
        logger.info(f"阈值已更新 - 置信度: {self.confidence_thres}, IoU: {self.iou_thres}")
