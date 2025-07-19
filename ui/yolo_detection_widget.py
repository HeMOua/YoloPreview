import sys
import os
import random
import json
from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QTextEdit,
    QSplitter, QToolBar, QMessageBox, QGroupBox, QScrollArea, QCheckBox,
    QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction as QActionGui, QIcon

from detection.strategy_base import DetectionResult
from detection.strategy_factory import DetectionContext, DetectionStrategyFactory
from ui.components.image_dispaly import ImageDisplayLabel
from utils.paint import draw_boxes_with_pil


class DetectionWorker(QThread):
    """检测工作线程"""
    finished = pyqtSignal(np.ndarray, str, object)
    error = pyqtSignal(str)

    def __init__(self, detection_context: DetectionContext, image_path: str,
                 conf_threshold: float, iou_threshold: float):
        super().__init__()
        self.detection_context = detection_context
        self.image_path = image_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._should_stop = False

    def run(self):
        try:
            if self._should_stop:
                return

            # 读取图像
            image = cv2.imread(self.image_path)
            if image is None:
                self.error.emit(f"无法读取图像: {self.image_path}")
                return

            if self._should_stop:
                return

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self._should_stop:
                return

            # 执行检测
            detection_result = self.detection_context.predict(
                self.image_path, self.conf_threshold, self.iou_threshold
            )

            if self._should_stop:
                return

            # 绘制结果
            annotated_image = draw_boxes_with_pil(
                image_rgb,
                detection_result.extract_visualization_data()
            )

            # 格式化检测信息
            detection_info = self._format_detection_results(detection_result)

            self.finished.emit(annotated_image, detection_info, detection_result)

        except Exception as e:
            self.error.emit(f"检测过程中出现错误: {str(e)}")

    def stop(self):
        self._should_stop = True

    def _format_detection_results(self, result: DetectionResult) -> str:
        """格式化检测结果为文本"""
        info_lines = []

        if result.boxes:
            info_lines.append(f"检测到 {len(result.boxes)} 个目标:")
            info_lines.append("-" * 30)

            for i, box in enumerate(result.boxes):
                info_lines.append(f"目标 {i + 1}:")
                info_lines.append(f"  类别: {box.class_name}")
                info_lines.append(f"  置信度: {box.confidence:.3f}")
                info_lines.append(f"  坐标: ({box.x1:.1f}, {box.y1:.1f}, {box.x2:.1f}, {box.y2:.1f})")
                info_lines.append("")
        else:
            info_lines.append("未检测到任何目标")

        # 添加模型信息
        info_lines.append("=" * 30)
        info_lines.append("模型信息:")
        model_info = result.model_info
        info_lines.append(f"引擎: {model_info.get('engine', 'Unknown')}")
        info_lines.append(f"类型: {model_info.get('model_type', 'Unknown')}")

        return "\n".join(info_lines)


class YOLODetectionWidget(QWidget):
    """YOLO检测GUI主窗口 - 支持多种推理引擎"""

    def __init__(self):
        super().__init__()
        self.detection_context = DetectionContext()
        self.current_image_path = None
        self.image_list = []
        self.current_index = 0
        self.is_folder_mode = False
        self.detection_worker = None
        self.current_detection_result = None
        self.current_detection_result_obj = None
        self.realtime_checkbox = None
        self.realtime_detection_enabled = False
        self.realtime_timer = QTimer(self)
        self.realtime_timer.setInterval(200)
        self.realtime_timer.setSingleShot(True)
        self.toolbar = QToolBar(self)
        self.statusbar = QStatusBar(self)
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.toolbar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)
        left_widget = self.create_main_area()
        splitter.addWidget(left_widget)
        right_widget = self.create_sidebar()
        splitter.addWidget(right_widget)
        splitter.setSizes([1000, 400])
        self.create_bottom_controls(main_layout)
        self.create_toolbar()
        main_layout.addWidget(self.statusbar)
        self.statusbar.showMessage("请先加载YOLO模型 (支持 .pt 和 .onnx 格式)")

    def create_toolbar(self):
        toolbar = self.toolbar
        toolbar.clear()  # 清空旧的action

        load_model_action = QActionGui("加载模型", self)
        load_model_action.triggered.connect(self.load_model)
        toolbar.addAction(load_model_action)

        toolbar.addSeparator()

        select_image_action = QActionGui("选择图片", self)
        select_image_action.triggered.connect(self.select_image)
        toolbar.addAction(select_image_action)

        select_folder_action = QActionGui("选择文件夹", self)
        select_folder_action.triggered.connect(self.select_folder)
        toolbar.addAction(select_folder_action)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("置信度:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.setFixedWidth(100)
        toolbar.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.25")
        toolbar.addWidget(self.conf_label)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("IOU:"))
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(45)
        self.iou_slider.setFixedWidth(100)
        toolbar.addWidget(self.iou_slider)
        self.iou_label = QLabel("0.45")
        toolbar.addWidget(self.iou_label)

        toolbar.addSeparator()

        self.realtime_checkbox = QCheckBox("实时检测")
        self.realtime_checkbox.setChecked(False)
        toolbar.addWidget(self.realtime_checkbox)

        self.detect_action = QActionGui("开始检测", self)
        self.detect_action.triggered.connect(self.start_detection)
        toolbar.addAction(self.detect_action)

    def create_main_area(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 图片显示区
        self.image_label = ImageDisplayLabel()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        return widget

    def create_bottom_controls(self, parent_layout):
        button_layout = QHBoxLayout()

        self.prev_btn = QPushButton("上一张")
        self.next_btn = QPushButton("下一张")
        self.random_btn = QPushButton("随机")
        self.save_btn = QPushButton("保存")
        self.export_json_btn = QPushButton("导出JSON")  # 新增JSON导出按钮
        self.reset_btn = QPushButton("复位")
        self.xy_label = QLabel("xy: -,-")
        self.xy_label.setStyleSheet("color: #555; padding: 2px 8px;")

        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.random_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.export_json_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)

        button_layout.addWidget(self.prev_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.random_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.export_json_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.xy_label, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        button_layout.addStretch()

        parent_layout.addLayout(button_layout)

    def create_sidebar(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        model_group = QGroupBox("模型信息")
        model_layout = QVBoxLayout(model_group)
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setText("未加载模型\n支持格式: .pt (PyTorch), .onnx (ONNX)")
        model_layout.addWidget(self.model_info_text)
        layout.addWidget(model_group)

        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout(result_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setText("暂无检测结果")
        result_layout.addWidget(self.result_text)
        layout.addWidget(result_group)

        return widget

    def setup_connections(self):
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        self.conf_slider.valueChanged.connect(self.on_detection_triggered_by_user)
        self.iou_slider.valueChanged.connect(self.on_detection_triggered_by_user)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.random_btn.clicked.connect(self.random_image)
        self.save_btn.clicked.connect(self.save_result)
        self.export_json_btn.clicked.connect(self.export_json)  # 连接JSON导出
        self.realtime_checkbox.stateChanged.connect(self.on_realtime_checkbox_changed)
        self.realtime_timer.timeout.connect(self._trigger_realtime_detection)
        self.reset_btn.clicked.connect(self.image_label.reset_view)
        self.image_label.mouse_image_pos_changed.connect(self.update_xy_label)

    def update_conf_label(self, value):
        conf_value = value / 100.0
        self.conf_label.setText(f"{conf_value:.2f}")

    def update_iou_label(self, value):
        iou_value = value / 100.0
        self.iou_label.setText(f"{iou_value:.2f}")

    def load_model(self):
        # 获取支持的格式
        supported_formats = DetectionStrategyFactory.get_supported_formats()
        format_filter = "模型文件 ("
        format_filter += " ".join([f"*{fmt}" for fmt in supported_formats])
        format_filter += ");;所有文件 (*)"

        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO模型文件", "", format_filter
        )

        if model_path:
            try:
                # 使用策略模式加载模型
                success = self.detection_context.load_model(model_path)

                if success:
                    model_info = self.detection_context.get_model_info()

                    # 格式化模型信息显示
                    info_text = f"模型路径: {model_info.get('model_path', 'Unknown')}\n"
                    info_text += f"模型类型: {model_info.get('model_type', 'Unknown')}\n"
                    info_text += f"推理引擎: {model_info.get('engine', 'Unknown')}\n"

                    if 'class_count' in model_info:
                        info_text += f"类别数量: {model_info['class_count']}\n"
                        if 'classes' in model_info:
                            classes = model_info['classes']
                            info_text += f"类别: {classes}"

                    self.model_info_text.setText(info_text)

                    model_name = Path(model_path).name
                    model_type = model_info.get('model_type', 'Unknown')
                    self.statusbar.showMessage(f"模型加载成功: {model_name} ({model_type})")

                    # 如果已有图像且启用实时检测，自动开始检测
                    if self.current_image_path is not None and self.realtime_detection_enabled:
                        self.start_detection()

            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")

    def select_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)"
        )
        if image_path:
            self.current_image_path = image_path
            self.image_list = [image_path]
            self.current_index = 0
            self.is_folder_mode = False
            self.display_current_image()
            self.update_button_states()
            self.on_detection_triggered_by_user()

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder_path:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            self.image_list = []
            for ext in image_extensions:
                self.image_list.extend(str(p) for p in Path(folder_path).glob(f"*{ext}"))
                self.image_list.extend(str(p) for p in Path(folder_path).glob(f"*{ext.upper()}"))
            if self.image_list:
                self.current_index = 0
                self.current_image_path = self.image_list[0]
                self.is_folder_mode = True
                self.display_current_image()
                self.update_button_states()
                self.statusbar.showMessage(f"文件夹加载成功，共 {len(self.image_list)} 张图片")
                self.on_detection_triggered_by_user()
            else:
                QMessageBox.warning(self, "警告", "所选文件夹中没有找到图片文件")

    def display_current_image(self):
        if self.current_image_path and os.path.exists(self.current_image_path):
            image = cv2.imread(self.current_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_label.set_image(image_rgb)
            filename = Path(self.current_image_path).name
            if self.is_folder_mode:
                self.statusbar.showMessage(f"[{self.current_index + 1}/{len(self.image_list)}] {filename}")
            else:
                self.statusbar.showMessage(f"当前图片: {filename}")
            self.save_btn.setEnabled(False)
            self.export_json_btn.setEnabled(False)
            self.result_text.setText("暂无检测结果")

    def update_button_states(self):
        if self.is_folder_mode and len(self.image_list) > 1:
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.random_btn.setEnabled(True)
        else:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.random_btn.setEnabled(False)

    def prev_image(self):
        if self.is_folder_mode and len(self.image_list) > 1:
            self.current_index = (self.current_index - 1) % len(self.image_list)
            self.current_image_path = self.image_list[self.current_index]
            self.display_current_image()
            self.on_detection_triggered_by_user()

    def next_image(self):
        if self.is_folder_mode and len(self.image_list) > 1:
            self.current_index = (self.current_index + 1) % len(self.image_list)
            self.current_image_path = self.image_list[self.current_index]
            self.display_current_image()
            self.on_detection_triggered_by_user()

    def random_image(self):
        if self.is_folder_mode and len(self.image_list) > 1:
            new_index = random.randint(0, len(self.image_list) - 1)
            while new_index == self.current_index and len(self.image_list) > 1:
                new_index = random.randint(0, len(self.image_list) - 1)
            self.current_index = new_index
            self.current_image_path = self.image_list[self.current_index]
            self.display_current_image()
            self.on_detection_triggered_by_user()

    def start_detection(self):
        if not self.detection_context.is_model_loaded():
            QMessageBox.warning(self, "警告", "请先加载YOLO模型")
            return
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先选择图片或文件夹")
            return

        conf_threshold = self.conf_slider.value() / 100.0
        iou_threshold = self.iou_slider.value() / 100.0

        self.detection_worker = DetectionWorker(
            self.detection_context, self.current_image_path, conf_threshold, iou_threshold
        )
        self.detection_worker.finished.connect(self.on_detection_finished)
        self.detection_worker.error.connect(self.on_detection_error)
        self.detection_worker.start()

        self.statusbar.showMessage("正在检测...")

        # 检测期间禁用控件
        self.set_controls_enabled(False)

    def on_detection_finished(self, annotated_image, detection_info, detection_result):
        self.image_label.set_image(annotated_image)
        self.result_text.setText(detection_info)
        self.save_btn.setEnabled(True)
        self.export_json_btn.setEnabled(True)
        self.current_detection_result = annotated_image
        self.current_detection_result_obj = detection_result

        self.statusbar.showMessage("检测完成")

        # 重新启用控件
        self.set_controls_enabled(True)

    def on_detection_error(self, error_message):
        QMessageBox.critical(self, "检测错误", error_message)
        self.statusbar.showMessage("检测失败")

        # 重新启用控件
        self.set_controls_enabled(True)

    def set_controls_enabled(self, enabled):
        """统一控制界面控件的启用状态"""
        self.update_button_states() if enabled else self._disable_navigation_buttons()
        self.detect_action.setEnabled(enabled)
        self.conf_slider.setEnabled(enabled)
        self.iou_slider.setEnabled(enabled)
        self.realtime_checkbox.setEnabled(enabled)

    def _disable_navigation_buttons(self):
        """禁用导航按钮"""
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.random_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.export_json_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

    def save_result(self):
        if self.current_detection_result is None:
            QMessageBox.warning(self, "警告", "没有检测结果可保存")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", "", "图片文件 (*.jpg *.png);;所有文件 (*)"
        )
        if save_path:
            try:
                result_bgr = cv2.cvtColor(self.current_detection_result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, result_bgr)
                QMessageBox.information(self, "成功", f"检测结果已保存到: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def export_json(self):
        """导出检测结果为JSON格式"""
        if self.current_detection_result_obj is None:
            QMessageBox.warning(self, "警告", "没有检测结果可导出")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出JSON结果", "", "JSON文件 (*.json);;所有文件 (*)"
        )
        if save_path:
            try:

                # 构建JSON数据
                export_data = {
                    "image_info": {
                        "image_path": self.current_detection_result_obj.image_path,
                        "image_shape": {
                            "height": self.current_detection_result_obj.image_shape[0],
                            "width": self.current_detection_result_obj.image_shape[1],
                            "channels": self.current_detection_result_obj.image_shape[2]
                        }
                    },
                    "model_info": self.current_detection_result_obj.model_info,
                    "detection_params": self.current_detection_result_obj.detection_params,
                    "detections": [
                        {
                            "class_id": box.class_id,
                            "class_name": box.class_name,
                            "confidence": box.confidence,
                            "bbox": {
                                "x1": box.x1,
                                "y1": box.y1,
                                "x2": box.x2,
                                "y2": box.y2,
                                "width": box.x2 - box.x1,
                                "height": box.y2 - box.y1
                            }
                        }
                        for box in self.current_detection_result_obj.boxes
                    ],
                    "summary": {
                        "total_detections": len(self.current_detection_result_obj.boxes),
                        "classes_detected": list(
                            set([box.class_name for box in self.current_detection_result_obj.boxes]))
                    }
                }

                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

                QMessageBox.information(self, "成功", f"检测结果已导出到: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def on_realtime_checkbox_changed(self, state):
        self.realtime_detection_enabled = (state == Qt.CheckState.Checked.value)
        if self.realtime_detection_enabled:
            self.on_detection_triggered_by_user()

    def on_detection_triggered_by_user(self, *args):
        if self.realtime_detection_enabled:
            self.realtime_timer.stop()
            self.realtime_timer.start()

    def _trigger_realtime_detection(self):
        self.start_detection()

    def update_xy_label(self, x, y):
        if x >= 0 and y >= 0:
            self.xy_label.setText(f"xy: {x}, {y}")
        else:
            self.xy_label.setText("xy: -,-")

