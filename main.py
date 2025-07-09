import sys
import os
import random
from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QTextEdit,
    QSplitter, QToolBar, QMessageBox, QGroupBox, QScrollArea, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QAction as QActionGui
from ultralytics import YOLO

class DetectionWorker(QThread):
    finished = pyqtSignal(np.ndarray, str)
    error = pyqtSignal(str)
    def __init__(self, model, image_path, conf_threshold, iou_threshold):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    def run(self):
        try:
            results = self.model(
                self.image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            result = results[0]
            annotated_image = result.plot()
            detection_info = self._format_detection_results(result)
            self.finished.emit(annotated_image, detection_info)
        except Exception as e:
            self.error.emit(f"检测过程中出现错误: {str(e)}")
    def _format_detection_results(self, result):
        info_lines = []
        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            info_lines.append(f"检测到 {len(result.boxes)} 个目标:")
            info_lines.append("-" * 30)
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = result.names[cls_id] if hasattr(result, 'names') else f"Class_{cls_id}"
                info_lines.append(f"目标 {i+1}:")
                info_lines.append(f"  类别: {class_name}")
                info_lines.append(f"  置信度: {conf:.3f}")
                info_lines.append(f"  坐标: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                info_lines.append("")
        else:
            info_lines.append("未检测到任何目标")
        return "\n".join(info_lines)

class YOLODetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image_path = None
        self.image_list = []
        self.current_index = 0
        self.is_folder_mode = False
        self.detection_worker = None
        self.current_detection_result = None
        self.realtime_checkbox = None
        self.realtime_detection_enabled = False
        self.realtime_timer = QTimer(self)
        self.realtime_timer.setInterval(200) # 防抖间隔ms
        self.realtime_timer.setSingleShot(True)
        self.init_ui()
        self.setup_connections()
    def init_ui(self):
        self.setWindowTitle("YOLO 目标检测可视化工具 (PyQt6)")
        self.setGeometry(100, 100, 1400, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        left_widget = self.create_main_area()
        splitter.addWidget(left_widget)
        right_widget = self.create_sidebar()
        splitter.addWidget(right_widget)
        splitter.setSizes([1000, 400])
        self.create_toolbar()
        self.statusBar().showMessage("请先加载YOLO模型")
    def create_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)
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
        # 实时检测复选框
        self.realtime_checkbox = QCheckBox("实时检测")
        self.realtime_checkbox.setChecked(False)
        toolbar.addWidget(self.realtime_checkbox)
        # 检测按钮
        detect_action = QActionGui("开始检测", self)
        detect_action.triggered.connect(self.start_detection)
        toolbar.addAction(detect_action)
    def create_main_area(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        self.image_label.setText("请选择图片或文件夹")
        self.image_label.setMinimumSize(800, 600)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        self.create_bottom_controls(layout)
        return widget
    def create_bottom_controls(self, parent_layout):
        button_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.next_btn = QPushButton("下一张")
        self.random_btn = QPushButton("随机")
        self.save_btn = QPushButton("保存")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.random_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.prev_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.random_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addStretch()
        parent_layout.addLayout(button_layout)
    def create_sidebar(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        model_group = QGroupBox("模型信息")
        model_layout = QVBoxLayout(model_group)
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setText("未加载模型")
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
        self.realtime_checkbox.stateChanged.connect(self.on_realtime_checkbox_changed)
        self.realtime_timer.timeout.connect(self._trigger_realtime_detection)
    def update_conf_label(self, value):
        conf_value = value / 100.0
        self.conf_label.setText(f"{conf_value:.2f}")
    def update_iou_label(self, value):
        iou_value = value / 100.0
        self.iou_label.setText(f"{iou_value:.2f}")
    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO模型文件", "", "模型文件 (*.pt *.onnx);;所有文件 (*)"
        )
        if model_path:
            try:
                self.model = YOLO(model_path)
                model_info = f"模型路径: {model_path}\n"
                model_info += f"模型类型: {Path(model_path).suffix}\n"
                if hasattr(self.model, 'names'):
                    model_info += f"类别数量: {len(self.model.names)}\n"
                    model_info += f"类别: {list(self.model.names.values())}"
                self.model_info_text.setText(model_info)
                self.statusBar().showMessage(f"模型加载成功: {Path(model_path).name}")
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
                self.statusBar().showMessage(f"文件夹加载成功，共 {len(self.image_list)} 张图片")
                self.on_detection_triggered_by_user()
            else:
                QMessageBox.warning(self, "警告", "所选文件夹中没有找到图片文件")
    def display_current_image(self):
        if self.current_image_path and os.path.exists(self.current_image_path):
            image = cv2.imread(self.current_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.display_image(image_rgb)
            filename = Path(self.current_image_path).name
            if self.is_folder_mode:
                self.statusBar().showMessage(f"[{self.current_index + 1}/{len(self.image_list)}] {filename}")
            else:
                self.statusBar().showMessage(f"当前图片: {filename}")
            self.save_btn.setEnabled(False)
            self.result_text.setText("暂无检测结果")
    def display_image(self, image_array):
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    def resizeEvent(self, event):
        if self.image_label.pixmap() is not None:
            self.display_current_image()
        super().resizeEvent(event)
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
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载YOLO模型")
            return
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先选择图片或文件夹")
            return
        conf_threshold = self.conf_slider.value() / 100.0
        iou_threshold = self.iou_slider.value() / 100.0
        self.detection_worker = DetectionWorker(
            self.model, self.current_image_path, conf_threshold, iou_threshold
        )
        self.detection_worker.finished.connect(self.on_detection_finished)
        self.detection_worker.error.connect(self.on_detection_error)
        self.detection_worker.start()
        self.statusBar().showMessage("正在检测...")
    def on_detection_finished(self, annotated_image, detection_info):
        self.display_image(annotated_image)
        self.result_text.setText(detection_info)
        self.save_btn.setEnabled(True)
        self.current_detection_result = annotated_image
        self.statusBar().showMessage("检测完成")
    def on_detection_error(self, error_message):
        QMessageBox.critical(self, "检测错误", error_message)
        self.statusBar().showMessage("检测失败")
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
    def on_realtime_checkbox_changed(self, state):
        self.realtime_detection_enabled = (state == Qt.CheckState.Checked.value)
        # 若切换为实时检测且当前图片存在，立刻检测一次
        if self.realtime_detection_enabled:
            self.on_detection_triggered_by_user()
    def on_detection_triggered_by_user(self, *args):
        # 只有实时检测勾选时才自动检测，且防抖
        if self.realtime_detection_enabled:
            # 防抖：每次触发重新计时，timer超时后执行检测
            self.realtime_timer.stop()
            self.realtime_timer.start()
    def _trigger_realtime_detection(self):
        self.start_detection()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = YOLODetectionGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()