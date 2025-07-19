import os
import random
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QTextEdit, QSplitter, QToolBar, QMessageBox, QGroupBox, QCheckBox,
    QStatusBar, QComboBox, QSpinBox, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction as QActionGui

from ui.components.image_dispaly import ImageDisplayLabel
from ui.components.message_box import CustomMessageBox
from utils.paint import draw_boxes_with_pil


class COCODatasetInfo:
    """COCO数据集信息类"""

    def __init__(self):
        self.dataset_path = ""
        self.annotation_file = ""
        self.images_dir = ""
        self.annotations_data = {}
        self.images_data = {}
        self.categories_data = {}
        self.image_files = []
        self.category_stats = {}
        self.total_annotations = 0
        self.annotation_types = set()  # 'bbox', 'segmentation'


class COCOAnnotation:
    """COCO标注类"""

    def __init__(self, annotation_data, category_name="unknown"):
        self.id = annotation_data.get('id', 0)
        self.image_id = annotation_data.get('image_id', 0)
        self.category_id = annotation_data.get('category_id', 0)
        self.category_name = category_name
        self.area = annotation_data.get('area', 0)
        self.iscrowd = annotation_data.get('iscrowd', 0)

        # 检测框
        self.bbox = annotation_data.get('bbox', [])  # [x, y, width, height]
        self.x1 = self.bbox[0] if len(self.bbox) >= 4 else 0
        self.y1 = self.bbox[1] if len(self.bbox) >= 4 else 0
        self.x2 = self.x1 + self.bbox[2] if len(self.bbox) >= 4 else 0
        self.y2 = self.y1 + self.bbox[3] if len(self.bbox) >= 4 else 0

        # 分割掩码
        self.segmentation = annotation_data.get('segmentation', [])
        self.has_segmentation = bool(self.segmentation)

        # 如果segmentation是RLE格式
        if isinstance(self.segmentation, dict):
            self.is_rle = True
        else:
            self.is_rle = False


class COCODatasetLoadWorker(QThread):
    """COCO数据集加载工作线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str, str)  # 增加详细日志参数

    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self._should_stop = False
        self.search_log = []  # 新增：记录搜索日志

    def run(self):
        try:
            dataset_info = COCODatasetInfo()
            dataset_info.dataset_path = self.dataset_path
            dataset_path = Path(self.dataset_path)

            self._log_and_progress(f"正在查找COCO标注文件... (目录: {dataset_path.resolve()})")

            # 查找标注文件
            annotation_file = self.find_annotation_file(dataset_path)
            if not annotation_file:
                self.search_log.append("未找到COCO标注文件")
                self.error.emit("未找到COCO标注文件", self._get_search_log_str())
                return

            dataset_info.annotation_file = str(annotation_file)
            self._log_and_progress(f"找到标注文件: {Path(annotation_file).resolve()}")

            # 加载COCO标注
            self._log_and_progress("正在加载COCO标注数据...")
            with open(annotation_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)

            # 解析COCO数据
            self._log_and_progress("正在解析COCO数据...")
            self.parse_coco_data(coco_data, dataset_info)

            # 查找图像目录
            self._log_and_progress("正在查找图像目录...")
            images_dir, images_log = self.find_images_dir(dataset_path, dataset_info)
            self.search_log.extend(images_log)
            if not images_dir:
                self.search_log.append("未找到图像目录")
                self.error.emit("未找到图像目录", self._get_search_log_str())
                return

            dataset_info.images_dir = str(images_dir)

            # 扫描图像文件
            self._log_and_progress(f"正在扫描图像文件... (目录: {Path(images_dir).resolve()})")
            self.scan_image_files(dataset_info)

            # 计算统计信息
            self._log_and_progress("正在计算统计信息...")
            self.calculate_statistics(dataset_info)

            self.finished.emit(dataset_info)

        except Exception as e:
            self.search_log.append(f"COCO数据集加载失败: {str(e)}")
            self.error.emit(f"COCO数据集加载失败: {str(e)}", self._get_search_log_str())

    def _log_and_progress(self, msg):
        self.search_log.append(msg)
        self.progress.emit(msg)

    def _get_search_log_str(self):
        return '\n'.join(self.search_log)

    def find_annotation_file(self, dataset_path):
        """查找COCO标注文件"""
        annotation_files = [
            'annotations.json', 'instances.json', 'coco.json',
            'train.json', 'val.json', 'test.json'
        ]
        found = False
        log = []
        # 先在根目录查找
        for filename in annotation_files:
            file_path = (dataset_path / filename).resolve()
            if file_path.exists():
                log.append(f"尝试查找标注文件: {file_path}（找到）")
                self.search_log.extend(log)
                return file_path
            else:
                log.append(f"尝试查找标注文件: {file_path}（未找到）")
        # 在annotations子目录查找
        annotations_dir = (dataset_path / 'annotations').resolve()
        if annotations_dir.exists():
            for filename in annotation_files:
                file_path = (annotations_dir / filename).resolve()
                if file_path.exists():
                    log.append(f"尝试查找标注文件: {file_path}（找到）")
                    self.search_log.extend(log)
                    return file_path
                else:
                    log.append(f"尝试查找标注文件: {file_path}（未找到）")
            # 查找所有json文件
            for json_file in annotations_dir.glob("*.json"):
                log.append(f"尝试查找标注文件: {json_file.resolve()}（找到）")
                self.search_log.extend(log)
                return json_file
        # 在根目录查找所有json文件
        for json_file in dataset_path.glob("*.json"):
            log.append(f"尝试查找标注文件: {json_file.resolve()}（找到）")
            self.search_log.extend(log)
            return json_file
        self.search_log.extend(log)
        return None

    def parse_coco_data(self, coco_data, dataset_info):
        """解析COCO数据"""
        # 解析类别
        categories = coco_data.get('categories', [])
        for cat in categories:
            dataset_info.categories_data[cat['id']] = cat

        # 解析图像信息
        images = coco_data.get('images', [])
        for img in images:
            dataset_info.images_data[img['id']] = img

        # 解析标注信息
        annotations = coco_data.get('annotations', [])
        for ann in annotations:
            image_id = ann['image_id']
            if image_id not in dataset_info.annotations_data:
                dataset_info.annotations_data[image_id] = []

            # 确定标注类型
            if 'bbox' in ann and ann['bbox']:
                dataset_info.annotation_types.add('bbox')
            if 'segmentation' in ann and ann['segmentation']:
                dataset_info.annotation_types.add('segmentation')

            dataset_info.annotations_data[image_id].append(ann)

    def find_images_dir(self, dataset_path, dataset_info):
        """查找图像目录"""
        log = []
        # 从图像信息中获取第一个图像的文件名，推断目录结构
        if dataset_info.images_data:
            first_image = next(iter(dataset_info.images_data.values()))
            filename = first_image.get('file_name', '')

            # 检查可能的图像目录
            possible_dirs = [
                dataset_path,
                dataset_path / 'images',
                dataset_path / 'img',
                dataset_path / 'train',
                dataset_path / 'val',
                dataset_path / 'test'
            ]

            for dir_path in possible_dirs:
                dir_path = dir_path.resolve()
                if dir_path.exists():
                    test_file = dir_path / filename
                    if test_file.exists():
                        log.append(f"尝试查找图像目录: {dir_path}，文件: {test_file}（找到）")
                        return dir_path, log
                    else:
                        log.append(f"尝试查找图像目录: {dir_path}，文件: {test_file}（未找到）")
                else:
                    log.append(f"尝试查找图像目录: {dir_path}（未找到）")

        # 如果找不到，使用包含最多图像的目录
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        best_dir = None
        max_images = 0
        for dir_path in dataset_path.rglob("*"):
            if dir_path.is_dir():
                image_count = sum(1 for f in dir_path.iterdir()
                                  if f.suffix.lower() in image_extensions)
                if image_count > max_images:
                    max_images = image_count
                    best_dir = dir_path
        if best_dir:
            log.append(f"回退：选择包含最多图像的目录: {best_dir.resolve()}（共{max_images}张）")
            return best_dir, log
        return None, log

    def scan_image_files(self, dataset_info):
        """扫描图像文件"""
        images_dir = Path(dataset_info.images_dir)

        # 根据COCO数据中的图像信息扫描
        for image_id, image_info in dataset_info.images_data.items():
            filename = image_info.get('file_name', '')
            if filename:
                file_path = images_dir / filename
                if file_path.exists():
                    dataset_info.image_files.append({
                        'id': image_id,
                        'path': str(file_path),
                        'info': image_info
                    })

    def calculate_statistics(self, dataset_info):
        """计算统计信息"""
        category_stats = {}
        total_annotations = 0

        # 初始化统计
        for cat_id, cat_info in dataset_info.categories_data.items():
            category_stats[cat_info['name']] = 0

        # 统计每个类别的标注数量
        for image_id, annotations in dataset_info.annotations_data.items():
            for ann in annotations:
                cat_id = ann.get('category_id', 0)
                if cat_id in dataset_info.categories_data:
                    cat_name = dataset_info.categories_data[cat_id]['name']
                    category_stats[cat_name] += 1
                    total_annotations += 1

        dataset_info.category_stats = category_stats
        dataset_info.total_annotations = total_annotations

    def stop(self):
        self._should_stop = True


class COCODatasetViewerWidget(QWidget):
    """COCO数据集预览控件"""

    def __init__(self):
        super().__init__()
        self.dataset_info = None
        self.current_image_path = None
        self.current_index = 0
        self.current_annotations = []
        self.load_worker = None
        self.show_annotations = True
        self.show_bbox = True
        self.show_segmentation = True
        self.show_seg_labels = True  # 新增：是否显示分割标签
        self.selected_categories = set()
        self.category_colors = {}

        self.toolbar = QToolBar(self)
        self.statusbar = QStatusBar(self)

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # 主显示区域
        left_widget = self.create_main_area()
        splitter.addWidget(left_widget)

        # 侧边栏
        right_widget = self.create_sidebar()
        splitter.addWidget(right_widget)

        splitter.setSizes([1000, 400])

        # 底部控制区
        self.create_bottom_controls(main_layout)

        # 工具栏
        self.create_toolbar()

        main_layout.addWidget(self.statusbar)
        self.statusbar.showMessage("请选择COCO数据集目录")
        self.update_button_states()

    def create_toolbar(self):
        toolbar = self.toolbar
        toolbar.clear()

        # 加载数据集
        load_dataset_action = QActionGui("加载数据集", self)
        load_dataset_action.triggered.connect(self.load_dataset)
        toolbar.addAction(load_dataset_action)

        # 查看加载日志
        self.view_log_action = QActionGui("查看加载日志", self)
        self.view_log_action.triggered.connect(self.show_load_log)
        toolbar.addAction(self.view_log_action)

        toolbar.addSeparator()

        # 显示控制
        self.show_annotations_checkbox = QCheckBox("显示标注")
        self.show_annotations_checkbox.setChecked(True)
        toolbar.addWidget(self.show_annotations_checkbox)

        self.show_bbox_checkbox = QCheckBox("检测框")
        self.show_bbox_checkbox.setChecked(True)
        toolbar.addWidget(self.show_bbox_checkbox)

        self.show_segmentation_checkbox = QCheckBox("分割掩码")
        self.show_segmentation_checkbox.setChecked(True)
        toolbar.addWidget(self.show_segmentation_checkbox)

        # 新增：分割标签显示控制
        self.show_seg_labels_checkbox = QCheckBox("分割标签")
        self.show_seg_labels_checkbox.setChecked(True)
        toolbar.addWidget(self.show_seg_labels_checkbox)

        toolbar.addSeparator()

        # 类别筛选
        toolbar.addWidget(QLabel("筛选类别:"))
        self.category_filter_combo = QComboBox()
        self.category_filter_combo.addItem("显示全部")
        self.category_filter_combo.setMinimumWidth(120)
        toolbar.addWidget(self.category_filter_combo)

        toolbar.addSeparator()

        # 跳转到指定索引
        toolbar.addWidget(QLabel("跳转到:"))
        self.goto_spinbox = QSpinBox()
        self.goto_spinbox.setMinimum(1)
        self.goto_spinbox.setMaximum(1)
        self.goto_spinbox.setEnabled(False)
        toolbar.addWidget(self.goto_spinbox)

        self.goto_action = QActionGui("GO", self)
        self.goto_action.triggered.connect(self.goto_image)
        toolbar.addAction(self.goto_action)

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

    def create_sidebar(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 数据集信息
        dataset_group = QGroupBox("数据集信息")
        dataset_layout = QVBoxLayout(dataset_group)
        self.dataset_info_text = QTextEdit()
        self.dataset_info_text.setReadOnly(True)
        self.dataset_info_text.setText("未加载数据集")
        dataset_layout.addWidget(self.dataset_info_text)
        layout.addWidget(dataset_group)

        # 当前图像标注信息
        annotation_group = QGroupBox("当前图像标注")
        annotation_layout = QVBoxLayout(annotation_group)
        self.annotation_text = QTextEdit()
        self.annotation_text.setReadOnly(True)
        self.annotation_text.setText("无标注信息")
        annotation_layout.addWidget(self.annotation_text)
        layout.addWidget(annotation_group)

        # 类别统计
        stats_group = QGroupBox("类别统计")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setText("无统计信息")
        stats_layout.addWidget(self.stats_text)
        layout.addWidget(stats_group)

        return widget

    def create_bottom_controls(self, parent_layout):
        button_layout = QHBoxLayout()

        self.first_btn = QPushButton("首张")
        self.prev_btn = QPushButton("上一张")
        self.next_btn = QPushButton("下一张")
        self.last_btn = QPushButton("末张")
        self.random_btn = QPushButton("随机")
        self.export_btn = QPushButton("导出信息")
        self.reset_btn = QPushButton("复位")
        self.xy_label = QLabel("xy: -,-")
        self.xy_label.setStyleSheet("color: #555; padding: 2px 8px;")

        # 初始状态禁用
        self.first_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.last_btn.setEnabled(False)
        self.random_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        button_layout.addWidget(self.first_btn)
        button_layout.addWidget(self.prev_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.last_btn)
        button_layout.addWidget(self.random_btn)
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.xy_label, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        button_layout.addStretch()

        parent_layout.addLayout(button_layout)

    def setup_connections(self):
        # 按钮连接
        self.first_btn.clicked.connect(self.first_image)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.last_btn.clicked.connect(self.last_image)
        self.random_btn.clicked.connect(self.random_image)
        self.export_btn.clicked.connect(self.export_dataset_info)
        self.reset_btn.clicked.connect(self.image_label.reset_view)

        # 显示控制
        self.show_annotations_checkbox.stateChanged.connect(self.on_show_annotations_changed)
        self.show_bbox_checkbox.stateChanged.connect(self.on_display_options_changed)
        self.show_segmentation_checkbox.stateChanged.connect(self.on_display_options_changed)
        self.show_seg_labels_checkbox.stateChanged.connect(self.on_display_options_changed)  # 新增
        self.category_filter_combo.currentTextChanged.connect(self.on_category_filter_changed)

        # 鼠标位置
        self.image_label.mouse_image_pos_changed.connect(self.update_xy_label)

    def load_dataset(self):
        dataset_path = QFileDialog.getExistingDirectory(self, "选择COCO数据集目录")
        if dataset_path:
            self.statusbar.showMessage("正在加载COCO数据集...")

            self.load_worker = COCODatasetLoadWorker(dataset_path)
            self.load_worker.progress.connect(self.on_load_progress)
            self.load_worker.finished.connect(self.on_load_finished)
            self.load_worker.error.connect(self.on_load_error)  # 注意：参数变为2个
            self.load_worker.start()

    def on_load_progress(self, message):
        self.statusbar.showMessage(message)

    def on_load_finished(self, dataset_info):
        self.dataset_info = dataset_info
        self.current_index = 0
        self._init_category_colors()

        # 更新界面
        self.update_dataset_info_display()
        self.update_category_filter_combo()
        self.update_button_states()
        self.update_goto_spinbox()

        # 显示第一张图像
        self.display_current_image()

        total_images = len(self.dataset_info.image_files)
        self.statusbar.showMessage(f"COCO数据集加载完成，共 {total_images} 张图像")

    def on_load_error(self, error_message, detail_log=None):
        try:
            msg_box = CustomMessageBox("加载错误", error_message, detail_log, self)
            msg_box.exec()
            self.statusbar.showMessage("COCO数据集加载失败")
        except Exception as e:
            print(f"on_load_error exception: {e}")
        self.update_button_states()

    def update_dataset_info_display(self):
        if not self.dataset_info:
            return

        info_lines = []
        info_lines.append(f"数据集路径: {self.dataset_info.dataset_path}")
        info_lines.append(f"标注文件: {Path(self.dataset_info.annotation_file).name}")
        info_lines.append(f"图像目录: {Path(self.dataset_info.images_dir).name}")
        info_lines.append(f"图像数量: {len(self.dataset_info.image_files)}")
        info_lines.append(f"类别数量: {len(self.dataset_info.categories_data)}")
        info_lines.append(f"总标注数: {self.dataset_info.total_annotations}")

        if self.dataset_info.annotation_types:
            info_lines.append(f"标注类型: {', '.join(self.dataset_info.annotation_types)}")

        if self.dataset_info.categories_data:
            info_lines.append("\n类别列表:")
            for cat_id, cat_info in self.dataset_info.categories_data.items():
                info_lines.append(f"  {cat_id}: {cat_info['name']}")

        self.dataset_info_text.setText("\n".join(info_lines))

        # 更新统计信息
        stats_lines = []
        if self.dataset_info.category_stats:
            stats_lines.append("各类别标注数量:")
            for cat_name, count in self.dataset_info.category_stats.items():
                percentage = (
                            count / self.dataset_info.total_annotations * 100) if self.dataset_info.total_annotations > 0 else 0
                stats_lines.append(f"  {cat_name}: {count} ({percentage:.1f}%)")

        self.stats_text.setText("\n".join(stats_lines))

    def update_category_filter_combo(self):
        self.category_filter_combo.clear()
        self.category_filter_combo.addItem("显示全部")

        if self.dataset_info and self.dataset_info.categories_data:
            for cat_info in self.dataset_info.categories_data.values():
                self.category_filter_combo.addItem(cat_info['name'])

    def update_button_states(self):
        has_images = (self.dataset_info and len(self.dataset_info.image_files) > 0)
        has_multiple = (has_images and len(self.dataset_info.image_files) > 1)

        if not self.dataset_info:
            # 没有数据集时全部禁用
            self.first_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.last_btn.setEnabled(False)
            self.random_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.show_annotations_checkbox.setEnabled(False)
            self.show_bbox_checkbox.setEnabled(False)
            self.show_segmentation_checkbox.setEnabled(False)
            self.show_seg_labels_checkbox.setEnabled(False)  # 新增
            self.category_filter_combo.setEnabled(False)
            self.goto_spinbox.setEnabled(False)
            if hasattr(self, 'goto_action'):
                self.goto_action.setEnabled(False)
            return

        self.first_btn.setEnabled(has_multiple)
        self.prev_btn.setEnabled(has_multiple)
        self.next_btn.setEnabled(has_multiple)
        self.last_btn.setEnabled(has_multiple)
        self.random_btn.setEnabled(has_multiple)
        self.export_btn.setEnabled(has_images)
        self.reset_btn.setEnabled(has_images)
        self.show_annotations_checkbox.setEnabled(True)
        self.show_bbox_checkbox.setEnabled(True)
        self.show_segmentation_checkbox.setEnabled(True)
        self.show_seg_labels_checkbox.setEnabled(True)  # 新增
        self.category_filter_combo.setEnabled(True)
        self.goto_spinbox.setEnabled(has_images)
        if hasattr(self, 'goto_action'):
            self.goto_action.setEnabled(has_images)

    def update_goto_spinbox(self):
        if self.dataset_info and self.dataset_info.image_files:
            max_count = len(self.dataset_info.image_files)
            self.goto_spinbox.setMaximum(max_count)
            self.goto_spinbox.setValue(self.current_index + 1)
            self.goto_spinbox.setEnabled(True)
        else:
            self.goto_spinbox.setEnabled(False)

    def display_current_image(self):
        if not self.dataset_info or not self.dataset_info.image_files:
            return

        if 0 <= self.current_index < len(self.dataset_info.image_files):
            image_info = self.dataset_info.image_files[self.current_index]
            self.current_image_path = image_info['path']

            # 读取图像
            if os.path.exists(self.current_image_path):
                image = cv2.imread(self.current_image_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # 加载并显示标注
                    self.load_current_annotations(image_info)

                    if self.show_annotations and self.current_annotations:
                        # 绘制标注
                        annotated_image = self.draw_annotations(image_rgb, image_info['info'])
                        self.image_label.set_image(annotated_image)
                    else:
                        self.image_label.set_image(image_rgb)

                    # 更新状态信息
                    filename = Path(self.current_image_path).name
                    self.statusbar.showMessage(
                        f"[{self.current_index + 1}/{len(self.dataset_info.image_files)}] {filename}"
                    )

                    # 更新跳转输入框
                    self.goto_spinbox.setValue(self.current_index + 1)

    def load_current_annotations(self, image_info):
        self.current_annotations = []

        image_id = image_info['id']
        annotations = self.dataset_info.annotations_data.get(image_id, [])

        if not annotations:
            self.annotation_text.setText("无标注信息")
            return

        # 构建显示信息
        annotation_lines = []
        annotation_lines.append(f"图像ID: {image_id}")
        annotation_lines.append(f"图像文件: {Path(self.current_image_path).name}")

        img_info = image_info['info']
        img_width = img_info.get('width', 0)
        img_height = img_info.get('height', 0)
        annotation_lines.append(f"图像尺寸: {img_width} x {img_height}")
        annotation_lines.append("-" * 30)

        current_filter = self.category_filter_combo.currentText()

        for i, ann_data in enumerate(annotations):
            category_id = ann_data.get('category_id', 0)
            category_name = "unknown"

            if category_id in self.dataset_info.categories_data:
                category_name = self.dataset_info.categories_data[category_id]['name']

            # 检查类别筛选
            if current_filter != "显示全部" and current_filter != category_name:
                continue

            # 创建标注对象
            coco_ann = COCOAnnotation(ann_data, category_name)
            self.current_annotations.append(coco_ann)

            # 添加到显示信息
            annotation_lines.append(f"标注 {i + 1}:")
            annotation_lines.append(f"  类别: {category_name} (ID: {category_id})")
            annotation_lines.append(f"  面积: {coco_ann.area}")

            if coco_ann.bbox:
                annotation_lines.append(
                    f"  检测框: [{coco_ann.x1:.1f}, {coco_ann.y1:.1f}, {coco_ann.x2:.1f}, {coco_ann.y2:.1f}]")

            if coco_ann.has_segmentation:
                if coco_ann.is_rle:
                    annotation_lines.append(f"  分割: RLE格式")
                else:
                    seg_count = len(coco_ann.segmentation)
                    annotation_lines.append(f"  分割: {seg_count} 个多边形")

            annotation_lines.append("")

        if len(annotation_lines) <= 4:  # 只有头部信息
            annotation_lines.append("无有效标注")

        self.annotation_text.setText("\n".join(annotation_lines))

    def calculate_polygon_centroid(self, polygon):
        """计算多边形的质心"""
        if len(polygon) < 6:  # 至少需要3个点
            return None

        # 将一维数组转换为二维点数组
        points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]

        # 计算多边形面积和质心
        area = 0
        cx = 0
        cy = 0

        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            xi, yi = points[i]
            xj, yj = points[j]

            cross = xi * yj - xj * yi
            area += cross
            cx += (xi + xj) * cross
            cy += (yi + yj) * cross

        if abs(area) < 1e-10:  # 避免除零
            # 如果面积为0，返回几何中心
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

        area = area / 2
        cx = cx / (6 * area)
        cy = cy / (6 * area)

        return (cx, cy)

    def get_font_for_image(self, image_size):
        """根据图像大小选择合适的字体大小"""
        max_dim = max(image_size)
        if max_dim < 500:
            font_size = 12
        elif max_dim < 1000:
            font_size = 16
        elif max_dim < 2000:
            font_size = 20
        else:
            font_size = 24

        try:
            # 尝试使用系统字体
            import platform
            system = platform.system()
            if system == "Windows":
                font_path = "C:/Windows/Fonts/arial.ttf"
            elif system == "Darwin":  # macOS
                font_path = "/System/Library/Fonts/Arial.ttf"
            else:  # Linux
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
        except:
            pass

        # 如果无法加载字体，使用默认字体
        try:
            return ImageFont.load_default()
        except:
            return None

    def draw_annotations(self, image_rgb, image_info):
        """绘制标注"""
        if not self.current_annotations:
            return image_rgb

        # 转换为PIL图像以便绘制分割掩码
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        img_width = image_info.get('width', image_rgb.shape[1])
        img_height = image_info.get('height', image_rgb.shape[0])

        # 获取字体
        font = self.get_font_for_image((img_width, img_height))

        # 绘制分割掩码
        if self.show_segmentation:
            overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            for ann in self.current_annotations:
                if ann.has_segmentation and not ann.is_rle:
                    color = self.category_colors.get(ann.category_name, (0, 255, 0))
                    color_with_alpha = (*color, 100)  # 半透明

                    for seg in ann.segmentation:
                        if len(seg) >= 6:  # 至少3个点
                            polygon = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                            overlay_draw.polygon(polygon, fill=color_with_alpha)

                            # 绘制分割标签
                            if self.show_seg_labels:
                                centroid = self.calculate_polygon_centroid(seg)
                                if centroid:
                                    cx, cy = centroid
                                    # 确保标签在图像范围内
                                    if 0 <= cx <= img_width and 0 <= cy <= img_height:
                                        self.draw_text_with_background(
                                            overlay_draw, ann.category_name,
                                            (int(cx), int(cy)), color, font
                                        )

            # 合并分割掩码
            pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')

        # 转换回numpy数组
        image_with_seg = np.array(pil_image)

        # 绘制检测框
        if self.show_bbox:
            visualization_data = []
            for ann in self.current_annotations:
                if ann.bbox:
                    color = self.category_colors.get(ann.category_name, (0, 255, 0))
                    visualization_data.append({
                        'box': [ann.x1, ann.y1, ann.x2, ann.y2],
                        'label': ann.category_name,
                        'confidence': 1.0,
                        'color': color
                    })

            if visualization_data:
                return draw_boxes_with_pil(image_with_seg, visualization_data)

        return image_with_seg

    def draw_text_with_background(self, draw, text, position, color, font):
        """绘制带背景的文本"""
        x, y = position

        # 获取文本尺寸
        if font:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width = len(text) * 8
                text_height = 12
        else:
            text_width = len(text) * 8
            text_height = 12

        # 调整位置使文本居中
        text_x = x - text_width // 2
        text_y = y - text_height // 2

        # 绘制半透明背景
        padding = 2
        bg_color = (0, 0, 0, 150)  # 半透明黑色背景
        draw.rectangle(
            [text_x - padding, text_y - padding,
             text_x + text_width + padding, text_y + text_height + padding],
            fill=bg_color
        )

        # 绘制文本
        text_color = (255, 255, 255, 255)  # 白色文本
        if font:
            try:
                draw.text((text_x, text_y), text, fill=text_color, font=font)
            except:
                draw.text((text_x, text_y), text, fill=text_color)
        else:
            draw.text((text_x, text_y), text, fill=text_color)

    def first_image(self):
        if self.dataset_info and self.dataset_info.image_files:
            self.current_index = 0
            self.display_current_image()

    def prev_image(self):
        if self.dataset_info and self.dataset_info.image_files:
            self.current_index = (self.current_index - 1) % len(self.dataset_info.image_files)
            self.display_current_image()

    def next_image(self):
        if self.dataset_info and self.dataset_info.image_files:
            self.current_index = (self.current_index + 1) % len(self.dataset_info.image_files)
            self.display_current_image()

    def last_image(self):
        if self.dataset_info and self.dataset_info.image_files:
            self.current_index = len(self.dataset_info.image_files) - 1
            self.display_current_image()

    def random_image(self):
        if self.dataset_info and len(self.dataset_info.image_files) > 1:
            new_index = random.randint(0, len(self.dataset_info.image_files) - 1)
            while new_index == self.current_index:
                new_index = random.randint(0, len(self.dataset_info.image_files) - 1)
            self.current_index = new_index
            self.display_current_image()

    def goto_image(self):
        if self.dataset_info and self.dataset_info.image_files:
            target_index = self.goto_spinbox.value() - 1
            if 0 <= target_index < len(self.dataset_info.image_files):
                self.current_index = target_index
                self.display_current_image()

    def on_show_annotations_changed(self, state):
        self.show_annotations = (state == Qt.CheckState.Checked.value)
        self.display_current_image()

    def on_display_options_changed(self):
        self.show_bbox = (self.show_bbox_checkbox.checkState() == Qt.CheckState.Checked)
        self.show_segmentation = (self.show_segmentation_checkbox.checkState() == Qt.CheckState.Checked)
        self.show_seg_labels = (self.show_seg_labels_checkbox.checkState() == Qt.CheckState.Checked)  # 新增
        self.display_current_image()

    def on_category_filter_changed(self):
        self.display_current_image()

    def export_dataset_info(self):
        """导出数据集信息为JSON"""
        if not self.dataset_info:
            QMessageBox.warning(self, "警告", "没有数据集信息可导出")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出数据集信息", "", "JSON文件 (*.json);;所有文件 (*)"
        )

        if save_path:
            try:
                export_data = {
                    "dataset_info": {
                        "dataset_path": self.dataset_info.dataset_path,
                        "annotation_file": self.dataset_info.annotation_file,
                        "images_dir": self.dataset_info.images_dir,
                        "total_images": len(self.dataset_info.image_files),
                        "total_annotations": self.dataset_info.total_annotations,
                        "annotation_types": list(self.dataset_info.annotation_types),
                        "categories": [
                            {"id": cat_id, "name": cat_info["name"]}
                            for cat_id, cat_info in self.dataset_info.categories_data.items()
                        ],
                        "category_statistics": self.dataset_info.category_stats
                    },
                    "images": [
                        {
                            "id": img["id"],
                            "path": img["path"],
                            "info": img["info"]
                        }
                        for img in self.dataset_info.image_files
                    ]
                }

                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

                QMessageBox.information(self, "成功", f"数据集信息已导出到: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def update_xy_label(self, x, y):
        if x >= 0 and y >= 0:
            self.xy_label.setText(f"xy: {x}, {y}")
        else:
            self.xy_label.setText("xy: -,-")

    def _init_category_colors(self):
        """为每个类别分配一个固定颜色"""
        if not self.dataset_info or not self.dataset_info.categories_data:
            self.category_colors = {}
            return

        n = len(self.dataset_info.categories_data)
        np.random.seed(42)
        colors = (np.random.uniform(0, 240, size=(n, 3))).astype(int)

        category_names = [cat_info["name"] for cat_info in self.dataset_info.categories_data.values()]
        self.category_colors = {
            name: tuple(map(int, colors[i]))
            for i, name in enumerate(category_names)
        }

    def show_load_log(self):
        log_text = None
        if hasattr(self, 'load_worker') and self.load_worker and hasattr(self.load_worker, 'search_log'):
            log_text = '\n'.join(self.load_worker.search_log)
        if not log_text:
            log_text = "暂无加载日志。请先加载数据集。"

        msg_box = CustomMessageBox("加载日志", "本次加载数据集的详细日志：", log_text, self)
        msg_box.exec()
