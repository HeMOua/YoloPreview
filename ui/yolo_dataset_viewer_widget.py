import os
import random
import json
import yaml
from pathlib import Path
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit,
    QSplitter, QToolBar, QMessageBox, QGroupBox, QScrollArea, QCheckBox,
    QStatusBar, QComboBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction as QActionGui
import numpy as np

from ui.components.image_dispaly import ImageDisplayLabel
from utils.paint import draw_boxes_with_pil


class YOLODatasetInfo:
    """YOLO数据集信息类"""

    def __init__(self):
        self.dataset_path = ""
        self.config_file = ""
        self.images_dirs = {}  # {'train': 'path/to/train', 'val': 'path/to/val'}
        self.labels_dirs = {}  # {'train': 'path/to/train_labels', 'val': 'path/to/val_labels'}
        self.classes = []
        self.image_files = {}  # {'train': [files], 'val': [files]}
        self.label_files = {}  # {'train': [files], 'val': [files]}
        self.class_stats = {}
        self.total_annotations = 0
        self.current_split = "train"  # 当前查看的数据集分割
        self.available_splits = []


class AnnotationBox:
    """标注框类"""

    def __init__(self, class_id, class_name, x_center, y_center, width, height,
                 x1=None, y1=None, x2=None, y2=None, confidence=1.0):
        self.class_id = class_id
        self.class_name = class_name
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.x1 = x1 if x1 is not None else x_center - width / 2
        self.y1 = y1 if y1 is not None else y_center - height / 2
        self.x2 = x2 if x2 is not None else x_center + width / 2
        self.y2 = y2 if y2 is not None else y_center + height / 2
        self.confidence = confidence


class DatasetLoadWorker(QThread):
    """数据集加载工作线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self._should_stop = False

    def run(self):
        try:
            dataset_info = YOLODatasetInfo()
            dataset_info.dataset_path = self.dataset_path
            dataset_path = Path(self.dataset_path)

            self.progress.emit("正在查找配置文件...")

            # 查找配置文件
            config_file = self.find_config_file(dataset_path)

            if config_file:
                dataset_info.config_file = str(config_file)
                self.progress.emit(f"找到配置文件: {config_file.name}")

                # 从配置文件加载信息
                config_data = self.load_config_file(config_file)
                dataset_info.classes = config_data.get('names', [])

                # 获取数据集路径（配置文件中的path字段）
                if 'path' in config_data:
                    base_path = Path(config_data['path'])
                    if not base_path.is_absolute():
                        base_path = dataset_path / base_path
                else:
                    base_path = dataset_path

                # 获取train和val路径
                splits_info = {}
                for split in ['train', 'val', 'test']:
                    if split in config_data:
                        split_path = config_data[split]
                        if isinstance(split_path, str):
                            full_path = base_path / split_path
                            if full_path.exists():
                                splits_info[split] = str(full_path)

                if not splits_info:
                    # 如果配置文件中没有指定，尝试默认路径
                    splits_info = self.find_default_splits(base_path)

            else:
                self.progress.emit("未找到配置文件，使用默认扫描...")
                # 没有配置文件，使用传统扫描方式
                splits_info = self.find_default_splits(dataset_path)

                # 尝试加载类别文件
                dataset_info.classes = self.load_classes_from_file(dataset_path)

            if not splits_info:
                self.error.emit("未找到有效的数据集目录结构")
                return

            dataset_info.available_splits = list(splits_info.keys())
            dataset_info.current_split = dataset_info.available_splits[0]

            # 扫描每个分割的图像和标签
            for split, images_dir in splits_info.items():
                self.progress.emit(f"正在扫描 {split} 分割...")

                images_path = Path(images_dir)
                dataset_info.images_dirs[split] = str(images_path)

                # 查找对应的labels目录
                labels_path = self.find_labels_dir(images_path, dataset_path)
                dataset_info.labels_dirs[split] = str(labels_path) if labels_path else ""

                # 扫描图像文件
                image_files = self.scan_image_files(images_path)
                dataset_info.image_files[split] = image_files

                # 扫描标签文件
                if labels_path:
                    label_files = self.scan_label_files(image_files, images_path, labels_path)
                    dataset_info.label_files[split] = label_files
                else:
                    dataset_info.label_files[split] = [""] * len(image_files)

            # 如果没有从配置文件加载到类别，尝试从标签推断
            if not dataset_info.classes:
                self.progress.emit("正在从标注文件推断类别...")
                dataset_info.classes = self.infer_classes_from_labels(dataset_info)

            # 统计信息
            self.progress.emit("正在统计数据集信息...")
            self.calculate_statistics(dataset_info)

            self.finished.emit(dataset_info)

        except Exception as e:
            self.error.emit(f"数据集加载失败: {str(e)}")

    def find_config_file(self, dataset_path):
        """查找配置文件"""
        config_files = ['data.yaml', 'dataset.yaml', 'config.yaml', 'data.yml', 'dataset.yml']

        for config_name in config_files:
            config_path = dataset_path / config_name
            if config_path.exists():
                return config_path
        return None

    def load_config_file(self, config_file):
        """加载YAML配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}

    def find_default_splits(self, base_path):
        """查找默认的数据集分割目录"""
        splits_info = {}

        # 检查 images 目录下的子目录
        images_dir = base_path / 'images'
        if images_dir.exists():
            for split in ['train', 'val', 'test']:
                split_dir = images_dir / split
                if split_dir.exists():
                    splits_info[split] = str(split_dir)

        # 如果没找到，检查根目录下的分割目录
        if not splits_info:
            for split in ['train', 'val', 'test']:
                split_dir = base_path / split
                if split_dir.exists():
                    splits_info[split] = str(split_dir)

        # 如果还没找到，检查是否直接包含图像
        if not splits_info:
            if self.has_images(base_path):
                splits_info['all'] = str(base_path)

        return splits_info

    def find_labels_dir(self, images_path, dataset_root):
        """查找对应的labels目录"""
        # 从images路径推断labels路径
        images_path = Path(images_path)
        dataset_root = Path(dataset_root)

        # 情况1: images/train -> labels/train
        if images_path.parent.name == 'images':
            labels_parent = images_path.parent.parent / 'labels'
            labels_dir = labels_parent / images_path.name
            if labels_dir.exists():
                return labels_dir

        # 情况2: train -> train_labels 或 labels/train
        split_name = images_path.name

        # 尝试 labels/split_name
        labels_dir = dataset_root / 'labels' / split_name
        if labels_dir.exists():
            return labels_dir

        # 尝试 split_name_labels
        labels_dir = dataset_root / f"{split_name}_labels"
        if labels_dir.exists():
            return labels_dir

        # 尝试同一目录下查找txt文件
        if list(images_path.glob("*.txt")):
            return images_path

        return None

    def scan_image_files(self, images_dir):
        """扫描图像文件，避免重复"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for f in images_dir.glob("*"):
            if f.suffix.lower() in image_extensions:
                image_files.append(f)
        return [str(f) for f in sorted(image_files)]

    def scan_label_files(self, image_files, images_dir, labels_dir):
        """扫描对应的标签文件"""
        label_files = []
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        for img_file in image_files:
            img_path = Path(img_file)
            label_name = img_path.stem + '.txt'
            label_path = labels_dir / label_name

            if label_path.exists():
                label_files.append(str(label_path))
            else:
                label_files.append("")  # 没有对应的标签文件

        return label_files

    def load_classes_from_file(self, dataset_path):
        """从类别文件加载类别"""
        class_files = ['classes.txt', 'class.names', 'obj.names', 'coco.names']

        for class_file in class_files:
            class_path = dataset_path / class_file
            if class_path.exists():
                try:
                    with open(class_path, 'r', encoding='utf-8') as f:
                        return [line.strip() for line in f.readlines() if line.strip()]
                except:
                    continue
        return []

    def infer_classes_from_labels(self, dataset_info):
        """从标签文件推断类别"""
        class_ids = set()

        for split in dataset_info.available_splits:
            for label_file in dataset_info.label_files[split]:
                if label_file and os.path.exists(label_file):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_ids.add(int(parts[0]))
                    except:
                        continue

        return [f"class_{i}" for i in sorted(class_ids)]

    def calculate_statistics(self, dataset_info):
        """计算统计信息"""
        class_stats = {cls: 0 for cls in dataset_info.classes}
        total_annotations = 0

        for split in dataset_info.available_splits:
            for label_file in dataset_info.label_files[split]:
                if label_file and os.path.exists(label_file):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_id = int(parts[0])
                                    if 0 <= class_id < len(dataset_info.classes):
                                        class_stats[dataset_info.classes[class_id]] += 1
                                        total_annotations += 1
                    except:
                        continue

        dataset_info.class_stats = class_stats
        dataset_info.total_annotations = total_annotations

    def has_images(self, path):
        """检查目录是否包含图像文件"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for ext in image_extensions:
            if list(path.glob(f"*{ext}")) or list(path.glob(f"*{ext.upper()}")):
                return True
        return False

    def stop(self):
        self._should_stop = True


class YOLODatasetViewerWidget(QWidget):
    """YOLO数据集预览控件"""

    def __init__(self):
        super().__init__()
        self.dataset_info = None
        self.current_image_path = None
        self.current_index = 0
        self.current_annotations = []
        self.load_worker = None
        self.show_annotations = True
        self.selected_classes = set()
        self.class_colors = {}

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
        self.statusbar.showMessage("请选择YOLO数据集目录")
        self.update_button_states()  # 初始化时禁用按钮

    def create_toolbar(self):
        toolbar = self.toolbar
        toolbar.clear()

        # 加载数据集
        load_dataset_action = QActionGui("加载数据集", self)
        load_dataset_action.triggered.connect(self.load_dataset)
        toolbar.addAction(load_dataset_action)

        toolbar.addSeparator()

        # 数据集分割选择
        toolbar.addWidget(QLabel("数据集分割:"))
        self.split_combo = QComboBox()
        self.split_combo.setMinimumWidth(80)
        toolbar.addWidget(self.split_combo)

        toolbar.addSeparator()

        # 显示控制
        self.show_annotations_checkbox = QCheckBox("显示标注")
        self.show_annotations_checkbox.setChecked(True)
        toolbar.addWidget(self.show_annotations_checkbox)

        toolbar.addSeparator()

        # 类别筛选
        toolbar.addWidget(QLabel("筛选类别:"))
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItem("显示全部")
        self.class_filter_combo.setMinimumWidth(120)
        toolbar.addWidget(self.class_filter_combo)

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
        self.class_filter_combo.currentTextChanged.connect(self.on_class_filter_changed)
        self.split_combo.currentTextChanged.connect(self.on_split_changed)

        # 鼠标位置
        self.image_label.mouse_image_pos_changed.connect(self.update_xy_label)

    def load_dataset(self):
        dataset_path = QFileDialog.getExistingDirectory(self, "选择YOLO数据集目录")
        if dataset_path:
            self.statusbar.showMessage("正在加载数据集...")

            self.load_worker = DatasetLoadWorker(dataset_path)
            self.load_worker.progress.connect(self.on_load_progress)
            self.load_worker.finished.connect(self.on_load_finished)
            self.load_worker.error.connect(self.on_load_error)
            self.load_worker.start()

    def on_load_progress(self, message):
        self.statusbar.showMessage(message)

    def on_load_finished(self, dataset_info):
        self.dataset_info = dataset_info
        self.current_index = 0
        self._init_class_colors()

        # 更新界面
        self.update_dataset_info_display()
        self.update_split_combo()
        self.update_class_filter_combo()
        self.update_button_states()
        self.update_goto_spinbox()

        # 显示第一张图像
        self.display_current_image()

        total_images = sum(len(files) for files in self.dataset_info.image_files.values())
        self.statusbar.showMessage(f"数据集加载完成，共 {total_images} 张图像")

    def on_load_error(self, error_message):
        try:
            QMessageBox.critical(self, "加载错误", error_message)
            self.statusbar.showMessage("数据集加载失败")
        except Exception as e:
            print(f"on_load_error exception: {e}")
        self.update_button_states()  # 加载失败时禁用按钮

    def update_dataset_info_display(self):
        if not self.dataset_info:
            return

        info_lines = []
        info_lines.append(f"数据集路径: {self.dataset_info.dataset_path}")

        if self.dataset_info.config_file:
            info_lines.append(f"配置文件: {Path(self.dataset_info.config_file).name}")

        info_lines.append(f"数据集分割: {', '.join(self.dataset_info.available_splits)}")

        # 各分割的图像数量
        for split in self.dataset_info.available_splits:
            img_count = len(self.dataset_info.image_files.get(split, []))
            info_lines.append(f"  {split}: {img_count} 张图像")

        info_lines.append(f"类别数量: {len(self.dataset_info.classes)}")
        info_lines.append(f"总标注数: {self.dataset_info.total_annotations}")

        if self.dataset_info.classes:
            info_lines.append("\n类别列表:")
            for i, cls in enumerate(self.dataset_info.classes):
                info_lines.append(f"  {i}: {cls}")

        self.dataset_info_text.setText("\n".join(info_lines))

        # 更新统计信息
        stats_lines = []
        if self.dataset_info.class_stats:
            stats_lines.append("各类别标注数量:")
            for cls, count in self.dataset_info.class_stats.items():
                percentage = (
                            count / self.dataset_info.total_annotations * 100) if self.dataset_info.total_annotations > 0 else 0
                stats_lines.append(f"  {cls}: {count} ({percentage:.1f}%)")

        self.stats_text.setText("\n".join(stats_lines))

    def update_split_combo(self):
        self.split_combo.clear()
        if self.dataset_info and self.dataset_info.available_splits:
            for split in self.dataset_info.available_splits:
                self.split_combo.addItem(split)

    def update_class_filter_combo(self):
        self.class_filter_combo.clear()
        self.class_filter_combo.addItem("显示全部")

        if self.dataset_info and self.dataset_info.classes:
            for cls in self.dataset_info.classes:
                self.class_filter_combo.addItem(cls)

    def update_button_states(self):
        # 如果没有数据集，全部禁用
        if not self.dataset_info:
            self.first_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.last_btn.setEnabled(False)
            self.random_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            # 禁用toolbar相关控件
            self.split_combo.setEnabled(False)
            self.show_annotations_checkbox.setEnabled(False)
            self.class_filter_combo.setEnabled(False)
            self.goto_spinbox.setEnabled(False)
            if hasattr(self, 'goto_action'):
                self.goto_action.setEnabled(False)
            return
        current_split = self.dataset_info.current_split if self.dataset_info else None
        has_images = (self.dataset_info and current_split and
                      len(self.dataset_info.image_files.get(current_split, [])) > 0)
        has_multiple = (has_images and
                        len(self.dataset_info.image_files.get(current_split, [])) > 1)

        self.first_btn.setEnabled(has_multiple)
        self.prev_btn.setEnabled(has_multiple)
        self.next_btn.setEnabled(has_multiple)
        self.last_btn.setEnabled(has_multiple)
        self.random_btn.setEnabled(has_multiple)
        self.export_btn.setEnabled(has_images)
        self.reset_btn.setEnabled(has_images)
        # 启用toolbar相关控件
        self.split_combo.setEnabled(True)
        self.show_annotations_checkbox.setEnabled(True)
        self.class_filter_combo.setEnabled(True)
        # goto_spinbox和GO按钮根据has_images
        self.goto_spinbox.setEnabled(has_images)
        if hasattr(self, 'goto_action'):
            self.goto_action.setEnabled(has_images)

    def update_goto_spinbox(self):
        if (self.dataset_info and self.dataset_info.current_split and
                self.dataset_info.image_files.get(self.dataset_info.current_split)):

            max_count = len(self.dataset_info.image_files[self.dataset_info.current_split])
            self.goto_spinbox.setMaximum(max_count)
            self.goto_spinbox.setValue(self.current_index + 1)
            self.goto_spinbox.setEnabled(True)
        else:
            self.goto_spinbox.setEnabled(False)

    def get_current_image_files(self):
        """获取当前分割的图像文件列表"""
        if not self.dataset_info or not self.dataset_info.current_split:
            return []
        return self.dataset_info.image_files.get(self.dataset_info.current_split, [])

    def get_current_label_files(self):
        """获取当前分割的标签文件列表"""
        if not self.dataset_info or not self.dataset_info.current_split:
            return []
        return self.dataset_info.label_files.get(self.dataset_info.current_split, [])

    def display_current_image(self):
        image_files = self.get_current_image_files()
        if not image_files:
            return

        if 0 <= self.current_index < len(image_files):
            self.current_image_path = image_files[self.current_index]

            # 读取图像
            if os.path.exists(self.current_image_path):
                image = cv2.imread(self.current_image_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # 加载并显示标注
                    self.load_current_annotations()

                    if self.show_annotations and self.current_annotations:
                        # 绘制标注框
                        annotated_image = self.draw_annotations(image_rgb)
                        self.image_label.set_image(annotated_image)
                    else:
                        self.image_label.set_image(image_rgb)

                    # 更新状态信息
                    filename = Path(self.current_image_path).name
                    current_split = self.dataset_info.current_split
                    self.statusbar.showMessage(
                        f"[{current_split}] [{self.current_index + 1}/{len(image_files)}] {filename}"
                    )

                    # 更新跳转输入框
                    self.goto_spinbox.setValue(self.current_index + 1)

    def load_current_annotations(self):
        self.current_annotations = []

        label_files = self.get_current_label_files()
        if not label_files or self.current_index >= len(label_files):
            self.annotation_text.setText("无标注文件")
            return

        label_file = label_files[self.current_index]
        if not label_file or not os.path.exists(label_file):
            self.annotation_text.setText("无标注文件")
            return

        # 获取图像尺寸
        image = cv2.imread(self.current_image_path)
        if image is None:
            self.annotation_text.setText("无法读取图像")
            return

        img_height, img_width = image.shape[:2]

        try:
            annotation_lines = []
            annotation_lines.append(f"标注文件: {Path(label_file).name}")
            annotation_lines.append(f"图像尺寸: {img_width} x {img_height}")
            annotation_lines.append(f"数据集分割: {self.dataset_info.current_split}")
            annotation_lines.append("-" * 30)

            with open(label_file, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # 转换为绝对坐标
                        abs_x_center = x_center * img_width
                        abs_y_center = y_center * img_height
                        abs_width = width * img_width
                        abs_height = height * img_height

                        x1 = abs_x_center - abs_width / 2
                        y1 = abs_y_center - abs_height / 2
                        x2 = abs_x_center + abs_width / 2
                        y2 = abs_y_center + abs_height / 2

                        # 获取类别名称
                        class_name = "unknown"
                        if (self.dataset_info.classes and
                                0 <= class_id < len(self.dataset_info.classes)):
                            class_name = self.dataset_info.classes[class_id]

                        # 创建标注框对象
                        box = AnnotationBox(
                            class_id=class_id,
                            class_name=class_name,
                            x_center=abs_x_center,
                            y_center=abs_y_center,
                            width=abs_width,
                            height=abs_height,
                            x1=x1, y1=y1, x2=x2, y2=y2
                        )

                        # 检查类别筛选
                        current_filter = self.class_filter_combo.currentText()
                        if current_filter == "显示全部" or current_filter == class_name:
                            self.current_annotations.append(box)

                        # 添加到显示信息
                        annotation_lines.append(f"标注 {i + 1}:")
                        annotation_lines.append(f"  类别: {class_name} (ID: {class_id})")
                        annotation_lines.append(f"  相对坐标: ({x_center:.3f}, {y_center:.3f})")
                        annotation_lines.append(f"  相对尺寸: {width:.3f} x {height:.3f}")
                        annotation_lines.append(f"  绝对坐标: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                        annotation_lines.append("")

            if len(annotation_lines) <= 4:  # 只有头部信息
                annotation_lines.append("无有效标注")

            self.annotation_text.setText("\n".join(annotation_lines))

        except Exception as e:
            self.annotation_text.setText(f"标注文件读取错误: {str(e)}")

    def draw_annotations(self, image_rgb):
        """绘制标注框"""
        if not self.current_annotations:
            return image_rgb

        # 转换为绘制函数所需的格式
        visualization_data = []
        for box in self.current_annotations:
            color = self.class_colors.get(box.class_name, (0, 255, 0))  # 默认绿色
            visualization_data.append({
                'box': [box.x1, box.y1, box.x2, box.y2],
                'label': box.class_name,
                'confidence': box.confidence,
                'color': color
            })

        return draw_boxes_with_pil(image_rgb, visualization_data)

    def on_split_changed(self):
        """数据集分割改变时的处理"""
        if not self.dataset_info:
            return

        new_split = self.split_combo.currentText()
        if new_split and new_split in self.dataset_info.available_splits:
            self.dataset_info.current_split = new_split
            self.current_index = 0
            self.update_button_states()
            self.update_goto_spinbox()
            self.display_current_image()

    def first_image(self):
        image_files = self.get_current_image_files()
        if image_files:
            self.current_index = 0
            self.display_current_image()

    def prev_image(self):
        image_files = self.get_current_image_files()
        if image_files:
            self.current_index = (self.current_index - 1) % len(image_files)
            self.display_current_image()

    def next_image(self):
        image_files = self.get_current_image_files()
        if image_files:
            self.current_index = (self.current_index + 1) % len(image_files)
            self.display_current_image()

    def last_image(self):
        image_files = self.get_current_image_files()
        if image_files:
            self.current_index = len(image_files) - 1
            self.display_current_image()

    def random_image(self):
        image_files = self.get_current_image_files()
        if len(image_files) > 1:
            new_index = random.randint(0, len(image_files) - 1)
            while new_index == self.current_index:
                new_index = random.randint(0, len(image_files) - 1)
            self.current_index = new_index
            self.display_current_image()

    def goto_image(self):
        image_files = self.get_current_image_files()
        if image_files:
            target_index = self.goto_spinbox.value() - 1
            if 0 <= target_index < len(image_files):
                self.current_index = target_index
                self.display_current_image()

    def on_show_annotations_changed(self, state):
        self.show_annotations = (state == Qt.CheckState.Checked.value)
        self.display_current_image()

    def on_class_filter_changed(self):
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
                # 计算总图像数
                total_images = sum(len(files) for files in self.dataset_info.image_files.values())

                export_data = {
                    "dataset_info": {
                        "dataset_path": self.dataset_info.dataset_path,
                        "config_file": self.dataset_info.config_file,
                        "available_splits": self.dataset_info.available_splits,
                        "total_images": total_images,
                        "total_annotations": self.dataset_info.total_annotations,
                        "classes": self.dataset_info.classes,
                        "class_statistics": self.dataset_info.class_stats,
                        "splits_info": {
                            split: {
                                "images_dir": self.dataset_info.images_dirs.get(split, ""),
                                "labels_dir": self.dataset_info.labels_dirs.get(split, ""),
                                "image_count": len(self.dataset_info.image_files.get(split, []))
                            }
                            for split in self.dataset_info.available_splits
                        }
                    },
                    "files": {
                        split: {
                            "images": self.dataset_info.image_files.get(split, []),
                            "labels": self.dataset_info.label_files.get(split, [])
                        }
                        for split in self.dataset_info.available_splits
                    }
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

    def _init_class_colors(self):
        """为每个类别分配一个固定颜色"""
        if not self.dataset_info or not self.dataset_info.classes:
            self.class_colors = {}
            return
        n = len(self.dataset_info.classes)
        np.random.seed(42)
        colors = (np.random.uniform(0, 240, size=(n, 3))).astype(int)
        self.class_colors = {cls: tuple(map(int, colors[i])) for i, cls in enumerate(self.dataset_info.classes)}
