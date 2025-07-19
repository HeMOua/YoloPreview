from PyQt6.QtWidgets import QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QToolBar, QSizePolicy

from ui.coco_dataset_viewer_widget import COCODatasetViewerWidget
from ui.yolo_dataset_viewer_widget import YOLODatasetViewerWidget
from ui.yolo_detection_widget import YOLODetectionWidget


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.toolbars = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("YOLO 目标检测可视化工具 (多引擎支持)")
        self.setGeometry(100, 100, 1400, 900)
        self.central_widget = QTabWidget()
        self.central_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 0;
            }
            QTabBar::tab {
                min-width: 0px;
                padding: 10px;
                background: #e0e0e0;
            }
            QTabBar::tab:selected {
                background: #ffffff;
            }
            QTabBar { qproperty-expanding: 1; }
        """)
        self.setCentralWidget(self.central_widget)

        # 调整tab控件
        tab_bar = self.central_widget.tabBar()
        tab_bar.setExpanding(True)
        tab_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


        # YOLO检测
        self.central_widget.addTab(YOLODetectionWidget(), "YOLO检测")
        # YOLO数据集预览
        self.central_widget.addTab(YOLODatasetViewerWidget(), "YOLO数据集预览")
        # COCO数据集预览
        self.central_widget.addTab(COCODatasetViewerWidget(), "COCO数据集预览")
