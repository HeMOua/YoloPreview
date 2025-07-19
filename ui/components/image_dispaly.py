from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QPixmap, QImage, QColor


class ImageDisplayLabel(QLabel):
    """图像显示标签 - 支持缩放和拖拽"""
    mouse_image_pos_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.setMinimumSize(400, 400)
        self._base_pixmap = None
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._mouse_pos = None
        self.setMouseTracking(True)
        self._dragging = False
        self._drag_start_pos = None
        self._offset_at_start = QPoint(0, 0)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._fit_to_widget = True

    def set_image(self, image_array):
        """设置RGB图片为QPixmap并自适应到label大小"""
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self._base_pixmap = QPixmap.fromImage(q_image)
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._fit_to_widget = True
        self.update_display()

    def update_display(self):
        if self._base_pixmap is None:
            self.clear()
            return
        if self._fit_to_widget:
            # Fit to widget
            widget_w, widget_h = self.width(), self.height()
            pixmap = self._base_pixmap.scaled(widget_w, widget_h, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(pixmap)
        else:
            # Custom zoom and offset
            pixmap = self._base_pixmap
            scaled_w = int(pixmap.width() * self._zoom)
            scaled_h = int(pixmap.height() * self._zoom)
            if scaled_w < 10 or scaled_h < 10:
                return
            scaled_pixmap = pixmap.scaled(scaled_w, scaled_h, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            # 画布
            widget_w, widget_h = self.width(), self.height()
            canvas = QPixmap(widget_w, widget_h)
            canvas.fill(QColor.fromRgb(240, 240, 240))
            # 计算偏移
            x = (widget_w - scaled_w) // 2 + self._offset.x()
            y = (widget_h - scaled_h) // 2 + self._offset.y()
            painter = None
            try:
                painter = QPainter(canvas)
                painter.drawPixmap(x, y, scaled_pixmap)
            finally:
                if painter: painter.end()
            self.setPixmap(canvas)

    def resizeEvent(self, event):
        if self._fit_to_widget:
            self.update_display()
        else:
            self.update_display()
        super().resizeEvent(event)

    def wheelEvent(self, event):
        if self._base_pixmap is None:
            return
        angle = event.angleDelta().y()
        factor = 1.2 if angle > 0 else 1 / 1.2
        self.zoom_at(event.position().toPoint(), factor)

    def zoom_at(self, mouse_pos, factor):
        if self._base_pixmap is None:
            return

        widget_w, widget_h = self.width(), self.height()
        pixmap_w, pixmap_h = self._base_pixmap.width(), self._base_pixmap.height()

        if self._fit_to_widget:
            # 计算当前fit模式下图片左上角在widget中的坐标
            fit_zoom = self._get_fit_zoom()
            display_w = int(pixmap_w * fit_zoom)
            display_h = int(pixmap_h * fit_zoom)
            offset_x = (widget_w - display_w) // 2
            offset_y = (widget_h - display_h) // 2
            # 鼠标在图片上的相对位置
            img_x = (mouse_pos.x() - offset_x) / fit_zoom
            img_y = (mouse_pos.y() - offset_y) / fit_zoom
            self._fit_to_widget = False
            self._zoom = fit_zoom
            self._offset = QPoint(0, 0)
        else:
            # 当前缩放下，鼠标在图片上的相对位置
            img_x = (mouse_pos.x() - ((widget_w - pixmap_w * self._zoom) // 2 + self._offset.x())) / self._zoom
            img_y = (mouse_pos.y() - ((widget_h - pixmap_h * self._zoom) // 2 + self._offset.y())) / self._zoom

        old_zoom = self._zoom
        self._zoom *= factor
        self._zoom = max(0.05, min(10.0, self._zoom))

        # 缩放后，保持鼠标点在图片上的位置不变
        new_display_w = pixmap_w * self._zoom
        new_display_h = pixmap_h * self._zoom
        new_offset_x = mouse_pos.x() - img_x * self._zoom - (widget_w - new_display_w) // 2
        new_offset_y = mouse_pos.y() - img_y * self._zoom - (widget_h - new_display_h) // 2
        self._offset = QPoint(int(new_offset_x), int(new_offset_y))
        self.update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._fit_to_widget and self._base_pixmap is not None:
            self._dragging = True
            self._drag_start_pos = event.position().toPoint()
            self._offset_at_start = QPoint(self._offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.position().toPoint() - self._drag_start_pos
            self._offset = self._offset_at_start + delta
            self.update_display()
        # 计算鼠标在图片中的坐标
        img_x, img_y = self._get_image_xy(event.position().toPoint())
        if img_x is not None and img_y is not None:
            self.mouse_image_pos_changed.emit(int(img_x), int(img_y))
        else:
            self.mouse_image_pos_changed.emit(-1, -1)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.mouse_image_pos_changed.emit(-1, -1)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def reset_view(self):
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._fit_to_widget = True
        self.update_display()

    def _get_fit_zoom(self):
        # 计算fit宽度的缩放比例
        if self._base_pixmap is None:
            return 1.0
        widget_w, widget_h = self.width(), self.height()
        pixmap_w, pixmap_h = self._base_pixmap.width(), self._base_pixmap.height()
        scale_w = widget_w / pixmap_w
        scale_h = widget_h / pixmap_h
        return min(scale_w, scale_h)

    def _get_image_xy(self, widget_pos):
        if self._base_pixmap is None:
            return None, None
        widget_w, widget_h = self.width(), self.height()
        pixmap_w, pixmap_h = self._base_pixmap.width(), self._base_pixmap.height()
        if self._fit_to_widget:
            fit_zoom = self._get_fit_zoom()
            display_w = int(pixmap_w * fit_zoom)
            display_h = int(pixmap_h * fit_zoom)
            offset_x = (widget_w - display_w) // 2
            offset_y = (widget_h - display_h) // 2
            x = (widget_pos.x() - offset_x) / fit_zoom
            y = (widget_pos.y() - offset_y) / fit_zoom
        else:
            x = (widget_pos.x() - ((widget_w - pixmap_w * self._zoom) // 2 + self._offset.x())) / self._zoom
            y = (widget_pos.y() - ((widget_h - pixmap_h * self._zoom) // 2 + self._offset.y())) / self._zoom
        if 0 <= x < pixmap_w and 0 <= y < pixmap_h:
            return x, y
        else:
            return None, None
