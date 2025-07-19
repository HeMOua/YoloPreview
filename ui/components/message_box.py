from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton, QDialogButtonBox


class CustomMessageBox(QDialog):
    def __init__(self, title, message, detail_log=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 400)  # 设置初始大小，可以自行调节

        layout = QVBoxLayout(self)

        label = QLabel(message)
        layout.addWidget(label)

        if detail_log:
            details = QTextEdit()
            details.setPlainText(detail_log)
            details.setReadOnly(True)
            layout.addWidget(details)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
