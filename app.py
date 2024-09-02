import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, 
                             QVBoxLayout, QWidget, QSlider, QHBoxLayout, QFrame, QGroupBox,
                             QStatusBar, QToolTip)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt
from psr_image_enhancer import enhance_psr_image

class PSREnhancerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PSR Image Enhancer')
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
                color: #ecf0f1;
            }

            QLabel {
                color: #ecf0f1;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }

            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
                margin: 5px;
            }

            QPushButton:hover {
                background-color: #2980b9;
            }

            QPushButton:pressed {
                background-color: #21618c;
            }

            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #34495e;
                height: 10px;
                border-radius: 4px;
            }

            QSlider::sub-page:horizontal {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3498db, stop: 1 #2980b9);
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ecf0f1, stop:1 #bdc3c7);
                border: 1px solid #777;
                width: 18px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 9px;
            }

            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f39c12, stop:1 #e67e22);
            }

            QGroupBox {
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: #ecf0f1;
                padding: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }

            QStatusBar {
                color: #ecf0f1;
                font-weight: bold;
                background-color: #34495e;
                padding: 5px;
            }

            QToolTip {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #2c3e50;
                padding: 5px;
            }

            #imageLabel {
                background-color: #34495e;
                border: 2px solid #3498db;
                border-radius: 10px;
                padding: 10px;
            }

            QMainWindow::separator {
                background-color: #34495e;
                width: 1px;
                height: 1px;
            }

            QScrollBar:vertical {
                border: none;
                background: #34495e;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }

            QScrollBar::handle:vertical {
                background: #3498db;
                min-height: 20px;
                border-radius: 5px;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar:horizontal {
                border: none;
                background: #34495e;
                height: 10px;
                margin: 0px 0px 0px 0px;
            }

            QScrollBar::handle:horizontal {
                background: #3498db;
                min-width: 20px;
                border-radius: 5px;
            }

            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)

        main_layout = QVBoxLayout()

        # Button layout
        button_group = QGroupBox("Image Controls")
        button_layout = QHBoxLayout()
        self.loadButton = QPushButton('Load Image', self)
        self.loadButton.setIcon(QIcon.fromTheme("document-open"))
        self.loadButton.clicked.connect(self.loadImage)
        self.loadButton.setToolTip("Load an image file")
        self.enhanceButton = QPushButton('Enhance Image', self)
        self.enhanceButton.setIcon(QIcon.fromTheme("edit-redo"))
        self.enhanceButton.clicked.connect(self.enhanceImage)
        self.enhanceButton.setToolTip("Apply enhancement to the loaded image")
        button_layout.addWidget(self.loadButton)
        button_layout.addWidget(self.enhanceButton)
        button_group.setLayout(button_layout)
        main_layout.addWidget(button_group)

        # Slider layouts
        slider_group = QGroupBox("Enhancement Controls")
        slider_layout = QVBoxLayout()
        
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel('Gamma:'))
        self.gammaSlider = QSlider(Qt.Horizontal)
        self.gammaSlider.setMinimum(50)
        self.gammaSlider.setMaximum(200)
        self.gammaSlider.setValue(120)
        self.gammaSlider.setTickPosition(QSlider.TicksBelow)
        self.gammaSlider.setTickInterval(10)
        self.gammaSlider.setToolTip("Adjust image gamma")
        self.gammaSlider.valueChanged.connect(self.updateGammaLabel)
        gamma_layout.addWidget(self.gammaSlider)
        self.gammaLabel = QLabel('1.20')
        gamma_layout.addWidget(self.gammaLabel)
        slider_layout.addLayout(gamma_layout)
        
        sharpen_layout = QHBoxLayout()
        sharpen_layout.addWidget(QLabel('Sharpen:'))
        self.sharpenSlider = QSlider(Qt.Horizontal)
        self.sharpenSlider.setMinimum(0)
        self.sharpenSlider.setMaximum(100)
        self.sharpenSlider.setValue(50)
        self.sharpenSlider.setTickPosition(QSlider.TicksBelow)
        self.sharpenSlider.setTickInterval(10)
        self.sharpenSlider.setToolTip("Adjust image sharpness")
        self.sharpenSlider.valueChanged.connect(self.updateSharpenLabel)
        sharpen_layout.addWidget(self.sharpenSlider)
        self.sharpenLabel = QLabel('50%')
        sharpen_layout.addWidget(self.sharpenLabel)
        slider_layout.addLayout(sharpen_layout)
        
        slider_group.setLayout(slider_layout)
        main_layout.addWidget(slider_group)

        # Image display
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setObjectName("imageLabel")
        main_layout.addWidget(self.imageLabel)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        self.imagePath = None

    def updateGammaLabel(self, value):
        gamma = value / 100
        self.gammaLabel.setText(f'{gamma:.2f}')

    def updateSharpenLabel(self, value):
        self.sharpenLabel.setText(f'{value}%')

    def loadImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Image", "", 
                                                  "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if fileName:
            self.imagePath = fileName
            self.displayImage(fileName)
            self.statusBar.showMessage(f"Image loaded: {fileName}")

    def enhanceImage(self):
        if self.imagePath:
            gamma = self.gammaSlider.value() / 100
            sharpen_strength = self.sharpenSlider.value() / 100
            self.statusBar.showMessage("Enhancing image...")
            enhanced_image = enhance_psr_image(self.imagePath, gamma=gamma, sharpen_strength=sharpen_strength)
            self.displayImage(enhanced_image, is_enhanced=True)
            self.statusBar.showMessage("Image enhancement complete")
        else:
            self.statusBar.showMessage("Please load an image first")

    def displayImage(self, image, is_enhanced=False):
        if is_enhanced:
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            q_image = QImage(image)
        
        pixmap = QPixmap.fromImage(q_image)
        self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), 
                                                aspectRatioMode=Qt.KeepAspectRatio, 
                                                transformMode=Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    QToolTip.setFont(QFont('SansSerif', 10))
    ex = PSREnhancerApp()
    ex.show()
    sys.exit(app.exec_())