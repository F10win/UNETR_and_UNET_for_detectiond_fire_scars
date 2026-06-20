# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QFileDialog, QComboBox, 
                             QGroupBox, QSpinBox, QDoubleSpinBox, QTextEdit, 
                             QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt
from qgis.core import QgsMapLayerProxyModel, QgsProject
from qgis.gui import QgsMapLayerComboBox
from pathlib import Path
import os


class BurnDetectionDialog(QDialog):
    def __init__(self, parent=None, model_dir=None):
        super().__init__(parent)
        self.model_dir = model_dir
        self.setWindowTitle("🔥 Детекция лесных гарей (UNETR)")
        self.setMinimumSize(700, 800)
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # === ВХОДНЫЕ СЛОИ ===
        group_inputs = QGroupBox("📂 Входные данные")
        layout_inputs = QVBoxLayout()
        
        # Маска
        layout_mask = QHBoxLayout()
        layout_mask.addWidget(QLabel("Слой-маска (опционально):"))
        self.combo_mask = QgsMapLayerComboBox()
        self.combo_mask.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.combo_mask.setAllowEmptyLayer(True)
        layout_mask.addWidget(self.combo_mask)
        layout_inputs.addLayout(layout_mask)
        
        # Растр
        layout_raster = QHBoxLayout()
        layout_raster.addWidget(QLabel("Растровый слой (Sentinel-2):"))
        self.combo_raster = QgsMapLayerComboBox()
        self.combo_raster.setFilters(QgsMapLayerProxyModel.RasterLayer)
        layout_raster.addWidget(self.combo_raster)
        layout_inputs.addLayout(layout_raster)
        
        group_inputs.setLayout(layout_inputs)
        layout.addWidget(group_inputs)
        
        # === ВЫБОР МОДЕЛИ ===
        group_model = QGroupBox("🤖 Модель")
        layout_model = QVBoxLayout()
        
        self.combo_model = QComboBox()
        self.combo_model.addItem("3 канала (SWIR, NIR, Red)", "3ch")
        self.combo_model.addItem("11 каналов (B2-B9, B8A, B11, B12)", "11ch")
        layout_model.addWidget(self.combo_model)
        
        self.label_model_info = QLabel("📝 3 канала: SWIR (B12), NIR (B8), Red (B4)")
        self.label_model_info.setStyleSheet("color: blue; font-style: italic; background-color: #f0f8ff; padding: 5px;")
        layout_model.addWidget(self.label_model_info)
        
        group_model.setLayout(layout_model)
        layout.addWidget(group_model)
        
        # === ПАРАМЕТРЫ ===
        group_params = QGroupBox("⚙️ Параметры")
        layout_params = QVBoxLayout()
        
        # THRESHOLD
        layout_thresh = QHBoxLayout()
        layout_thresh.addWidget(QLabel("THRESHOLD (порог):"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 1.0)
        self.spin_threshold.setValue(0.5)
        self.spin_threshold.setSingleStep(0.05)
        layout_thresh.addWidget(self.spin_threshold)
        layout_params.addLayout(layout_thresh)
        
        # MIN_BURN_AREA_PX
        layout_area = QHBoxLayout()
        layout_area.addWidget(QLabel("MIN_BURN_AREA_PX:"))
        self.spin_area = QSpinBox()
        self.spin_area.setRange(1, 10000)
        self.spin_area.setValue(100)
        layout_area.addWidget(self.spin_area)
        layout_params.addLayout(layout_area)
        
        # PIXEL_RESOLUTION
        layout_res = QHBoxLayout()
        layout_res.addWidget(QLabel("PIXEL_RESOLUTION_M:"))
        self.spin_resolution = QSpinBox()
        self.spin_resolution.setRange(1, 100)
        self.spin_resolution.setValue(10)
        layout_res.addWidget(self.spin_resolution)
        layout_params.addLayout(layout_res)
        
        # Папка вывода
        layout_output = QHBoxLayout()
        layout_output.addWidget(QLabel("Папка для результатов:"))
        self.line_output = QLineEdit()
        self.line_output.setPlaceholderText("Выберите папку...")
        layout_output.addWidget(self.line_output)
        self.btn_output = QPushButton("📁")
        self.btn_output.setMaximumWidth(40)
        layout_output.addWidget(self.btn_output)
        layout_params.addLayout(layout_output)
        
        group_params.setLayout(layout_params)
        layout.addWidget(group_params)
        
        # === ПРОГРЕСС ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # === ЛОГ ===
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setPlaceholderText("Лог выполнения...")
        self.text_log.setMaximumHeight(150)
        layout.addWidget(self.text_log)
        
        # === КНОПКИ ===
        layout_buttons = QHBoxLayout()
        self.btn_run = QPushButton("🚀 Запустить анализ")
        self.btn_run.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 10px;")
        layout_buttons.addWidget(self.btn_run)
        
        self.btn_cancel = QPushButton("❌ Отмена")
        layout_buttons.addWidget(self.btn_cancel)
        
        layout.addLayout(layout_buttons)
        
        self.setLayout(layout)
    
    def setup_connections(self):
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_output.clicked.connect(self.select_output_folder)
        self.combo_model.currentIndexChanged.connect(self.update_model_info)
    
    def update_model_info(self):
        model_type = self.combo_model.currentData()
        if model_type == "3ch":
            self.label_model_info.setText("📝 3 канала: SWIR (B12), NIR (B8), Red (B4)")
        else:
            self.label_model_info.setText("📝 11 каналов: B2-B9, B8A, B11, B12")
    
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для результатов")
        if folder:
            self.line_output.setText(folder)
    
    def log_message(self, message):
        self.text_log.append(message)
        self.text_log.verticalScrollBar().setValue(self.text_log.verticalScrollBar().maximum())
    
    def run_analysis(self):
        # Здесь будет вызов inference_engine
        QMessageBox.information(self, "Инфо", "Запуск анализа... (здесь будет код инференции)")
        self.accept()