from qgis.PyQt.QtCore import QCoreApplication, QSettings, QTranslator, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsProject
import os.path

from .burn_detection_dialog import BurnDetectionDialog


class BurnDetectionPlugin:
    """QGIS Plugin Implementation"""
    
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = '🔥 Burn Detection UNETR'
        self.toolbar = None
        self.dlg = None
        
        # Путь к моделям
        self.model_dir = os.path.join(
            os.path.dirname(os.path.dirname(self.plugin_dir)),
            'checkpoints'
        )
    
    def initGui(self):
        """Create the menu entries and toolbar icons"""
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        
        # Toolbar action
        self.action = QAction(
            QIcon(icon_path if os.path.exists(icon_path) else ''),
            "🔥 Burn Detection", self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)
        
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu(self.menu, self.action)
    
    def unload(self):
        """Removes the plugin menu item and icon"""
        self.iface.removePluginMenu(self.menu, self.action)
        self.iface.removeToolBarIcon(self.action)
    
    def run(self):
        """Run method that performs all the real work"""
        if self.dlg is None:
            self.dlg = BurnDetectionDialog(self.iface.mainWindow(), self.model_dir)
        
        self.dlg.show()
        result = self.dlg.exec_()
        
        if result:
            self.iface.messageBar().pushMessage("Burn Detection", "Analysis completed!", level=1, duration=3)