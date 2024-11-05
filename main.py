# main.py
import sys
from PyQt5.QtWidgets import QApplication
from app.main_app import MainApp  # MainApp をインポート

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())