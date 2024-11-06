# main.py
import sys
from PyQt5.QtWidgets import QApplication
from app.main_app import MainApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
