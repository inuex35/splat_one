# camera_models.py

import os
import json
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QPushButton, QLabel, QComboBox, QDialogButtonBox, QMessageBox
)

class CameraModelEditor(QDialog):
    """Camera model editor dialog"""
    def __init__(self, camera_models, workdir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Camera Models")
        self.setFixedSize(600, 400)

        self.camera_models = camera_models
        self.workdir = workdir

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Camera Model Overrides"))

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Key", "Parameter", "Value"])
        layout.addWidget(self.table)

        save_button = QPushButton("Save Changes")
        save_button.clicked.connect(self.save_changes)
        layout.addWidget(save_button)

        self.setLayout(layout)

        self.load_camera_models()

    def load_camera_models(self):
        """Load camera models into the table"""
        self.table.setRowCount(0)
        row = 0
        for key, params in self.camera_models.items():
            # key = 'Perspective' etc. (e.g., "Spherical")
            # params = { 'projection_type': 'perspective', 'width': ..., etc. }
            for param, value in params.items():
                self.table.insertRow(row)
                # Column 1 (Key)
                self.table.setItem(row, 0, QTableWidgetItem(key))

                # Column 2 (Parameter)
                self.table.setItem(row, 1, QTableWidgetItem(param))

                # Column 3 (Value) - Use ComboBox for projection_type
                if param == "projection_type":
                    combo = QComboBox()
                    combo.addItems(["perspective", "spherical"])
                    # Set selection based on current value
                    if str(value) in ["perspective", "spherical"]:
                        combo.setCurrentText(str(value))
                    else:
                        # If the value isn't registered, add it to the front and select it
                        combo.insertItem(0, str(value))
                        combo.setCurrentIndex(0)
                    self.table.setCellWidget(row, 2, combo)
                else:
                    # For other params, show as text
                    self.table.setItem(row, 2, QTableWidgetItem(str(value)))

                row += 1

    def save_changes(self):
        """Save changes to camera_models_overrides.json"""
        updated_models = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            param_item = self.table.item(row, 1)

            if not key_item or not param_item:
                continue

            key = key_item.text()
            param = param_item.text()

            # Check if Value is a ComboBox
            cell_widget = self.table.cellWidget(row, 2)
            if isinstance(cell_widget, QComboBox):
                # If ComboBox, get current text
                value = cell_widget.currentText()
            else:
                # If QTableWidgetItem, get text
                value_item = self.table.item(row, 2)
                if value_item:
                    value = value_item.text()
                else:
                    value = ""

            # Convert to float/int if it's a number
            # projection_type is usually a string, so it normally won't be converted
            try:
                # If '.' is included, convert to float, otherwise int
                if '.' in value:
                    num = float(value)
                    value = num
                else:
                    num = int(value)
                    value = num
            except ValueError:
                # If conversion fails, keep as string
                pass

            if key not in updated_models:
                updated_models[key] = {}
            updated_models[key][param] = value

        # Write JSON
        try:
            overrides_path = os.path.join(self.workdir, "camera_models_overrides.json")
            with open(overrides_path, "w") as f:
                json.dump(updated_models, f, indent=4)
                
            # After saving overrides, update the camera_models.json file as well
            # This fixes the issue where camera models aren't updated
            self.update_base_camera_models(updated_models)

            QMessageBox.information(self, "Success", "Camera models saved successfully!")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save camera models: {e}")
            
    def update_base_camera_models(self, overrides):
        """Update the camera_models.json file with the latest overrides
        This fixes the issue where camera models aren't updated when overrides is generated first time"""
        try:
            camera_models_path = os.path.join(self.workdir, "camera_models.json")
            
            # Load existing camera models if available
            if os.path.exists(camera_models_path):
                with open(camera_models_path, "r") as f:
                    base_models = json.load(f)
            else:
                base_models = {}
                
            # Merge with overrides
            merged_models = base_models.copy()
            for key, params in overrides.items():
                if key in merged_models:
                    merged_models[key].update(params)
                else:
                    merged_models[key] = params
                    
            # Save updated camera models
            with open(camera_models_path, "w") as f:
                json.dump(merged_models, f, indent=4)
                
            return True
        except Exception as e:
            print(f"Error updating camera_models.json: {e}")
            return False


class CameraModelManager:
    """Manager class for camera models"""
    def __init__(self, workdir):
        self.workdir = workdir
        self.camera_models = {}
        self._default_model = {
            "Perspective": {
                "projection_type": "perspective",
                "width": 1920,
                "height": 1080,
                "focal_ratio": 1.0
            }
        }
        self.load_camera_models()
    
    def load_camera_models(self):
        """Load camera_models.json and apply camera_models_overrides.json"""
        camera_models_path = os.path.join(self.workdir, "camera_models.json")
        overrides_path = os.path.join(self.workdir, "camera_models_overrides.json")

        # Load base models, if missing use default
        if os.path.exists(camera_models_path):
            try:
                with open(camera_models_path, "r") as f:
                    base_models = json.load(f)
            except Exception as e:
                print(f"Error loading camera_models.json: {e}")
                base_models = self._default_model.copy()
                # Try to create the file with default model
                try:
                    with open(camera_models_path, "w") as f:
                        json.dump(base_models, f, indent=4)
                except Exception as e:
                    print(f"Error creating default camera_models.json: {e}")
        else:
            base_models = self._default_model.copy()
            # Try to create the file with default model
            try:
                with open(camera_models_path, "w") as f:
                    json.dump(base_models, f, indent=4)
            except Exception as e:
                print(f"Error creating default camera_models.json: {e}")

        # Load overrides if they exist
        overrides = {}
        if os.path.exists(overrides_path):
            try:
                with open(overrides_path, "r") as f:
                    overrides = json.load(f)
            except Exception as e:
                print(f"Error loading camera_models_overrides.json: {e}")

        # Merge models with overrides
        merged_models = base_models.copy()
        for key, params in overrides.items():
            if key in merged_models:
                merged_models[key].update(params)
            else:
                merged_models[key] = params

        # Write the merged models back to camera_models.json to ensure it's always up-to-date
        # This fixes the issue where camera models aren't updated when overrides changes
        try:
            with open(camera_models_path, "w") as f:
                json.dump(merged_models, f, indent=4)
        except Exception as e:
            print(f"Error updating camera_models.json: {e}")

        self.camera_models = merged_models
        return merged_models
    
    def get_camera_models(self):
        """Get the current camera models"""
        return self.camera_models
    
    def open_camera_model_editor(self, parent=None):
        """Open camera model editor dialog"""
        if self.workdir:
            # Reload models to get latest changes
            self.load_camera_models()
            
            try:
                dialog = CameraModelEditor(self.camera_models, self.workdir, parent=parent)
                if dialog.exec_():
                    # Reload after saving
                    self.load_camera_models()
                return True
            except Exception as e:
                if parent:
                    QMessageBox.critical(parent, "Error", f"Failed to open camera model editor: {e}")
                return False
        else:
            if parent:
                QMessageBox.warning(parent, "Error", "Workdir is not set.")
            return False