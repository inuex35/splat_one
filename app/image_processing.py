# image_processing.py

import os
import json
import shutil
import datetime
import fractions
from PIL import Image
import piexif
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QComboBox, QDialogButtonBox, QCheckBox
from PyQt5.QtCore import Qt

class ResolutionDialog(QDialog):
    """Dialog for resizing images"""
    def __init__(self, current_width, current_height, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resize Images")
        self.setFixedSize(300, 150)

        self.original_width = current_width
        self.original_height = current_height
        self.aspect_ratio = current_width / current_height

        layout = QVBoxLayout()

        self.resize_method_combo = QComboBox()
        self.resize_method_combo.addItems(["Percentage (%)", "Width (px)", "Height (px)"])
        self.resize_method_combo.currentIndexChanged.connect(self.update_label)
        layout.addWidget(QLabel("Resize method:"))
        layout.addWidget(self.resize_method_combo)

        form_layout = QFormLayout()
        self.value_input = QLineEdit("100")
        form_layout.addRow(QLabel("Value:"), self.value_input)

        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(button_box)
        self.setLayout(layout)

    def update_label(self):
        method = self.resize_method_combo.currentText()
        if method == "Percentage (%)":
            self.value_input.setText("100")
        elif method == "Width (px)":
            self.value_input.setText(str(self.original_width))
        else:
            self.value_input.setText(str(self.original_height))

    def get_values(self):
        method = self.resize_method_combo.currentText()
        value = float(self.value_input.text())
        return method, value


class ExifExtractProgressDialog(QDialog):
    """Progress dialog for EXIF extraction"""
    def __init__(self, message="Please Wait...", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extracting EXIF Data")
        self.setModal(True)
        self.setFixedSize(520, 120)

        layout = QVBoxLayout()
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(label)

        self.setLayout(layout)


class ImageProcessor:
    """Class for handling image processing operations"""
    def __init__(self, workdir):
        self.workdir = workdir
        self.images_folder = os.path.join(workdir, "images")
        self.images_org_folder = os.path.join(workdir, "images_org")
        self.exif_folder = os.path.join(workdir, "exif")
    
    def resize_images(self, method, value, progress_callback=None):
        """Resize images in the images folder"""
        # Backup original images if not already backed up
        if not os.path.exists(self.images_org_folder):
            shutil.copytree(self.images_folder, self.images_org_folder)
        
        # Get list of image files
        image_files = [f for f in os.listdir(self.images_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Process each image
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(self.images_folder, image_file)
            
            # Open image
            with Image.open(image_path) as img:
                # Calculate new dimensions based on method
                if method == "Percentage (%)":
                    scale = value / 100
                    new_width = int(img.width * scale)
                    new_height = int(img.height * scale)
                elif method == "Width (px)":
                    new_width = int(value)
                    new_height = int(new_width * img.height / img.width)
                else:  # Height (px)
                    new_height = int(value)
                    new_width = int(new_height * img.width / img.height)
                
                # Resize image
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save resized image
                img_resized.save(image_path)
            
            # Call progress callback if provided
            if progress_callback:
                progress = (i + 1) / len(image_files) * 100
                progress_callback(int(progress))
        
        return len(image_files)
    
    def restore_original_images(self):
        """Restore original images from backup"""
        if os.path.exists(self.images_org_folder):
            shutil.rmtree(self.images_folder)
            shutil.copytree(self.images_org_folder, self.images_folder)
            return True
        return False
    
    def get_sample_image_dimensions(self):
        """Get dimensions of a sample image in the folder"""
        image_files = [f for f in os.listdir(self.images_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            return None, None
        
        sample_image_path = os.path.join(self.images_folder, image_files[0])
        with Image.open(sample_image_path) as img:
            return img.width, img.height
    
    @staticmethod
    def convert_to_degrees(value):
        """Convert decimal GPS coordinates to degrees, minutes, seconds"""
        d = int(value)
        m = int((value - d) * 60)
        s = (value - d - m / 60) * 3600
        return d, m, s

    @staticmethod
    def convert_to_rational(number):
        """Convert a number to a rational for EXIF data"""
        f = fractions.Fraction(str(number)).limit_denominator()
        return f.numerator, f.denominator

    def apply_exif_from_mapillary_json(self, json_path, images_dir):
        """Apply EXIF data from Mapillary JSON to images"""
        with open(json_path, 'r') as file:
            metadata_list = json.load(file)

        processed_count = 0
        for metadata in metadata_list:
            image_filename = metadata.get('filename')
            if not image_filename:
                continue

            image_path = os.path.join(images_dir, os.path.basename(image_filename))
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            try:
                img = Image.open(image_path)
                try:
                    exif_dict = piexif.load(img.info['exif'])
                except (KeyError, piexif.InvalidImageDataError):
                    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

                lat_deg = self.convert_to_degrees(metadata['MAPLatitude'])
                lon_deg = self.convert_to_degrees(metadata['MAPLongitude'])

                gps_ifd = {
                    piexif.GPSIFD.GPSLatitudeRef: 'N' if metadata['MAPLatitude'] >= 0 else 'S',
                    piexif.GPSIFD.GPSLongitudeRef: 'E' if metadata['MAPLongitude'] >= 0 else 'W',
                    piexif.GPSIFD.GPSLatitude: [
                        self.convert_to_rational(lat_deg[0]),
                        self.convert_to_rational(lat_deg[1]),
                        self.convert_to_rational(lat_deg[2])
                    ],
                    piexif.GPSIFD.GPSLongitude: [
                        self.convert_to_rational(lon_deg[0]),
                        self.convert_to_rational(lon_deg[1]),
                        self.convert_to_rational(lon_deg[2])
                    ],
                    piexif.GPSIFD.GPSAltitude: self.convert_to_rational(metadata['MAPAltitude']),
                    piexif.GPSIFD.GPSAltitudeRef: 0 if metadata['MAPAltitude'] >= 0 else 1,
                }

                exif_dict['GPS'] = gps_ifd

                capture_time = datetime.datetime.strptime(metadata['MAPCaptureTime'], '%Y_%m_%d_%H_%M_%S_%f')
                exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = capture_time.strftime('%Y:%m:%d %H:%M:%S')
                exif_dict['0th'][piexif.ImageIFD.Orientation] = metadata.get('MAPOrientation', 1)

                exif_bytes = piexif.dump(exif_dict)
                img.save(image_path, "jpeg", exif=exif_bytes)
                img.close()

                processed_count += 1

            except Exception as e:
                print(f"Failed to update EXIF for {image_path}: {e}")

        # After writing EXIF, rename the folder to 'images'
        images_parent_dir = os.path.dirname(images_dir)
        final_images_dir = os.path.join(images_parent_dir, "images")

        # If 'images' folder already exists, remove it and replace
        if os.path.exists(final_images_dir):
            shutil.rmtree(final_images_dir)
        os.rename(images_dir, final_images_dir)
        
        return processed_count