import sys
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QListWidget, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer


class MultiCamMonitoringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.model = torch.hub.load('yolov5', 'yolov5s', source='local')
        self.model.conf = 0.5

        self.caps = [
            cv2.VideoCapture("dataset/cam1.avi"),
            cv2.VideoCapture("dataset/cam2.avi"),
            cv2.VideoCapture("dataset/cam3.avi"),
        ]

        self.detections = {}
        self.selected_objects = set()
        self.person_detected = False
        self.blinking = False

        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_background)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

    def initUI(self):
        self.cam_labels = [QLabel(self) for _ in range(3)]
        for label in self.cam_labels:
            label.setFixedSize(320, 240)
            label.setStyleSheet("background-color: black;")

        self.detect_list = QListWidget(self)
        self.detect_list.itemClicked.connect(self.toggle_object_selection)

        self.cam_button = QPushButton("CAM", self)

        layout = QGridLayout()
        for i, label in enumerate(self.cam_labels):
            layout.addWidget(label, i // 2, i % 2)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.cam_button)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.detect_list)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("다중캠 모니터링 시스템")
        self.resize(1000, 600)

    def update_frames(self):
        person_detected = False

        for idx, cap in enumerate(self.caps):
            ret, frame = cap.read()
            if ret:
                results = self.model(frame)
                detections = results.pandas().xyxy[0]

                self.detections = {}
                self.detect_list.clear()

                for _, row in detections.iterrows():
                    x1, y1, x2, y2, conf, cls, label = row
                    label = label.lower()
                    if label == "person": # 변경&추가해야함
                        person_detected = True
                    if label not in self.detections:
                        self.detections[label] = []
                    self.detections[label].append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

                    item = QListWidgetItem(label)
                    if label in self.selected_objects:
                        item.setBackground(QColor("lightblue"))
                    self.detect_list.addItem(item)

                for selected_object in self.selected_objects:
                    if selected_object in self.detections:
                        for box in self.detections[selected_object]:
                            x, y, w, h = box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 녹색 바운딩 박스

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                qimg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                self.cam_labels[idx].setPixmap(QPixmap.fromImage(qimg))
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if person_detected and not self.blink_timer.isActive():
            self.blink_timer.start(500)
        elif not person_detected and self.blink_timer.isActive():
            self.blink_timer.stop()
            self.setStyleSheet("")

    def toggle_background(self):
        if self.blinking:
            self.setStyleSheet("background-color: #FFBCBC;")
        else:
            self.setStyleSheet("")
        self.blinking = not self.blinking

    def toggle_object_selection(self, item):
        object_name = item.text().lower()
        if object_name in self.selected_objects:
            self.selected_objects.remove(object_name)
            item.setBackground(QColor("white"))
        else:
            self.selected_objects.add(object_name)
            item.setBackground(QColor("lightblue"))
        print(f"Selected objects: {self.selected_objects}")

    def closeEvent(self, event):
        for cap in self.caps:
            cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiCamMonitoringApp()
    window.show()
    sys.exit(app.exec_())
