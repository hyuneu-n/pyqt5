import sys
import cv2
import os
import numpy as np
import requests
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QListWidget, QVBoxLayout,
    QHBoxLayout, QWidget, QGridLayout, QListWidgetItem
)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer


class MultiCamMonitoringApp(QWidget):
    def __init__(self, mode="dataset"):
        super().__init__()
        self.mode = mode
        self.initUI()

        self.stream_url = "http://127.0.0.1:20001/"  # stream.py 서버 URL

        # 로컬 테스트용 dataset 설정
        if self.mode == "dataset":
            self.video_paths = [
                os.path.join(os.getcwd(), "dataset/cam1.avi"),
                os.path.join(os.getcwd(), "dataset/cam2.avi"),
                os.path.join(os.getcwd(), "dataset/cam3.avi"),
            ]
            self.caps = [cv2.VideoCapture(path) for path in self.video_paths]

        self.detections = {}
        self.selected_object = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

    def initUI(self):
        # 좌측 CAM 버튼
        self.cam_button = QPushButton("CAM", self)

        # 중앙 카메라 화면
        self.cam_labels = [QLabel(self) for _ in range(3)]
        for label in self.cam_labels:
            label.setFixedSize(320, 240)  # 각 비디오 크기
            label.setStyleSheet("background-color: black;")

        # 우측 감지된 객체 리스트
        self.detect_list = QListWidget(self)
        self.detect_list.itemClicked.connect(self.toggle_object_selection)

        # 레이아웃 구성
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.cam_button)

        grid_layout = QGridLayout()
        for i, label in enumerate(self.cam_labels):
            grid_layout.addWidget(label, i // 2, i % 2)  # 2x2 배치

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.detect_list)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(grid_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("다중캠 모니터링 시스템")
        self.resize(1000, 600)

    def fetch_stream(self): #프레임 맞추기
        try:
            response = requests.get(self.stream_url, stream=True, timeout=5)
            byte_data = b""
            for chunk in response.iter_content(chunk_size=1024):
                byte_data += chunk
                start = byte_data.find(b"--frame")
                end = byte_data.find(b"--frame", start + 1)
                if start != -1 and end != -1:
                    frame_data = byte_data[start:end]
                    byte_data = byte_data[end:]
                    img_start = frame_data.find(b"\xff\xd8")
                    img_end = frame_data.find(b"\xff\xd9") + 2
                    if img_start != -1 and img_end != -1:
                        jpg = frame_data[img_start:img_end]
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        return img
        except Exception as e:
            print(f"Error fetching stream: {e}")
            return None

    def update_frames(self):
        if self.mode == "fastapi":
            # FastAPI 스트림 모드
            frame = self.fetch_stream()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                qimg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                self.cam_labels[0].setPixmap(QPixmap.fromImage(qimg))

        elif self.mode == "dataset":
            # dataset 모드
            for idx, cap in enumerate(self.caps):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (320, 240))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = frame.shape
                    step = channel * width
                    qimg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                    self.cam_labels[idx].setPixmap(QPixmap.fromImage(qimg))
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_detect_list(self):
        current_items = [self.detect_list.item(i).text() for i in range(self.detect_list.count())]
        for label in self.detections.keys():
            if label not in current_items:
                item = QListWidgetItem(label)
                self.detect_list.addItem(item)

    def toggle_object_selection(self, item):
        object_name = item.text()
        if self.selected_object == object_name:
            self.selected_object = None
            item.setBackground(QColor("white"))
        else:
            self.selected_object = object_name
            for i in range(self.detect_list.count()):
                list_item = self.detect_list.item(i)
                if list_item.text() == object_name:
                    list_item.setBackground(QColor("lightblue"))
                else:
                    list_item.setBackground(QColor("white"))
        print(f"Selected object: {self.selected_object}")

    def closeEvent(self, event):
        self.timer.stop()
        if self.mode == "dataset":
            for cap in self.caps:
                cap.release()

if __name__ == "__main__":
    mode = "dataset"  # dataset 또는 fastapi
    app = QApplication(sys.argv)
    window = MultiCamMonitoringApp(mode=mode)
    window.show()
    sys.exit(app.exec_())