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
    def __init__(self):
        super().__init__()
        self.mode = "fastapi"
        self.initUI()

        self.stream_url = "http://127.0.0.1:20001/"  # JPG 스트림 URL
        self.json_url = "http://127.0.0.1:20001/json"  # JSON 데이터 URL

        self.detections = {}  # JSON 데이터를 기반으로 객체 정보 저장
        self.selected_object = None  # 선택된 객체

        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

    def initUI(self):
        """UI 초기화"""
        # 좌측 CAM 버튼
        self.cam_button = QPushButton("CAM", self)

        # 중앙 카메라 화면 (캠별 QLabel 생성)
        self.cam_labels = QLabel(self)
        self.cam_labels.setFixedSize(640, 480)  # 영상 크기
        self.cam_labels.setStyleSheet("background-color: black;")

        # 우측 감지된 객체 리스트
        self.detect_list = QListWidget(self)
        self.detect_list.itemClicked.connect(self.toggle_object_selection)

        # 전체 레이아웃 구성
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.cam_button)

        center_layout = QVBoxLayout()
        center_layout.addWidget(self.cam_labels)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.detect_list)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(center_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("다중캠 모니터링 시스템")
        self.resize(1000, 600)

    def fetch_stream(self):
        """서버에서 JPG 스트림 프레임 받아오기"""
        try:
            # 스트림 형식의 HTTP 요청을 보냄 (JPG 데이터를 실시간으로 받음)
            response = requests.get(self.stream_url, stream=True, timeout=5)
            byte_data = b""  # 받은 데이터를 저장할 바이트 배열 초기화

            # 스트림에서 데이터를 청크 단위로 읽음
            for chunk in response.iter_content(chunk_size=1024):
                byte_data += chunk  # 청크 데이터를 누적
                # JPG 이미지의 시작(b'\xff\xd8')과 끝(b'\xff\xd9')을 찾음
                start = byte_data.find(b"\xff\xd8")  # JPEG 시작 바이트
                end = byte_data.find(b"\xff\xd9") + 2  # JPEG 끝 바이트 (+2는 마지막 바이트 포함)
                
                if start != -1 and end != -1:  # 이미지의 시작과 끝을 모두 찾았을 경우
                    jpg = byte_data[start:end]  # JPG 데이터 추출
                    byte_data = byte_data[end:]  # 추출한 데이터 이후 남은 데이터 저장
                    
                    # JPG 데이터를 디코딩하여 OpenCV 이미지 형식으로 변환
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    return img  # 디코딩된 이미지를 반환

        except Exception as e:
            # 요청이나 디코딩 중 에러가 발생한 경우
            print(f"Error fetching stream: {e}")
            return None  # 에러 발생 시 None 반환

    def fetch_json(self):
        """서버에서 JSON 데이터 받아오기"""
        try:
            response = requests.get(self.json_url, timeout=5)
            if response.status_code == 200:
                return response.json()  # JSON 데이터를 반환
        except Exception as e:
            print(f"Error fetching JSON: {e}")
        return []

    def update_frames(self):
        """프레임 및 JSON 데이터 업데이트"""
        # JPG 스트림 프레임 가져오기
        frame = self.fetch_stream()
        if frame is not None:
            # JSON 데이터 가져오기
            json_data = self.fetch_json()
            self.update_detections(json_data)

            # 바운딩 박스 그리기
            for detection in self.detections.get("objects", []):
                bbox = detection["bbox"]
                label = f'ID: {detection["global_id"]}'
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if label == self.selected_object else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # QLabel에 업데이트
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qimg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.cam_labels.setPixmap(QPixmap.fromImage(qimg))

    def update_detections(self, json_data):
        """JSON 데이터를 기반으로 객체 리스트 업데이트"""
        self.detections = {"objects": json_data}  # 객체 정보 저장
        current_items = [self.detect_list.item(i).text() for i in range(self.detect_list.count())]

        for detection in json_data:
            label = f'ID: {detection["global_id"]}'
            if label not in current_items:
                item = QListWidgetItem(label)
                self.detect_list.addItem(item)

    def toggle_object_selection(self, item):
        """리스트에서 객체 선택 시 강조"""
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
        """GUI 종료 시 타이머 정지"""
        self.timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiCamMonitoringApp()
    window.show()
    sys.exit(app.exec_())
