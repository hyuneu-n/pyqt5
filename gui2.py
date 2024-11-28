import sys
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QListWidget, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer


class MultiCamMonitoringApp(QWidget):
    def __init__(self, mode="dataset"):
        """
        :param mode: 실행 모드 ("fastapi" 또는 "dataset")
        """
        super().__init__()
        self.mode = mode
        self.initUI()

        # YOLO 모델 로드
        self.model = torch.hub.load('yolov5', 'yolov5s', source='local')
        self.model.conf = 0.5  # 감지 임계값 설정

        # FastAPI 스트림 URL
        self.stream_url = "http://0.0.0.0:20001/"

        if self.mode == "dataset":
            # 데이터셋 비디오 경로 설정
            self.video_paths = [
                "dataset/cam1.avi",
                "dataset/cam2.avi",
                "dataset/cam3.avi",
            ]
            self.caps = [cv2.VideoCapture(path) for path in self.video_paths]

        # 감지된 객체 및 선택된 객체 관리
        self.detections = {}
        self.selected_object = None

        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

    def initUI(self):
        """
        UI 초기화
        """
        # 좌측 CAM 버튼
        self.cam_button = QPushButton("CAM", self)

        # 중앙 카메라 QLabel
        self.cam_labels = [QLabel(self) for _ in range(3)]
        for label in self.cam_labels:
            label.setFixedSize(320, 240)  # QLabel 크기 설정
            label.setStyleSheet("background-color: black;")

        # 우측 감지된 객체 리스트
        self.detect_list = QListWidget(self)
        self.detect_list.itemClicked.connect(self.toggle_object_selection)

        # 전체 레이아웃 구성
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
        """
        타이머를 통해 주기적으로 호출되는 메서드
        데이터셋의 각 비디오 또는 FastAPI 스트림에서 프레임을 읽어 QLabel에 표시
        """
        if self.mode == "dataset":
            for idx, cap in enumerate(self.caps):
                ret, frame = cap.read()
                if ret:
                    # YOLO 감지 수행
                    results = self.model(frame)
                    detections = results.pandas().xyxy[0]

                    # 바운딩 박스와 객체 이름 저장
                    self.detections = {}
                    for _, row in detections.iterrows():
                        x1, y1, x2, y2, conf, cls, label = row
                        label = label.lower()
                        if label not in self.detections:
                            self.detections[label] = []
                        self.detections[label].append((int(x1), int(y1), int(x2), int(y2)))

                        # 바운딩 박스 그리기 (선택된 객체만 표시)
                        if self.selected_object == label:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # QLabel 업데이트
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = frame.shape
                    step = channel * width
                    qimg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                    self.cam_labels[idx].setPixmap(QPixmap.fromImage(qimg))

            # 감지된 객체 리스트 업데이트
            self.update_detect_list()

    def update_detect_list(self):
        """
        감지된 객체를 QListWidget에 업데이트
        """
        current_items = [self.detect_list.item(i).text() for i in range(self.detect_list.count())]
        for label in self.detections.keys():
            if label not in current_items:
                item = QListWidgetItem(label)
                if label == self.selected_object:
                    item.setBackground(QColor("lightblue"))  # 선택된 객체 강조 표시
                self.detect_list.addItem(item)

    def toggle_object_selection(self, item):
        """
        QListWidget에서 객체 선택 시 호출
        선택된 객체를 저장하고 바운딩 박스를 표시
        """
        object_name = item.text()
        if self.selected_object == object_name:
            # 이미 선택된 객체를 클릭한 경우 선택 해제
            self.selected_object = None
            item.setBackground(QColor("white"))
        else:
            # 새로운 객체를 선택한 경우
            self.selected_object = object_name
            for i in range(self.detect_list.count()):
                list_item = self.detect_list.item(i)
                if list_item.text() == object_name:
                    list_item.setBackground(QColor("lightblue"))  # 선택된 객체 강조
                else:
                    list_item.setBackground(QColor("white"))
        print(f"Selected object: {self.selected_object}")

    def closeEvent(self, event):
        """
        GUI 종료 시 타이머 정지 및 리소스 해제
        """
        self.timer.stop()
        if self.mode == "dataset":
            for cap in self.caps:
                cap.release()


if __name__ == "__main__":
    mode = "dataset"  # "fastapi" 또는 "dataset" 실행 모드 설정

    app = QApplication(sys.argv)
    window = MultiCamMonitoringApp(mode=mode)
    window.show()
    sys.exit(app.exec_())
