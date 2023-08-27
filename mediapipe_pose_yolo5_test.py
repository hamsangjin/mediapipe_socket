import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 
import mediapipe as mp
import torch 

# Torch Hub에서 'ultralytics/yolov5' 저장소로부터 YOLOv5 모델을 불러오기
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# YOLOv5 모델에 클래스 설정
# 여기서는 인덱스 0인 하나의 클래스만 사용
# 사람만 탐지한다는 뜻으로 받아들이면 됨
yolo_model.classes = [0]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

# 웹캠으로부터 프레임을 처리하는 루프를 시작
while cap.isOpened():
    # 프레임을 읽기
    success, image = cap.read()
    
    # 프레임을 못 읽은 경우 기다리기 (캠 로딩 문제)
    if not success:
        continue
    
    # 이미지를 저장할 빈 리스트 초기화
    img_list = []
    
    # 여백 값 설정
    MARGIN = 10
    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # YOLOv5 모델 사용해 객체 탐지 수행
    result = yolo_model(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 감지된 객체들에 대해서 반복
    for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
        # 자세 추정을 위한 mediapipe holistic 객체를 초기화
        with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
            # 감지된 객체 이미지 부분 처리
            results = holistic.process(image[int(ymin)+MARGIN:int(ymax)+MARGIN, int(xmin)+MARGIN:int(xmax)+MARGIN:])
            
            # 랜드마크를 그림
            mp_drawing.draw_landmarks(
                image[int(ymin)+MARGIN:int(ymax)+MARGIN, int(xmin)+MARGIN:int(xmax)+MARGIN:],
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
            # img_list에 이미지 부분 추가
            img_list.append(image[int(ymin):int(ymax), int(xmin):int(xmax):])
    
    # 랜드마크 화면 출력
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    
    # 'Esc' 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 웹캠을 해제
cap.release()