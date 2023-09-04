import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 
import mediapipe as mp
import torch
from socket import *
import re

# -----------------------------------------------------
# 랜드마크 추출 함수
# 정규 표현식(x, y, z) 추출
# landmarks에서 x, y, z만 추출
# ([+-]?\d+\.\d+) 부분이 각각의 x, y, z 값을 추출하는데 사용
# - 또는 + 부호의 유무와 상관없이 소수점이 포함된 숫자를 추출
pattern = r"landmark {\s+x: ([+-]?\d+\.\d+)\s+y: ([+-]?\d+\.\d+)\s+z: ([+-]?\d+\.\d+)"

def transformer(data, part):

  # 랜드마크 전처리
  matches_list = []

  # 전처리 전의 데이터를 matches_list에 하나씩 저장
  for i in range(len(data)):
    matches_list.append(re.findall(pattern, data[i]))

  # face 제외하고 일단 사전형 생성
  output_data = {part[0]: [], part[1]: [], part[2]: []}

  # left - right - pose 순
  for i, matches in enumerate(matches_list):
    for _, match in enumerate(matches):
        x, y, z = map(float, match)
        output_data[part[i]].append({"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})

  output_json = json.dumps(output_data, indent=2, ensure_ascii=False)

  return output_json

# -----------------------------------------------------

# 소켓 사전 연결
port = 25001

# AF는 주소 체계로, 거의 AF_INET 사용 AF_INET은 IPv4, AF_INET6은 IPv6를 의미
serverSock = socket(AF_INET, SOCK_STREAM)

# 재사용 대기시간 없애기
serverSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

# ''는 AF_INET에서 모든 인터페이스와 연결한다는 의미
serverSock.bind(('', port))

# 인자는 동시접속 허용 개수
serverSock.listen(1)

# client가 접속 요청할 때 결과값이 return됨 / accept() 실행시 새로운 소켓이 생성됨
# 기존에 생성한 serverSock이 아닌 새로운 connectionSock를 통해서 데이터 주고받음
connectionSock, addr = serverSock.accept()

flag = False

# --------------------------------------

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

            # --------------------------------------------
            # 왼손, 오른손, 몸, 얼굴 좌표 추출해 str로 변환 후 저장
            data_left = str(results.left_hand_landmarks)
            data_right = str(results.right_hand_landmarks)
            data_pose = str(results.pose_world_landmarks)
            data_face = str(results.face_landmarks)

            # face 포함해서 전송
            # part = ["face", "left", "right", "pose"]
            # part_data = [data_face, data_left, data_right, data_pose]

            # face 제외하고 전송
            part = ["left", "right", "pose"]
            part_data = [data_left, data_right, data_pose]
            
            # s키로 landmark 전송 제어
            if cv2.waitKey(5) & 0xFF == ord('s'):
                flag = not flag
            
            # flag가 True면 Landmarks 전송
            if flag:
                # 소켓 통신을 이용한 좌표 전송
                landmarks = transformer(part_data, part)

                # 전체 랜드마크 전송
                connectionSock.send(landmarks.encode('utf-8'))

            # --------------------------------------------

    # 랜드마크 화면 출력
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

    # 'Esc' 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 웹캠을 해제
cap.release()