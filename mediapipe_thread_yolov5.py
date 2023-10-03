import cv2
import mediapipe as mp
from mediapipe.python.solutions import holistic as mp_holistic
import torch
import re
import json
from socket import *
import time
import threading

global part_data
global holistic
global MARGIN
MARGIN = 1
part_data = []

holistic = []
for i in range(5):
    holistic.append(mp_holistic.Holistic(
        min_detection_confidence=0.1,
        min_tracking_confidence=0.9,
        model_complexity=0,
        smooth_segmentation=False,
        enable_segmentation=True,
        refine_face_landmarks=True))
pattern = r"landmark {\s+x: ([+-]?\d+\.\d+)\s+y: ([+-]?\d+\.\d+)\s+z: ([+-]?\d+\.\d+)"

def transformer(data, part):
    matches_list = []

    for i in range(len(data)):
        matches_list.append(re.findall(pattern, data[i]))

    output_data = {part[0]: [], part[1]: [], part[2]: []}

    for i, matches in enumerate(matches_list):
        for _, match in enumerate(matches):
            x, y, z = map(float, match)
            output_data[part[i]].append({"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})

    return output_data

def holiticProcess(image, i, resultList):
    xmin, ymin, xmax, ymax, confidence, clas = resultList[i]

    results = holistic[i].process(image[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:])
    
    mp_drawing.draw_landmarks(image[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:], results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                            )
    
    data_left = str(results.left_hand_landmarks)
    data_right = str(results.right_hand_landmarks)
    data_pose = str(results.pose_world_landmarks)
    # data_face = str(results.face_landmarks)

    temp = [data_left, data_right, data_pose]

    part_data.append(temp)


if __name__ == '__main__':
    # -----------------------------------------------------

    port = 25001
    serverSock = socket(AF_INET, SOCK_STREAM)
    serverSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    serverSock.bind(('', port))
    serverSock.listen(1)
    connectionSock, addr = serverSock.accept()

    # --------------------------------------

    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

    yolo_model.conf = 0.5         # 객체 감지의 신뢰도 임계값(0~1)
    yolo_model.dynamic = True     # 입력 이미지 크기를 동적으로 조정하여 최상의 성능을 얻을지 여부
    yolo_model.pretrained = True  # 이미 학습된 가중치를 사용하여 모델을 초기화할지 여부
    yolo_model.classes=[0]
    yolo_model.to(torch.device('cuda'))


    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)

    frameCnt = 0
    start = True
    startTime = 0

    # holistic = []
    # for i in range(5):
    #     holistic.append(mp_holistic.Holistic(
    #         min_detection_confidence=0.1,
    #         min_tracking_confidence=0.9,
    #         model_complexity=0,
    #         smooth_segmentation=False,
    #         enable_segmentation=True,
    #         refine_face_landmarks=True))

    while cap.isOpened():    
        success, image = cap.read()
        if not success:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False    

        if start:
            start = False
            startTime = time.time()
            result = yolo_model(image)

        if (frameCnt == 3):
            result = yolo_model(image)
            frameCnt = 0
        frameCnt += 1

        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        resultList = result.xyxy[0].tolist()
        
        part_data = []
        procs = []
        for i in range(len(resultList)):

            p = threading.Thread(target=holiticProcess, args=(image, i, resultList, ))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()  # 프로세스가 모두 종료될 때까지 대기

        landmarks = []

        for i in range(len(part_data)):
            landmarks.append(transformer(part_data[i], ["left", "right", "pose"]))

        landmarks = {"data": landmarks}
        landmarks = json.dumps(landmarks, indent=2, ensure_ascii=False)
        connectionSock.send(landmarks.encode('utf-8'))

        # --------------------------------------

        fps = 1 / (time.time() - startTime)
        startTime = time.time()
        fps_str = "FPS : %0.1f" %fps

        image = cv2.flip(image, 1)
        cv2.putText(image, fps_str, (0, 75), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 0))
        cv2.imshow('MediaPipe Holistic', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    for h in holistic:
        h.close

    cap.release()
    cv2.destroyAllWindows()