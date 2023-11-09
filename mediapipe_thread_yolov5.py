import cv2
import mediapipe as mp
from mediapipe.python.solutions import holistic as mp_holistic
import torch
import re
import json
from socket import *
import time
import threading

pattern = r"landmark {\s+x: ([+-]?\d+\.\d+)\s+y: ([+-]?\d+\.\d+)\s+z: ([+-]?\d+\.\d+)"

# Landmarks -> Json
def transformer(data, part):
    landmarks = []
    for i in range(len(data)):
        matches_list = []
        for j in range(len(data[i])):
            matches_list.append(re.findall(pattern, data[i][j]))

        output_data = {part[0]: [], part[1]: [], part[2]: []}

        for j, matches in enumerate(matches_list):
            for _, match in enumerate(matches):
                x, y, z = map(float, match)
                output_data[part[j]].append({"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})

        landmarks.append(output_data)
    
    landmarks = {"data": landmarks}
    landmarks = json.dumps(landmarks, indent=2, ensure_ascii=False)

    return landmarks

global MARGIN
MARGIN = 1

# Holistic Model Processing
def holiticProcess(image, resultList, holistic, part_data, i):
    xmin, ymin, xmax, ymax, _, _ = map(int, resultList)

    x1, y1 = xmin, ymin
    x2, y2 = xmax, ymax

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    results = holistic.process(image[ymin+MARGIN:ymax+MARGIN,xmin+MARGIN:xmax+MARGIN:])
    
    mp_drawing.draw_landmarks(
        image[ymin+MARGIN:ymax+MARGIN,xmin+MARGIN:xmax+MARGIN:], 
        results.pose_landmarks, 
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

    mp_drawing.draw_landmarks(
        image[ymin+MARGIN:ymax+MARGIN,xmin+MARGIN:xmax+MARGIN:],
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style()
    )
    
    mp_drawing.draw_landmarks(
        image[ymin+MARGIN:ymax+MARGIN,xmin+MARGIN:xmax+MARGIN:],
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style()
    )    
    
    mp_drawing.draw_landmarks(
      image[ymin+MARGIN:ymax+MARGIN,xmin+MARGIN:xmax+MARGIN:],
      results.left_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles
      .get_default_hand_landmarks_style()
    )  
    
    data_left = str(results.left_hand_landmarks)
    data_right = str(results.right_hand_landmarks)
    data_pose = str(results.pose_world_landmarks)

    part_data[i] = [data_left, data_right, data_pose]


if __name__ == '__main__':
    # Socket Connect
    port = 25001
    serverSock = socket(AF_INET, SOCK_STREAM)
    serverSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    serverSock.bind(('', port))
    serverSock.listen(1)
    connectionSock, addr = serverSock.accept()

    # YOLO Model Setting
    yolo_model = torch.hub.load('/Users/sangjin/Desktop/yolov5', 'custom',  'yolov5x6.pt', source='local')
    yolo_model.conf = 0.5
    yolo_model.dynamic = True
    yolo_model.pretrained = True
    yolo_model.classes=[0]
    yolo_model.to(torch.device('cuda'))

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)

    # 웹캠 해상도 설정
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frameCnt, fps, startTime, fps_avg, start = 0, 0, 0, [], True

    holistic = []
    for i in range(2):
        holistic.append(mp_holistic.Holistic(
        min_detection_confidence=0.1,
        min_tracking_confidence=0.9,
        model_complexity=1,
        smooth_segmentation=False,
        enable_segmentation=True,
        refine_face_landmarks=True))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False    
        
        # Start Setting
        if start:
            start = False
            startTime = time.time()
            fps_avg_start = time.time()
            result = yolo_model(image)

        # Yolo Operates Every 3 Prames
        if (frameCnt == 3):
            result = yolo_model(image)
            fps = 3 / (time.time() - startTime)
            fps_avg.append(fps)
            print(f"Frames per second: {fps:.2f}")
            frameCnt = 0
            startTime = time.time()

        frameCnt += 1

        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        resultList = sorted(result.xyxy[0].tolist())

        part_data, threads = [], []
        for _ in range(len(resultList)):
            part_data.append([])

        # MediaPipe Holistic Model Multi Thread Proccesing
        for i in range(len(resultList)):
            th = threading.Thread(target=holiticProcess, args=(image, resultList[i], holistic[i], part_data, i, ))
            th.start()
            threads.append(th)

        for th in threads:
            th.join()

        # Landmarks -> Json
        # Socket Send
        landmarks = transformer(part_data, ["left", "right", "pose"])
        connectionSock.send(landmarks.encode('utf-8'))
        
        # FPS Print
        fps_str = "FPS : %0.1f" %fps
        image = cv2.flip(image, 1)
        cv2.putText(image, fps_str, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

        cv2.imshow('MediaPipe Holistic', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    for h in holistic:
        h.close

    print('실행 시간 ' + str(round(time.time() - fps_avg_start, 2)), "초동안 평균은 " + str(round(sum(fps_avg) / len(fps_avg), 2)) + 'fps입니다.')

    cap.release()
    cv2.destroyAllWindows()