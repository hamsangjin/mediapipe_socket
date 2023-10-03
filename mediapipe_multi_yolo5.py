import cv2
import mediapipe as mp
from mediapipe.python.solutions import holistic as mp_holistic
import torch
import re
import json
from socket import *

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

# -----------------------------------------------------

port = 25001
serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
serverSock.bind(('', port))
serverSock.listen(1)
connectionSock, addr = serverSock.accept()

# --------------------------------------

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# yolo_model.to(torch.device('cuda'))

yolo_model.classes=[0]

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

holistic = []
for i in range(5):
  holistic.append(mp_holistic.Holistic(
    min_detection_confidence=0.1,
    min_tracking_confidence=0.9,
    model_complexity=0,
    smooth_segmentation=False,
    enable_segmentation=True,
    refine_face_landmarks=True))

while cap.isOpened():    
    success, image = cap.read()
    if not success:
      continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False    

    result = yolo_model(image)    
    
    image.flags.writeable = True   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    MARGIN=10
    part_data = []

    resultlist = result.xyxy[0].tolist()

    for i in range(len(resultlist)):
        xmin, ymin, xmax, ymax, confidence, clas = resultlist[i]

        results = holistic[i].process(image[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:])
        
        mp_drawing.draw_landmarks(image[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:], results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                              )
        
        data_left = str(results.left_hand_landmarks)
        data_right = str(results.right_hand_landmarks)
        data_pose = str(results.pose_world_landmarks)
        data_face = str(results.face_landmarks)

        temp = [data_left, data_right, data_pose]

        part_data.append(temp)
        
    # --------------------------------------

    landmarks = []

    for i in range(len(part_data)):
        landmarks.append(transformer(part_data[i], ["left", "right", "pose"]))

    landmarks = {"data": landmarks}
    landmarks = json.dumps(landmarks, indent=2, ensure_ascii=False)
    connectionSock.send(landmarks.encode('utf-8'))

    # --------------------------------------

    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break

for h in holistic:
   h.close

cap.release()
cv2.destroyAllWindows()