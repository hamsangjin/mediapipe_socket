import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
from socket import *
import re
import json

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

  output_json = json.dumps(output_data, indent=2, ensure_ascii=False)

  return output_json

# -----------------------------------------------------

port = 25001
serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
serverSock.bind(('', port))
serverSock.listen(1)
connectionSock, addr = serverSock.accept()

# --------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.1,
    min_tracking_confidence=0.9,
    model_complexity=0,
    smooth_segmentation=True,
    enable_segmentation=True,
    refine_face_landmarks=True) as holistic:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("카메라 인식 불가")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = holistic.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())

    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())    
    
    mp_drawing.draw_landmarks(
      image,
      results.left_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles
      .get_default_hand_landmarks_style())  

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

    # ------------------------------------

    data_left = str(results.left_hand_landmarks)
    data_right = str(results.right_hand_landmarks)
    data_pose = str(results.pose_world_landmarks)
    data_face = str(results.face_landmarks)
    
    # part_data = [data_face, data_left, data_right, data_pose]
    part_data = [data_left, data_right, data_pose]
    landmarks = transformer(part_data, ["left", "right", "pose"])
    connectionSock.send(landmarks.encode('utf-8'))

    if cv2.waitKey(5) & 0xFF == 27:
      serverSock.close()
      break

cap.release()