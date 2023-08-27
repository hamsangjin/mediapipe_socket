import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
from socket import *
import re
import json

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

# --------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 웹캠으로 입력
cap = cv2.VideoCapture(0)

# 모델 초기화
with mp_holistic.Holistic(
    min_detection_confidence=0.1,
    min_tracking_confidence=0.9,

    # 모델 복잡성으로 0 ~ 2, 정확도와 소요시간은 복잡성이 늘어날수록 증가
    model_complexity=0,

    smooth_segmentation=True,

    # 얼굴/손 외에도 분할 마스크를 생성하는 유무
    enable_segmentation=True,

    # 눈과 입술 주변의 랜드마크 좌표를 더 세분화할 것인지
    # github엔 fine_face_landmarks라고 되어있음
    refine_face_landmarks=True) as holistic:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("카메라 인식 불가")
      # 카메라가 로딩중일 수도 있으므로 break가 아닌 continue로
      continue

    # 성능을 향상시키기 위해, 선택적으로 참조로 전달하기 위해 이미지를 쓸 수 없는 것으로 표시하세요.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # 영상 출력
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 눈코입 엣지
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    
    # 얼굴 엣지
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
    
    # 오른손 포인트 추가
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())    
    
    # 왼손 포인트 추가
    mp_drawing.draw_landmarks(
      image,
      results.left_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles
      .get_default_hand_landmarks_style())  

    # 포즈(손 닭발 포함)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    
    # ------------------------------------
    # 랜드마크 추출 및 전송
    
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

    # 소켓 통신을 이용한 좌표 전송
    landmarks = transformer(part_data, part)

    # 전체 랜드마크 전송
    connectionSock.send(landmarks.encode('utf-8'))
    
    # txt파일로 출력
    # f = open("pose_landmarks.txt", 'w')
    # f.write(pose)
    # f.close()

    # --------------------------------------
    # 좌우반전
    # esc키를 누르면 종료되며 소켓 닫음
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      serverSock.close()
      connectionSock.close()
      break

cap.release()