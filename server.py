import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
from socket import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 웹캠으로 입력
cap = cv2.VideoCapture(0)

# --------------------------------------

# 소켓 사전 연결
port = 25001

# AF는 주소 체계로, 거의 AF_INET 사용 AF_INET은 IPv4, AF_INET6은 IPv6를 의미
serverSock = socket(AF_INET, SOCK_STREAM)

# ''는 AF_INET에서 모든 인터페이스와 연결한다는 의미
serverSock.bind(('', port))

# 인자는 동시접속 허용 개수
serverSock.listen(1)

# client가 접속 요청할 때 결과값이 return됨 / accept() 실행시 새로운 소켓이 생성됨
# 기존에 생성한 serverSock이 아닌 새로운 connectionSock를 통해서 데이터 주고받음
connectionSock, addr = serverSock.accept()


# --------------------------------------


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
    
    # ------------------------------------
    # 랜드마크 추출
    
    # 왼손, 오른손, 몸, 얼굴 좌표 추출해 str로 변환
    data_left = str(results.left_hand_landmarks)
    data_right = str(results.right_hand_landmarks)
    data_pose = str(results.pose_world_landmarks)
    data_face = str(results.face_landmarks)
    data = 'Face Landmarks\n' + data_face + '\n\n\n\n\n\n\n' + 'Left Hands Landmarks\n' + data_left + '\n\n\n\n\n\n\n' + 'Right Hands Landmarks\n' + data_right + '\n\n\n\n\n\n\n' + 'Pose Landmarks\n' + data_pose

    # txt파일로 출력
    f = open("all_landmarks.txt", 'w')
    f.write(data)
    f.close()

    f = open("pose_world_landmarks.txt", 'w')
    f.write(data_pose)
    f.close()

    # f = open("face_landmarks.txt", 'w')
    # f.write(data_face)
    # f.close()

    # f = open("right_hand_landmarks.txt", 'w')
    # f.write(data_right)
    # f.close()

    # f = open("left_hand_landmarks.txt", 'w')
    # f.write(data_left)
    # f.close()

    # # 랜드마크 터미널 출력
    # print(data_pose)
    # print(data_face)
    # print(data_right)
    # print(data_left)

    # --------------------------------------
    # 소켓 통신을 이용한 좌표 전송

    connectionSock.send(data.encode('utf-8'))
    
    # --------------------------------------

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
    
    # 좌우반전
    # esc키를 누르면 종료되며 소켓 닫음
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      serverSock.close()
      connectionSock.close()
      break

cap.release()