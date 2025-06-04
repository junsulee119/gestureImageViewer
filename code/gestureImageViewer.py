import mediapipe as mp
#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import threading
import math

image_path = '../media/baboon.png'
model_path = '../mediapipeModel/hand_landmarker.task'
video_path = 0

img_original = cv.imread(image_path)

# 결과 이미지 공유용 변수
latest_annotated_frame = None
# 결과 데이터 공유용 변수
latest_fingertip_data = None
# 핀치액션 좌표 공유용 변수
pts_i = None
pts_c = None
frame_lock = threading.Lock()
scaled_img = np.copy(img_original)

def dist3(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def is_pinch(result, pinch_threshold): # threshold값은 경험적으로 기입
    if result is None or not result.hand_landmarks:
        return False, False        # (isPinch, isTwo)

    isPinch = False
    isTwo = len(result.handedness) == 2

    if isTwo:
        hands = {result.handedness[i][0].category_name: result.hand_landmarks[i] for i in range(2)}
        right = hands.get("Right")
        left = hands.get("Left")

        if right and left:
            r_tmb = [right[4].x, right[4].y, right[4].z]
            r_idx = [right[8].x, right[8].y, right[8].z]
            l_tmb = [left[4].x, left[4].y, left[4].z]
            l_idx = [left[8].x, left[8].y, left[8].z]

            r_mid = [(r_tmb[i] + r_idx[i]) / 2 for i in range(3)]
            l_mid = [(l_tmb[i] + l_idx[i]) / 2 for i in range(3)]
            
            distance = dist3(r_mid, l_mid)
            zMid = abs((r_mid[2] + l_mid[2]) / 2)
            distance_n = distance / zMid

            if distance_n < pinch_threshold:
                isPinch = True
    else:
        tmb = [result.hand_landmarks[0][4].x, result.hand_landmarks[0][4].y, result.hand_landmarks[0][4].z]
        idx = [result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y, result.hand_landmarks[0][8].z]
        
        distance = dist3(tmb, idx)
        zMid = abs((tmb[2] + idx[2]) / 2)
        distance_n = distance / zMid

        if distance_n < pinch_threshold:
            isPinch = True

    return isPinch, isTwo

def scale_img_w_diag(img_original, pts_i, pts_c):    # 양손 핀치액션 대각선 길이에 log scale로 비례하여 이미지 확대 / 축소, l_initial: 핀치액션 초기 대각선 길이, l_current: 핀치액션 현재 길이
    # pts_i, pts_c -> l_i, l_c 연산 알고리즘, pts는 2차원 리스트 [[l_mid_x, l_mid_y, l_mid_z], [r_mid_x, r_mid_y, r_mid_z]]
    l_i = dist3(pts_i[0], pts_i[1])
    l_c = dist3(pts_c[0], pts_c[1])
    
    scale_factor = math.log(l_c/l_i, 2) # base값 조절해서 scaling transition의 자연스러움 향상해야 함
    scale_factor = max(scale_factor, 0.01) # zero division error 및 음수 결과 방지
    scaled_img = cv.resize(img_original, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_NEAREST)    # 속도 / 품질 고려해서 보간법 선택
    
    return scaled_img

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in hand_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        x_coords = [l.x for l in hand_landmarks]
        y_coords = [l.y for l in hand_landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - 10

        cv.putText(annotated_image, handedness[0].category_name, (text_x, text_y),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (88, 205, 54), 2, cv.LINE_AA)

    return annotated_image

# 비동기 콜백 함수
def result_callback(result, output_image, timestamp_ms):
    global latest_annotated_frame
    global latest_fingertip_data
    with frame_lock:
        latest_annotated_frame = draw_landmarks_on_image(output_image.numpy_view(), result)
        
        """if result.hand_landmarks:
            latest_fingertip_data = result
            #print(result)
            #print(result.hand_landmarks)
            tmb_x, tmb_y, tmb_z = result.hand_landmarks[0][4].x, result.hand_landmarks[0][4].y, result.hand_landmarks[0][4].z
            idx_x, idx_y, idx_z = result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y, result.hand_landmarks[0][8].z
            mid_x, mid_y, mid_z = result.hand_landmarks[0][11].x, result.hand_landmarks[0][11].y, result.hand_landmarks[0][11].z
            
            #tmb_x, tmb_y, tmb_z = result.hand_world_landmarks[0][4].x, result.hand_world_landmarks[0][4].y, result.hand_world_landmarks[0][4].z
            #idx_x, idx_y, idx_z = result.hand_world_landmarks[0][7].x, result.hand_world_landmarks[0][7].y, result.hand_world_landmarks[0][7].z
            #mid_x, mid_y, mid_z = result.hand_world_landmarks[0][11].x, result.hand_world_landmarks[0][11].y, result.hand_world_landmarks[0][11].z
            
            distance = math.sqrt(
                                (tmb_x - idx_x) ** 2 +
                                (tmb_y - idx_y) ** 2 +
                                (tmb_z - idx_z) ** 2
                                )
            
            distance_n = distance / (abs(tmb_z + idx_z)/2)
            
            print(distance_n)
            
            #print(f"tmb_x: {tmb_x:.3f}, tmb_y: {tmb_y:.3f}, tmb_z: {tmb_z:.3f}")
            #print(f"idx_x: {idx_x:.3f}, idx_y: {idx_y:.3f}, idx_z: {idx_z:.3f}")
            #print(f"mid_x: {mid_x:.3f}, mid_y: {mid_y:.3f}, mid_z: {mid_z:.3f}")
            
            #print(f"tmb_x_n: {tmb_x / tmb_z:.3f}, tmb_y_n: {tmb_y / tmb_z:.3f}")
            #print(f"idx_x_n: {idx_x / idx_z:.3f}, idx_y_n: {idx_y / idx_z:.3f}")
            
            print()
        else:
            print("No hands detected.")
            print()""" # for debugging

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

with HandLandmarker.create_from_options(options) as landmarker:
    video = cv.VideoCapture(video_path)
    fps = video.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while video.isOpened():
        valid, frame = video.read()
        if not valid:
            break

        # BGR -> RGB 변환
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp = int(video.get(cv.CAP_PROP_POS_MSEC))
        landmarker.detect_async(mp_image, timestamp) # mediapipe landmark 추출
        
        # is_pinch
        isPinch, isTwo = is_pinch(latest_fingertip_data, 0.9) # 0.9 - 1.0 사이 적절한듯
        print(f"isPinch: {isPinch}, isTwo: {isTwo}\n")
        
        if latest_fingertip_data is None:
            continue
        
        if isTwo:
            hands = {latest_fingertip_data.handedness[i][0].category_name: latest_fingertip_data.hand_landmarks[i] for i in range(2)}
            right = hands.get("Right")
            left = hands.get("Left")
            
        if len(latest_fingertip_data.handedness) == 2:
            hands = {latest_fingertip_data.handedness[i][0].category_name: latest_fingertip_data.hand_landmarks[i]
            for i in range(2)}
            
            if "Right" in hands and "Left" in hands:
                right = hands["Right"]
                left  = hands["Left"]
                
                if isPinch == True and isTwo == True and left:  # 이미지 확대액션 pts_i 설정 (양 손)
                    mid_l = [(left[4].x + left[8].x)/2, (left[4].y+left[8].y)/2, (left[4].z+left[8].z)/2]
                    mid_r = [(right[4].x + right[8].x)/2, (right[4].y+right[8].y)/2, (right[4].z+right[8].z)/2]
                    pts_i = [mid_l, mid_r]
                if isPinch == False and isTwo == True: # 이미지 확대액션 pts_c 설정 (양 손)
                    mid_l = [(left[4].x + left[8].x)/2, (left[4].y+left[8].y)/2, (left[4].z+left[8].z)/2]
                    mid_r = [(right[4].x + right[8].x)/2, (right[4].y+right[8].y)/2, (right[4].z+right[8].z)/2]
                    pts_c = [mid_l, mid_r]
                    if pts_i is None or pts_c is None:
                        continue
                    
                    scaled_img = scale_img_w_diag(img_original, pts_i, pts_c)
                #else:
                #    continue
                
        # 결과 이미지 출력
        if latest_annotated_frame is not None:
            cv.imshow("LIVE_STREAM video", cv.cvtColor(latest_annotated_frame, cv.COLOR_RGB2BGR))
            
        cv.imshow("test", scaled_img)
            
        key = cv.waitKey(delay)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27:
            break


    video.release()
    cv.destroyAllWindows()
