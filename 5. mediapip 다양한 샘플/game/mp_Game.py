import numpy as np
import math
import random
import time
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Find Function
# x 는 행의 거리, y는 웹캠과의 CM 값
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C == 2차 함수

# Game 변수 지정
cx, cy = 250, 250  # 첫번 째 타겟을 그릴 위치
color = (255, 0, 255)  # 첫번 째 타겟의 색
counter = 0  # 시작 카운터
score = 0  # 시작 스코어
timeStart = time.time()  # 시작 시간은 웹캠이 열리는 시점의 현재시간
totalTime = 20  # 총 게임 시간은 20초로 지정

# 웹캠
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  ##영상 너비 = 1280
cap.set(4, 720)  ##영상 높이 = 720

#손 찾기 (탐지 민감도 0.8, 최대 손 갯수 1)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:

    # 웹 캠이 열려있는 동안 실행
    while cap.isOpened():
        success, image = cap.read() #웹캠의 이미지 가져오기
        image = cv2.flip(image, 1) #이미지 좌우 반전
        h, w, _ = image.shape #이미지의 높이(h)와 너비(w)
        rec_size = int(w/10) #이미지 너비의 1/10 사이즈 (추후 손 중앙 기준 사각형 그릴 때 사용)

        if time.time()-timeStart < totalTime:  #현재 시간 - 게임 시작 시간이 totalTime 보다 작으면 실행

            if not success:  #웹캠 이미지를 못가져왔을 경우 break
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #이미지를 BGR에서 RGB로 변환
            results = hands.process(image)  # mp.solutions.hands의 이미지 처리된 결과값
            
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  #이미지를 BGR로 다시 변환
            
            #손 랜드마크
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    #21개의 손 좌표 = hand_landmarks.landmark

                    #각 좌표값을 통해 픽셀값 도출 (손 5번과 17번에 원을 그린 후, 선 긋기)
                    hand_5 = hand_landmarks.landmark[5]  #손 5번의 좌표          
                    hand_17 = hand_landmarks.landmark[17]  #손 17번의 좌표
                    
                    hand_5_px = (int(hand_5.x*w), int(hand_5.y*h))  #손 5번의 픽셀값
                    hand_17_px = (int(hand_17.x*w), int(hand_17.y*h))  #손 17번의 픽셀값
                    
                    cv2.circle(image,hand_5_px,10,(255,0,0),2,cv2.LINE_AA) #손 5번에 원 그리기
                    cv2.circle(image,hand_17_px,10,(0,0,255),2,cv2.LINE_AA)  #손 17번에 원 그리기
                    cv2.line(image, hand_5_px, hand_17_px, (0,255,0), 5) #손 5번과 17번 사이에 선 긋기

                    # 각 좌표값을 통해 픽셀값 도출2 (손 0번과 12번을 통해 중앙 좌표 및 픽셀 구하고, 원 및 손 중앙 영역에 사각형)
                    hand_0 = hand_landmarks.landmark[0]  #손 0번의 좌표          
                    hand_12 = hand_landmarks.landmark[12]  #손 12번의 좌표

                    hand_0_px = (int(hand_0.x*w) , int(hand_0.y*h))  #손 0번의 픽셀값
                    hand_12_px = (int(hand_12.x*w) , int(hand_12.y*h))  #손 12번의 픽셀값

                    hand_cx = int((hand_12_px[0] + hand_0_px[0])/2) #손바닥 중앙의 x값
                    hand_cy = int((hand_12_px[1] + hand_0_px[1])/2)  #손바닥 중앙의 y값

                    cv2.circle(image,(hand_cx, hand_cy),10,(255,0,255),2,cv2.LINE_AA) #손바닥 중앙에 원 그리기
                    # 손바닥 중앙을 중심으로 영상의 '픽셀 너비의 1/10 사이즈(rec_size)'로 사각형 그리기
                    cv2.rectangle(image, (hand_cx - rec_size, hand_cy + rec_size),
                                    (hand_cx + rec_size, hand_cy - rec_size),(255,0,255),4)

                    #픽셀값을 통해 이미지상 좌표간의 거리, 웹캠과의 거리 도출
                    distance = int(math.dist(hand_5_px, hand_17_px)) #손 5번과 17번의 거리 좌표

                    A, B, C = coff #계수 A, B, C
                    distanceCM = A * distance ** 2 + B * distance + C  #웹캠과의 거리 Y(distanceCM) 구하기

                    #손 5번, 손 17번 간의 거리와 Y(웹캠과의 거리) 텍스트로 출력
                    cv2.putText(image, f'distance: {distance}, distanceCM: {int(distanceCM)}CM',
                                (hand_cx - rec_size, hand_cy - rec_size - 20), 1, 2, (255,0,255), 2, cv2.LINE_AA)
                    '''<<cv2.putText>>
                    param image: 문자열을 작성할 대상 행렬
                    param text: 작성할 문자열
                    param org: 문자열의 시작 좌표, 문자열에서 가장 왼쪽 하단을 의미
                    param fontFace: 문자열에 대한 글꼴
                    param fontScale: 글자 크기 확대 비율
                    param color: 글자의 색상
                    param thickness: 글자의 굵기
                    param lineType: 글자 선의 형태
                    param bottomLeftOrigin 영상의 원점 좌표를 하단 왼쪽으로 설장(기본값 - 하단 왼쪽)
                    '''
        
                    # ★터치=웹캠과의 거리 Y가 40cm 미만이고, 손의 사각 영역에 표적(cx,cy의 사이즈)이 들어오면 터치 성공 (counter=1)
                    if distanceCM < 40:
                        if hand_cx - rec_size < cx < hand_cx + rec_size and hand_cy - rec_size < cy < hand_cy + rec_size:
                            counter = 1
                    
            # ★터치 성공시 counter가 1이 되었을때 다음을 실행
            if counter:
                counter += 1 #counter는 바로 2가 되고
                color = (0, 255, 0) #타겟 버튼을 초록색으로 그림
                if counter == 3: #counter가 3이 되면 새로운 (cx, cy) 좌표에 보라색 타겟 버튼을 생성
                    cx = random.randint(100, 1100)  # 100부터 1100까지 랜덤숫자 1개 (영상 너비는 1280)
                    cy = random.randint(100, 600)  # 100부터 1100까지 랜덤숫자 1개 (영상 높이는 720)
                    color = (255, 0, 255)  # 첫번 째 타겟(터치 안된 상태)의 색상 지정
                    score +=1    #점수 누적+1
                    counter = 0  #counter는 0이 되어 원점으로 돌아감
            
            # 타겟 버튼
            cv2.circle(image, (cx, cy), 30, color, cv2.FILLED)
            cv2.circle(image, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
            cv2.circle(image, (cx, cy), 20, (255, 255, 255), 2)
            cv2.circle(image, (cx, cy), 30, (50, 50, 50), 2)

            # Game HUD
            cv2.putText(image, f'Time: {int(totalTime-(time.time()-timeStart))}', (1000, 75), 1,3,(255, 0, 255),3)
            cv2.putText(image, f'Score: {str(score).zfill(2)}', (60, 75), 1,3,(255, 0, 255),3)
        else:
            cv2.putText(image, 'Game Over', (400, 400), 1,5, (255, 0, 255),7)
            cv2.putText(image, f'Your Score: {score}', (450, 500), 1,3, (255, 0, 255),4)
            cv2.putText(image, 'Press R to restart', (460, 575), 1, 2, (255, 0, 255),4)
            cv2.putText(image, 'Press Q to quit', (480, 625), 1, 2, (255, 0, 255),4)

        # 이미지 출력
        cv2.imshow('MediaPipe Hands', image)  

        key = cv2.waitKey(5)
        if key == ord('r'):
            timeStart = time.time()
            score = 0
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
