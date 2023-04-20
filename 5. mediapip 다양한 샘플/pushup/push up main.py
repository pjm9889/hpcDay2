# from asyncore import poll3
# from fcntl import F_SEAL_SEAL
from ctypes import resize
from unittest import result
import cv2
import mediapipe as mp
import math
import pandas as pd
#출처: https://youtu.be/sRqfQPlNa3M
# https://google.github.io/mediapipe/solutions/pose
cap = cv2.VideoCapture("pushup.mp4")

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
a=[] # 각도 담는 리스트
count = 0
position = None
while True:
    succes, img = cap.read()
    if not succes:
        break
    
    #높이 넓이 각도
    h, w, c = img.shape

    #색을 변환해 포즈 추정
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    #포즈를 검출한다면
    if result.pose_landmarks:
        #mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS) #전체 랜드마크 표시

        p1 = result.pose_landmarks.landmark[11] #왼쪽 어깨
        p2 = result.pose_landmarks.landmark[13] #왼쪽 팔꿈치
        p3 = result.pose_landmarks.landmark[15] #왼쪽 손목
        p4 = result.pose_landmarks.landmark[23] #왼쪽 대관절
        p5 = result.pose_landmarks.landmark[25] #왼쪽 무릎
        p6 = result.pose_landmarks.landmark[27] #왼쪽 발목

        #실제 좌표
        x1 = int(p1.x*w) #왼쪽 어깨
        y1 = int(p1.y*h) #왼쪽 어깨

        x2 = int(p2.x*w) #왼쪽 팔꿈치
        y2 = int(p2.y*h) #왼쪽 팔꿈치

        x3 = int(p3.x*w) #왼쪽 손목
        y3 = int(p3.y*h) #왼쪽 손목

        x4 = int(p4.x*w) #왼쪽 대관절   
        y4 = int(p4.y*h) #왼쪽 대관절

        x5 = int(p5.x*w) #왼쪽 무릎
        y5 = int(p5.y*h) #왼쪽 무릎

        x6 = int(p6.x*w) #왼쪽 발목
        y6 = int(p6.y*h) #왼쪽 발목

        #선 그리기
        cv2.line(img, (x4, y4), (x5, y5), (255, 255, 255), 3)
        cv2.line(img, (x6, y6), (x5, y5), (255, 255, 255), 3)
        cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3) 
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

        #원 그리기
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), 5)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), 5)
        cv2.circle(img, (x4, y4), 10, (255, 0, 0), 5)
        cv2.circle(img, (x5, y5), 10, (255, 0, 0), 5)
        cv2.circle(img, (x6, y6), 10, (255, 0, 0), 5)
        cv2.circle(img, (x3, y3), 10, (255, 0, 0), 5)

        #좌측 상체 역탄젠트 계산 (팔꿈치 밖을 기준으로 2번 각도(두 점(1번과 3번) 사이의 절대각도)) 
        #atan2는 두 점 사이의 상대좌표를 받아 절대각을 -π ~ π의 라디안 값으로 반환
        #상대적인 위치라 x나 y가 음수값이 될 수 있어, +, - 극이 표시되는 데카르트 좌표계에서 사용할 때 유용
        angle1 = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle1 < 0:
            angle1 = int(angle1 + 360)
        #좌측 하체 역탄젠트 (무릎 밖을 기준으로 5번 각도(두 점(4번과 6번) 사이의 절대각도)) 
        angle = math.degrees(math.atan2(y6-y5, x6-x5) - math.atan2(y4-y5, x4-x5))
        #각도가 -면 360도가 나오도록 0~360도 범위
        if angle < 0:
            angle = int(angle + 360)
        a.append([angle1, angle]) #각도 리스트에 담기
        
        #angle1
        print("angle1",int(angle1))
        print("angle",int(angle))
        print("-----------------------------------------------------------------")

        #푸시업 숫자세기(270보다 크게 내려갔을 때(down)와 270에 걸렸는데 작아졌을 때 position=up이 됐다=count+1)

        if angle1 > 270:
            position = "down"
        if angle1 < 200 and position =='down':
            position="up"
            count +=1
            #print(count)
            
        # #푸시업 횟수 칸
        cv2.putText(img, f'Score: {count}', (60, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        #각도 숫자 보이기
        cv2.putText(img, 'angle:'+str(int(angle)), (x5-50, y5-50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cv2.putText(img, 'angle1:'+str(int(angle1)), (x2-50, y2-50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
    cv2.imshow("Image", cv2.resize(img, (1280, 720)))
    

    if cv2. waitKey(5) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
df=pd.DataFrame(a)
df.to_csv('abc.csv')