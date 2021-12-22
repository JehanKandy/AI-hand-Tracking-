import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap=cv2.VideoCapture(0)
hands=mphands.Hands()
while True:
    data,image=cap.read()
    image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
    result=hands.process(image)

    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,mphands.HAND_CONNECTIONS)
    cv2.imshow('AI Hand Tracking', image)
    cv2.waitKey(1)
