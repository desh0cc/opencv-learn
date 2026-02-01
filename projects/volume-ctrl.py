import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2 as cv
import numpy as np

import pyvolume

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

def get_points(landmark, height: int, width: int):
    return (int(landmark.x * width), int(landmark.y*height))

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(10)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)

    if not cap.isOpened():
        raise ValueError

    while True:
        ret,frame = cap.read()

        if cv.waitKey(1) == ord('q'):
            break

        frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, frameRGB)

        height,width,_ = frame.shape

        annotated = landmarker.detect(mp_image)
        landmarks = [landmark for landmark in annotated.hand_landmarks]

        THUMB: list[int] = [i for i in range(1,5)] # THUMB
        FINGERS: dict[int, list[int]] = {
            5: [6,7,8], # INDEX
            9: [10,11,12], # MIDDLE 
            13: [14,15,16], # RING 
            17: [18,19,20] # PINKY
        }

        if landmarks:
            fingers = landmarks[0]

            thumb = get_points(fingers[3], height, width)
            index = get_points(fingers[7], height, width)

            cv.line(frame, thumb, index, (255,255,255), 3, 1)
            cv.circle(frame,thumb,20,(255,255,255),-1,2)
            cv.circle(frame,index,20,(255,255,255),-1,2)

            dx = thumb[0] - index[0]
            dy = thumb[1] - index[1]
            dt = int((dx*dx + dy*dy) ** 0.5)
            
            if dt < 0:
                dt = 0
            elif dt > 100:
                dt = 100

            pyvolume.custom(percent=dt)


        deb = 1

        cv.imshow('hands',frame)

    cap.release()
    cv.destroyAllWindows()