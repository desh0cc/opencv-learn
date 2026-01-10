import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2 as cv
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2
    )
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)

    if not cap.isOpened():
        raise(ValueError)

    while True:
        ret,frame = cap.read()

        if cv.waitKey(1) == ord('q'):
            break

        frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, frameRGB)

        height,width,_ = frame.shape

        annotated = landmarker.detect(mp_image)

        for landmarks in annotated.hand_landmarks:
            print(len(landmarks))
            for lm in landmarks:
                cv.drawMarker(frame,(int(lm.x * width),int(lm.y*height)),(255,0,0))
                cv.circle(frame,(int(lm.x * width),int(lm.y*height)),20,(0,0,255))


        deb = 1

        cv.imshow('hands',frame)

    cap.release()
    cv.destroyAllWindows()