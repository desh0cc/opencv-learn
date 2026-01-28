import math
import cv2 as cv
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLand = vision.FaceDetector
FaceLandOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

options = FaceLandOptions(
        base_options=BaseOptions(model_asset_path='models/face_detection.tflite'),
        running_mode=VisionRunningMode.IMAGE,

    )
with FaceLand.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(10)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

    if not cap.isOpened():
        raise(ValueError)

    while True:
        ret,frame = cap.read()

        height,width,_ = frame.shape

        mp_image = mp.Image(mp.ImageFormat.SRGB,frame)
        annotated = landmarker.detect(mp_image)

        debug = 1

        detections = [d_obj for d_obj in annotated.detections]

        if detections:
            values = detections[0].bounding_box

            area = int(min(height*0.15,width*0.15))

            points = [
                (int(values.origin_x-area),int(values.origin_y-area)),
                (int(values.origin_x+values.width+area),int(values.origin_y+values.height+area))
            ]

            face_reg = frame[
                points[0][1]:points[1][1],
                points[0][0]:points[1][0]
            ]
            
            debug = 1

            h, w, _ = face_reg.shape

            k = int(max(h*0.3,w*0.3))

            for y in range(0, h, k):
                for x in range(0, w, k):
                    block = face_reg[y:y+k, x:x+k]

                    if block.size == 0:
                        continue

                    median_color = np.median(
                        block.reshape(-1, 3),
                        axis=0
                    ).astype(np.uint8)

                    face_reg[y:y+k, x:x+k] = median_color

            
        if cv.waitKey(1) == ord('q'):
            break

        cv.imshow('anonim-face',frame)

    cap.release()
    cv.destroyAllWindows()