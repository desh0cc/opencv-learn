import random
import cv2 as cv
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarkerResult

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

            starting_point = (int(values.origin_x),int(values.origin_y))

            for _ in range(10):

                differ_val = int(min(height*0.2, width*0.2))

                new_start_point = (
                    random.randint(starting_point[0]-differ_val, starting_point[0]+differ_val), 
                    random.randint(starting_point[1]-differ_val, starting_point[1]+differ_val)
                )

                new_end_point = (
                    random.randint(starting_point[0]+values.width-differ_val, starting_point[0]+values.width+differ_val), 
                    random.randint(starting_point[1]+values.height-differ_val, starting_point[1]+values.height+differ_val)
                )

                color = (
                    random.randint(0,255), # R
                    random.randint(0,255), # G
                    random.randint(0,255)  # B
                )

                cv.rectangle(
                    frame,
                    new_start_point,
                    new_end_point,
                    color,
                    -1
                )

            
        if cv.waitKey(1) == ord('q'):
            break

        cv.imshow('matrix_glitch',frame)

    cap.release()
    cv.destroyAllWindows()