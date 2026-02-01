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
    running_mode=VisionRunningMode.IMAGE
)

ASCII_CHARS = r'$@B%8&WM#*/\|()1{}[]?-_+~<>i!lI;:,"^`'

def pixel_to_ascii(pixel):
    n = len(ASCII_CHARS)
    index = int(pixel / 255 * (n - 1))
    return ASCII_CHARS[index]
    
def convert_to_ascii_art(frame: np.ndarray, scale: int=8) -> np.ndarray:
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        h, w = gray.shape
        small = cv.resize(gray, (w // scale, h // scale))

        ascii_img = np.zeros((h, w, 3), dtype=np.uint8)

        for y in range(small.shape[0]):
            for x in range(small.shape[1]):
                pixel = small[y, x]
                char = pixel_to_ascii(pixel)

                cv.putText(
                    ascii_img,
                    char,
                    (x * scale, y * scale),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA
                )

        return ascii_img
    except:
        return frame
    
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

        # frame = convert_to_ascii_art(frame)

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

            if face_reg is None:
                continue

            frame[
                points[0][1]:points[1][1],
                points[0][0]:points[1][0]
            ] = convert_to_ascii_art(face_reg)


        if cv.waitKey(1) == ord('q'):
            break

        cv.imshow('asciiface', frame)

    cap.release()
    cv.destroyAllWindows()