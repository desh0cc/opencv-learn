import cv2 as cv
import os

VIDEO_FOLDER = 'videos'

def cap_webcam():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        exit()

    while True:
        ret, frame = cap.read()
        if ret:
            cv.imshow('Webcam', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def read_video(path: str):
    vidPath = os.path.join(VIDEO_FOLDER, path)
    cap = cv.VideoCapture(vidPath)

    while cap.isOpened():
        ret, frame = cap.read()
        cv.imshow('video', frame)
        delay = int(1000/60)

        if cv.waitKey(1) == ord('q'):
            break

def write_video():
    pass


if __name__ == "__main__":
    # cap_webcam()
    read_video('shibuya.mp4')