import math
import sys
import random
import cv2 as cv
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from PIL import ImageFont, Image, ImageDraw

from utils import FINGERS

FaceLand = vision.FaceLandmarker
FaceLandOptions = vision.FaceLandmarkerOptions

HandLand = vision.HandLandmarker
HandLandOptions = vision.HandLandmarkerOptions

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = vision.RunningMode

options = FaceLandOptions(
    base_options=BaseOptions(model_asset_path='models/face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE
)

hand_options = HandLandOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE
)

class SmoothValue:
    def __init__(self, initial_value, smoothing=0.3):
        self.current = initial_value
        self.target = initial_value
        self.smoothing = smoothing
    
    def update(self, new_target):
        self.target = new_target
        self.current += (self.target - self.current) * self.smoothing
        return self.current
    
    def get(self):
        return self.current

class SmoothPoint:
    def __init__(self, initial_point, smoothing=0.3):
        self.current = list(initial_point)
        self.target = list(initial_point)
        self.smoothing = smoothing
    
    def update(self, new_target):
        self.target = list(new_target)
        self.current[0] += (self.target[0] - self.current[0]) * self.smoothing
        self.current[1] += (self.target[1] - self.current[1]) * self.smoothing
        return tuple(int(c) for c in self.current)
    
    def get(self):
        return tuple(int(c) for c in self.current)

def get_cords(frame: np.ndarray, landmark) -> tuple[int, int]:
    height, width, _ = frame.shape
    return (int(landmark.x*width), int(landmark.y*height))

def draw_mask(
        mask: list, 
        point: tuple[int, int], 
        frame: np.ndarray, 
        scale: float = 1, 
        direction: int = 0,
        reveal_progress: float = 1.0
    ) -> np.ndarray:

    if mask is None:
        return frame
    
    if direction == 1:
        mask = mask[::-1]
    elif direction == 2:
        mask = [row[::-1] for row in mask]
    elif direction == 3:
        mask = [row[::-1] for row in mask[::-1]]

    height, width, _ = frame.shape
    curr_point = list(point)

    try:
        font = ImageFont.truetype("CaskaydiaCoveNerdFont-Regular.ttf", int(30*scale))
    except:
        font = ImageFont.load_default(size=int(30*scale))

    dx, dy = int(10 * scale), int(20 * scale)
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    decipher_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?/~`0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    total_rows = len(mask)
    
    for row_idx, row in enumerate(mask):
        row_progress = (reveal_progress * total_rows) - row_idx
        row_progress = max(0, min(1, row_progress))
        
        if row_progress <= 0:
            break
            
        for col_idx, col in enumerate(row):
            if 0 <= curr_point[0] <= width and 0 <= curr_point[1] <= height:
                if row_progress < 1.0:
                    col_progress = row_progress + (col_idx * 0.05)
                    col_progress = min(1.0, col_progress)
                    
                    if random.random() > col_progress:
                        display_char = random.choice(decipher_chars)
                    else:
                        display_char = col
                    
                    color_value = int(255 * (0.5 + 0.5 * col_progress))
                else:
                    display_char = col
                    color_value = 255
                
                draw.text(
                    (float(curr_point[0]), float(curr_point[1])),
                    display_char,
                    font=font,
                    fill=(color_value, color_value, color_value)
                )

                if direction == 2 or direction == 3:
                    curr_point[0] -= dx
                else:
                    curr_point[0] += dx

        if direction == 1 or direction == 3:
            curr_point[1] += dy
        else:
            curr_point[1] -= dy

        curr_point[0] = point[0]

    return np.array(pil_img)

cursor_pos: tuple[int,int] = (0, 0)
current_point: int = 299
mask_pos: int = 299
pulse_time: int = 0

def mouse_callback(event, x, y, *args):
    global cursor_pos, mask_pos, pulse_time
    cursor_pos = (x, y)
    
    if event == 1:
        mask_pos = current_point
        pulse_time = 30

def main():
    global cursor_pos, mask_pos, current_point, pulse_time
    winname = 'asciimasks'

    while True:
        print("Paste ASCII mask, then Ctrl+D:")
        mask = sys.stdin.read().splitlines()
        mask = [line for line in mask if line.strip()][::-1]

        if not mask:
            print("Enter correct value pls :c")
            continue
        break

    smooth_scale = SmoothValue(1.0, smoothing=0.15)
    smooth_mask_pos = None
    
    animation_frame = 0
    animation_duration = 30
    animation_complete = False
    
    with HandLand.create_from_options(hand_options) as handmarker:
        with FaceLand.create_from_options(options) as landmarker:
            cap = cv.VideoCapture(0)
            
            cv.namedWindow(winname)
            cv.createTrackbar('show_face', winname, 0, 1, lambda x: None)
            cv.createTrackbar('direction', winname, 0, 3, lambda x: None)

            if not cap.isOpened():
                raise ValueError("No camera found")

            scale = 1
            show_face = False

            while True:
                _, frame = cap.read()

                overlay = frame.copy()

                mp_image = mp.Image(mp.ImageFormat.SRGB, frame)
                annotated = landmarker.detect(mp_image)
                hand_annotated = handmarker.detect(mp_image)

                landmarks = {
                    "face": [lm for lm in annotated.face_landmarks],
                    "hands": [lm for lm in hand_annotated.hand_landmarks]
                }

                if landmarks['hands']:
                    hand_lms = landmarks['hands'][0]

                    thumb = hand_lms[FINGERS['THUMB'][-1]]
                    index = hand_lms[FINGERS['INDEX'][-1]]

                    thb_cords = get_cords(frame, thumb)
                    idx_cords = get_cords(frame, index)

                    cv.line(overlay, thb_cords, idx_cords, (255, 255, 255), 3)
                    cv.circle(overlay, thb_cords, 5, (255, 255, 255), -1)
                    cv.circle(overlay, idx_cords, 5, (255, 255, 255), -1)
                    
                    dx = thb_cords[0] - idx_cords[0]
                    dy = thb_cords[1] - idx_cords[1]
                    dist = (dx*dx + dy*dy) ** 0.5

                    target_scale = max(1, min(dist / 80, 3))
                    
                    smooth_scale.update(target_scale)
                    scale = smooth_scale.get()

                    text_pos = (idx_cords[0], idx_cords[1] - 30)
                    scale_text = f"{scale:.1f}x"
                    
                    cv.putText(
                        overlay,
                        scale_text,
                        text_pos,
                        cv.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        3
                    )

                if landmarks['face']:
                    face_lms = landmarks['face'][0]

                    target_cords = get_cords(frame, face_lms[mask_pos])
                    
                    if smooth_mask_pos is None:
                        smooth_mask_pos = SmoothPoint(target_cords, smoothing=0.25)
                    
                    cords: tuple[int,int] = smooth_mask_pos.update(target_cords)  # type: ignore
                    
                    direction = int(cv.getTrackbarPos('direction', winname))
                    
                    if not animation_complete:
                        reveal_progress = min(1.0, animation_frame / animation_duration)
                        animation_frame += 1
                        if animation_frame >= animation_duration:
                            animation_complete = True
                    else:
                        reveal_progress = 1.0
                    
                    overlay = draw_mask(mask, cords, overlay, scale, direction, reveal_progress)
                    
                    if pulse_time > 0:
                        pulse_alpha = pulse_time / 30
                        pulse_radius = int(20 * (1 - pulse_alpha) + 10)
                        pulse_color = (
                            int(255 * pulse_alpha),
                            int(200 * pulse_alpha),
                            int(150 * pulse_alpha)
                        )
                        cv.circle(overlay, cords, pulse_radius, pulse_color, 2)
                        pulse_time -= 1
                    
                    if show_face:
                        for i, lm in enumerate(face_lms):
                            lm_scale = 1
                            lm_cords = get_cords(frame, lm)

                            distance = math.dist(lm_cords, cursor_pos)

                            if distance < 30:
                                lm_scale = 5
                            if distance < 20:
                                lm_scale = 10
                            if distance < 10:
                                lm_scale = 15
                                current_point = i

                            color = (255, 255, 255) if i != mask_pos else (100, 255, 150)
                            cv.circle(overlay, lm_cords, 2 * lm_scale, color, -1)

                cv.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

                if cv.waitKey(1) == ord('q'):
                    break

                show_face = bool(cv.getTrackbarPos('show_face', winname))

                cv.imshow(winname, frame)
                cv.setMouseCallback(winname, mouse_callback)

            cap.release()
            cv.destroyAllWindows()

if __name__ == '__main__':
    main()