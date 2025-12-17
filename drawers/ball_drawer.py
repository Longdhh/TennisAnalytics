import cv2

class BallDrawer:
    def __init__(self):
        pass

    def draw_ball_annotation(self, frames, detections):
        output_frames = []
        for frame, ball_dict in zip(frames, detections):
            for _, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_frames.append(frame)
        return output_frames