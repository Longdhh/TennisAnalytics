import cv2

class KeyPointsDrawer:
    def __init__(self):
        pass

    def draw_kps_on_vid(self, frames, kps):
        output_frames = []
        for frame in frames:
            for i in range(0, len(kps), 2):
                x = int(kps[i])
                y = int(kps[i + 1])
                cv2.putText(frame, str(i // 2 + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            output_frames.append(frame)
        return output_frames