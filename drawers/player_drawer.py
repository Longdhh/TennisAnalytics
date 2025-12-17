import cv2

class PlayerDrawer:
    def __init__(self):
        pass

    def draw_player_annotation(self, frames, detections):
        output_frames = []
        for frame, player_dict in zip(frames, detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255,255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_frames.append(frame)
        return output_frames