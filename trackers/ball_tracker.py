import os
import pickle
import pandas as pd
import utils
from supervision import ByteTrack
from ultralytics import YOLO

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = ByteTrack

    def ball_tracker(self, frames, kps, read_from_stub, stub_path=None):
        if read_from_stub is True and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_tracks = pickle.load(f)
            return ball_tracks

        # Track ball by batch
        batch_size = 20
        frame_nums = len(frames)
        detections = []
        for i in range(0, frame_nums, batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf = 0.2)
            detections += detection_batch

        # Save into list of dictionaries
        ball_tracks = []
        for frame_num, detection in enumerate(detections):
            ball_dict = {}
            chosen_ball = self.filter_ball_detection(detection, kps)
            ball_dict[1] = chosen_ball
            ball_tracks.append(ball_dict)
        ball_tracks = self.interpolate_ball(ball_tracks)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_tracks, f)
        return ball_tracks

    # Choose detection with the least distance from court in case multiple balls detected
    def filter_ball_detection(self, detection, kps):
        chosen_ball = []
        for box in detection.boxes:
            result = box.xyxy.tolist()[0]
            min_distance = float("inf")
            ball_center = utils.get_box_center(result)
            for i in range(0, len(kps), 2):
                kp = (kps[i], kps[i + 1])
                ball_kp_distance = utils.calculate_distance(ball_center, kp)
                if ball_kp_distance < min_distance:
                    min_distance = ball_kp_distance
                    chosen_ball = result
        return chosen_ball

    # Ball interpolation (fill missing detections)
    def interpolate_ball(self, ball_tracks):
        ball_tracks = [x.get(1, []) for x in ball_tracks]
        ball_tracks_df = pd.DataFrame(ball_tracks, columns=['x1', 'y1', 'x2', 'y2'])
        ball_tracks_df = ball_tracks_df.interpolate()
        ball_tracks_df = ball_tracks_df.bfill()
        ball_tracks = [{1: x} for x in ball_tracks_df.to_numpy()]

        return ball_tracks