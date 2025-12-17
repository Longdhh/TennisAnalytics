import os
import pickle
from ultralytics import YOLO
import utils
import pandas as pd
from collections import defaultdict

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def track_player(self, frames, court_key_points, read_from_stub, stub_path=None):
        # Read from stub if exists
        if read_from_stub is True and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                player_tracks = pickle.load(f)
            return player_tracks

        # Track players by batch
        batch_size = 20
        frame_num = len(frames)
        detections = []

        for i in range(0, frame_num, batch_size):
            detection_batch = self.model.track(frames[i:i+batch_size], persist=True, conf=0.1)
            detections += detection_batch

        # Save results into a list of dictionaries
        class_name_dict = detections[0].names
        player_tracks = []

        for frame, detection in enumerate(detections):
            player_dict = self.filter_non_person(detection, class_name_dict)
            player_tracks.append(player_dict)

        # Only track players
        player_tracks = self.filter_non_players(court_key_points, player_tracks)

        # Interpolate players positions
        list_players = self.get_players(court_key_points, player_tracks[0])
        player_tracks = self.interpolate_players(list_players, player_tracks)

        # Save stubs if not exists
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_tracks, f)
        return player_tracks

    def filter_non_person(self, detection, class_name_dict):
        player_dict = {}
        for box in detection.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = int(box.cls.tolist()[0])
            object_cls_name = class_name_dict[object_cls_id]
            if object_cls_name == 'person':
                player_dict[track_id] = result
        return player_dict

    def get_players(self, kps, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            min_distance = float("inf")
            player_position = utils.get_box_center(bbox)
            for i in range(24, len(kps), 2):
                kp = (kps[i], kps[i+1])
                player_kp_distance = utils.calculate_distance(kp, player_position)
                if player_kp_distance < min_distance:
                    min_distance = player_kp_distance
            distances.append((track_id, min_distance))
        distances.sort(key=lambda x: x[1])
        players = [distances[0][0], distances[1][0]]
        return players

    def filter_non_players(self, kps, player_tracks):
        list_players = self.get_players(kps, player_tracks[0])
        filtered_tracks = []
        for player_dict in player_tracks:
            filtered_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in list_players}
            filtered_tracks.append(filtered_dict)
        return filtered_tracks

    def interpolate_players(self, list_players, player_tracks):
        list_df = []
        for track_id in list_players:
            player_track = [x.get(track_id, []) for x in player_tracks]
            player_tracks_df = pd.DataFrame(player_track, columns=['x1', 'y1', 'x2', 'y2'])
            player_tracks_df = player_tracks_df.interpolate()
            player_tracks_df = player_tracks_df.bfill()
            list_df.append(player_tracks_df)

        new_track = []
        for i in range(len(list_df[0])):
            player_dict = {}
            for index, track_id in enumerate(list_players):
                player_dict[track_id] = list_df[index].loc[i].to_numpy()
            new_track.append(player_dict)
        return new_track