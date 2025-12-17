import utils
import argparse
from trackers import PlayerTracker, BallTracker
from drawers import BallDrawer, PlayerDrawer, KeyPointsDrawer
from constants import constants
from court_key_points_detection import CourtKeyPointsDetection

def get_args():
    parser = argparse.ArgumentParser("Tennis Analytics")
    parser.add_argument("-i", "--input", type=str, default=constants.INPUT_PATH, help="Path to input video")
    parser.add_argument("-o", "--output", type=str, default=constants.OUTPUT_PATH, help="Path to output video")
    parser.add_argument("-kp", "--key_points", type=bool, default=False, help="Enable key points")
    parser.add_argument("-r", "--read_stubs", type=bool, default=False, help="If you already run prediction on the same video and don't want to predict again to save times")
    args = parser.parse_args()
    return args

def main(args):
    frames = utils.read_video(args.input)

    #Run AI models
    #Key points detection
    court_key_points_detection = CourtKeyPointsDetection()
    court_key_points = court_key_points_detection.detect(frames[0])

    # Ball tracking
    ball_tracker = BallTracker(constants.BALL_MODEL_PATH)
    ball_track = ball_tracker.ball_tracker(frames, court_key_points, args.read_stubs, stub_path=constants.BALL_TRACK_STUB_PATH)

    #Player tracking
    player_tracker = PlayerTracker(constants.PLAYER_MODEL_PATH)
    player_track = player_tracker.track_player(frames, court_key_points, args.read_stubs, stub_path=constants.PLAYER_TRACK_STUB_PATH)

    #Draw results
    #Ball
    ball_drawer = BallDrawer()
    output_frames = ball_drawer.draw_ball_annotation(frames, ball_track)

    #Player
    player_drawer = PlayerDrawer()
    output_frames = player_drawer.draw_player_annotation(output_frames, player_track)

    #Key points
    if args.key_points:
        key_points_drawer = KeyPointsDrawer()
        output_frames = key_points_drawer.draw_kps_on_vid(output_frames, court_key_points)

    #Export
    utils.export_video(output_frames, args.output)

if __name__ == '__main__':
    opt = get_args()
    main(opt)