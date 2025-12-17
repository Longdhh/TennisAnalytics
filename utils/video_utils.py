import cv2

def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    if not video.isOpened():
        print("Can't open video!")
        exit()
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def export_video(frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        video.write(frame)
    video.release()
    print("Video exported successfully!!!")
