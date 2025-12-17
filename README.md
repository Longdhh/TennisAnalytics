# Tennis Analytic System

This project is a tennis analytic system that utilizes YOLO models to analyze players performance in a tennis match. As in the first version, it can only track ball and players

## Demo



https://github.com/user-attachments/assets/a830a66b-3dc3-4728-b904-db8901d6a76f




## Key Features

*   **Player & Ball Tracking:** Detects and tracks players and the ball throughout the video.

### Working on
*   **Speed & Distance Estimation:** Calculates the real-world speed of players and the distances they cover.
*   **Hit speed** Determines hit speed of each players and calculate their average hit speed
*   **Bird-eye view** Small diagram of the match from top view

## Usage

### Running the Analysis

To run the analysis, you need to run main.py file by writing the follow bash to the terminal

```bash
python main.py -i <YOUR_VIDEO> -o <OUTPUT>
```
where:
* `-i` or `--input` is the path to the input video
* `-o` or `--output` is the path to the output video

Other flags:
* `-kp` or `--key_points` to draw court key points on the video. Default option is `False`.
* `-r` or `--read_stubs` Default option is `False`. Set is as `True` when you already run analyze as it will skip the analyze process.

## Environment
* OpenCV 4.12
* pandas 2.3.5
* NumPy 1.26.4
* Python 3.12
* ultralytics
