# Push-up form analysis

The portfolio shows a webcam-based push-up prototype using Python, OpenCV, MediaPipe, NumPy, and elbow-angle calculations. This implementation uses the current [MediaPipe Pose Landmarker Tasks API](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) because older `mp.solutions.pose` code no longer works in newer MediaPipe releases.

1. Install the vision dependencies: `python -m pip install -e '.[vision]'`.
2. Download the official **Pose Landmarker** task model from Google's [model section](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models). Save it locally as `models/pose_landmarker.task` (ignored by Git).
3. From the repository root, run:

```bash
python -m stem_projects.run_pushup --model models/pose_landmarker.task --source 0 --side left
# Or process a saved clip:
python -m stem_projects.run_pushup --model models/pose_landmarker.task --source your_clip.mp4
```

Press `Q` to quit. For headless processing of a video file, add `--no-display`. A completed repetition requires the measured elbow to pass the down threshold (90°) and return above the up threshold (160°); a shoulder/hip/ankle angle below 150° flags possible loss of body alignment and rejects that cycle. These thresholds can be changed with `--down-angle`, `--up-angle`, and `--min-body-angle`.

Use a side view with the chosen arm and hip/ankle visible. Camera perspective and missing landmarks can affect counts. Thresholds and the body-alignment rule are choices made for this reconstruction; the portfolio does not provide the original complete form rules or a validation study. This tool is not a safety assessment.

