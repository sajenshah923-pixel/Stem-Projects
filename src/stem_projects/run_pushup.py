"""Run push-up rep tracking with an official MediaPipe task model."""

import argparse
import time
from pathlib import Path

from .pushup import RepCounter, joint_angle

LANDMARKS = {
    "left": (11, 13, 15, 23, 27),
    "right": (12, 14, 16, 24, 28),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Official pose_landmarker.task file")
    parser.add_argument("--source", default="0", help="Camera index or video filename")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--down-angle", type=float, default=90.0)
    parser.add_argument("--up-angle", type=float, default=160.0)
    parser.add_argument("--min-body-angle", type=float, default=150.0)
    parser.add_argument("--no-display", action="store_true", help="Process a file without opening a window")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    if args.no_display and args.source.isdecimal():
        parser.error("--no-display needs a finite video file, not a live webcam")
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("Install the vision extra: python -m pip install -e '.[vision]'") from exc

    counter = RepCounter(args.down_angle, args.up_angle, args.min_body_angle)
    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {args.source}")
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    start = time.monotonic()
    last_ms, index = -1, 0
    fps = cap.get(cv2.CAP_PROP_FPS) if isinstance(source, str) else 0
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int(index * 1000 / fps) if fps > 0 else int((time.monotonic() - start) * 1000)
                timestamp = max(last_ms + 1, timestamp)
                last_ms, index = timestamp, index + 1
                result = landmarker.detect_for_video(image, timestamp)
                message = "Pose not detected"
                if result.pose_landmarks:
                    points = result.pose_landmarks[0]
                    ids = LANDMARKS[args.side]
                    chosen = [points[i] for i in ids]
                    if all(getattr(point, "visibility", 1.0) >= 0.5 for point in chosen):
                        shoulder, elbow, wrist, hip, ankle = [
                            (p.x * width, p.y * height) for p in chosen
                        ]
                        try:
                            arm = joint_angle(shoulder, elbow, wrist)
                            body = joint_angle(shoulder, hip, ankle)
                            message = counter.update(arm, body)
                            for a, b in ((shoulder, elbow), (elbow, wrist), (shoulder, hip), (hip, ankle)):
                                cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (0, 200, 0), 2)
                            for point in (shoulder, elbow, wrist, hip, ankle):
                                cv2.circle(frame, tuple(map(int, point)), 5, (0, 200, 0), -1)
                        except ValueError:
                            message = "Landmarks too close to measure"
                cv2.putText(frame, f"Phase: {counter.phase}  Reps: {counter.repetitions}",
                            (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, message, (16, 62), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 255) if message != "Good form" else (0, 255, 0), 2)
                if not args.no_display:
                    cv2.imshow("Push-up form | Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Completed reps: {counter.repetitions}; rejected cycles: {counter.rejected}")


if __name__ == "__main__":
    main()

