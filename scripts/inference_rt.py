import argparse
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="/dev/video6")
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--imgsz", type=int, default=640)

    args = parser.parse_args()

    cam_index = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"Requested {args.width}x{args.height}, camera gave "
          f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source {args.source}")

    model = YOLO(model=args.weights)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed, stopping.")
                break

            results = model.predict(
                source=frame,
                conf=args.confidence,
                imgsz=args.imgsz,
                verbose=False,
            )

            annotated = results[0].plot()  # draws boxes on the frame
            cv2.imshow("inference", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()