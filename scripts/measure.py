import argparse
import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO


STANDARD_SIZE = {"M4": 7, "M6": 10, "M8": 13, "M10": 17, "M12": 19, "M16": 25}

def get_nearest_size(size, dev=5.0):
    key, val = min(STANDARD_SIZE.items(), key=lambda kv: abs(kv[1] - size))
    if abs(val - size) > dev:
        return "Unknown", None
    return key, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="/dev/video6")
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(model=args.weights)

    camera_stream = rs.pipeline()
    camera_config = rs.config()
    camera_config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    camera_config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, 30)

    profile = camera_stream.start(camera_config)
    align = rs.align(rs.stream.color)
    focal_length = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics().fx
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print(f"Streaming at {args.width} x {args.height}. Press q to quit")

    try:
        while True:
            frames = align.process(camera_stream.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            results = model.predict(source=color_image, conf=args.confidence,
                                     imgsz=args.imgsz, verbose=False)

            annotated = color_image.copy()

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pixel_width = x2 - x1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # shrink the ROI a bit inward to avoid edge/background contamination
                pad_x = max(1, int(pixel_width * 0.15))
                pad_y = max(1, int((y2 - y1) * 0.15))
                rx1, rx2 = x1 + pad_x, x2 - pad_x
                ry1, ry2 = y1 + pad_y, y2 - pad_y

                depth_region = depth_image[ry1:ry2, rx1:rx2]
                valid_region = depth_region[depth_region > 0]

                if valid_region.size == 0:
                    label = "No depth data"
                else:
                    depth_mm = np.percentile(valid_region, 10) * depth_scale * 1000
                    real_width = (pixel_width * depth_mm) / focal_length
                    size_name, _ = get_nearest_size(real_width)
                    label = f"{size_name} ({real_width:.1f}mm)"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.imshow("inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera_stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()