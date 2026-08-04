import cv2
import argparse
import os
import time
import logging
import numpy as np
from deepface import DeepFace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')


class FacePipeline:
    """
    Main pipeline for face detection, alignment, identification (ArcFace),
    and age estimation using DeepFace.
    """
    def __init__(self, db_path='db', rec_threshold=0.50, min_conf=0.60,
                 detector_backend='yolov8', min_face_size=40):
        logging.info(f"Initializing Face Pipeline (Detector: {detector_backend})...")
        self.db_path = db_path
        self.rec_threshold = rec_threshold
        self.min_conf = min_conf
        self.detector_backend = detector_backend
        self.min_face_size = min_face_size
        self.last_detected_faces = []  # Temporary memory for detected faces metadata

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)

        # Warm up DeepFace age model with a dummy image to reduce initial runtime latency
        dummy_face = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(dummy_face, actions=['age'], enforce_detection=False,
                             detector_backend="skip", silent=True)
        except Exception:
            pass

    def _get_distance_column(self, df):
        """
        Dynamically locate the distance metric column in the DataFrame returned by DeepFace.find
        to avoid relying on hardcoded column indexes.
        """
        candidates = [c for c in df.columns if 'distance' in c.lower() or 'cosine' in c.lower()]
        if candidates:
            return candidates[0]
        # Fallback for legacy DeepFace versions
        return df.columns[-1]

    def run_ml_inference(self, frame):
        """
        Executes deep learning inference (detection, recognition, age estimation)
        on a given frame.
        """
        self.last_detected_faces = []
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            face_objs = DeepFace.extract_faces(
                img_path=frame_rgb,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True  # Enables face alignment for improved recognition accuracy
            )
        except Exception as e:
            logging.debug(f"extract_faces found no face / failed: {e}")
            return

        for idx, face_obj in enumerate(face_objs):
            facial_area = face_obj["facial_area"]
            x_min, y_min = max(0, int(facial_area["x"])), max(0, int(facial_area["y"]))
            x_max = min(w, x_min + int(facial_area["w"]))
            y_max = min(h, y_min + int(facial_area["h"]))
            score = face_obj.get("confidence", 0.0)

            box_w, box_h = x_max - x_min, y_max - y_min

            # Filter out faces with low detector confidence
            if score < self.min_conf:
                continue

            # Filter out faces smaller than specified pixel size threshold
            if box_w < self.min_face_size or box_h < self.min_face_size:
                continue

            # Extract normalized aligned face array [0, 1] provided by extract_faces
            aligned_face = face_obj["face"]
            aligned_face_uint8 = np.clip(aligned_face * 255, 0, 255).astype(np.uint8)
            aligned_face_bgr = cv2.cvtColor(aligned_face_uint8, cv2.COLOR_RGB2BGR)

            # Raw unaligned crop used for age estimation and visual bounding box
            raw_crop = frame[y_min:y_max, x_min:x_max]
            if raw_crop.shape[0] == 0 or raw_crop.shape[1] == 0:
                continue

            identity = "Unknown"
            age = "N/A"
            box_color = (0, 165, 255)  # Orange for unknown faces

            # Face recognition on aligned face image
            try:
                recognition = DeepFace.find(
                    img_path=aligned_face_bgr, db_path=self.db_path, model_name='ArcFace',
                    enforce_detection=False, detector_backend="skip", silent=True
                )
                if len(recognition) > 0 and not recognition[0].empty:
                    df = recognition[0]
                    dist_col = self._get_distance_column(df)
                    distance = df[dist_col].values[0]

                    if distance < self.rec_threshold:
                        identity = os.path.basename(os.path.dirname(df['identity'][0])).replace("_", " ")
                        box_color = (0, 255, 0)  # Green for recognized identity
            except Exception as e:
                logging.debug(f"Face recognition failed for index {idx}: {e}")

            # Age analysis on raw crop
            try:
                analysis = DeepFace.analyze(
                    img_path=raw_crop, actions=['age'],
                    enforce_detection=False, detector_backend="skip", silent=True
                )
                age = analysis[0]['age'] if isinstance(analysis, list) else analysis['age']
            except Exception:
                pass

            # Cache detected face metadata for drawing
            self.last_detected_faces.append({
                "box": (x_min, y_min, x_max, y_max),
                "label": f"{identity} | Age: {age} | Conf: {score:.2f}",
                "color": box_color
            })

    def draw_faces(self, frame):
        """
        Draw bounding boxes and smart non-overlapping labels on the frame.
        Handles label collisions, prevents text overflow, and provides spacious padding.
        """
        h, w = frame.shape[:2]
        font_scale = max(0.5, w / 1500)
        thickness = max(1, int(w / 800))

        # =========================================================================
        # Label background padding settings (adjust as needed)
        # =========================================================================
        padding_x = 12  # Horizontal padding (left and right)
        padding_y = 8  # Vertical padding (top and bottom)
        # =========================================================================

        occupied_label_rects = []

        for idx, face in enumerate(self.last_detected_faces):
            x_min, y_min, x_max, y_max = face["box"]
            color = face["color"]
            label = face["label"]

            # Draw primary face bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness + 1)

            # Calculate text dimensions and baseline
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Calculate total bounding box dimensions (text + padding + baseline)
            box_w = text_w + (2 * padding_x)
            box_h = text_h + baseline + (2 * padding_y)

            def is_valid_position(rect):
                rx1, ry1, rx2, ry2 = rect
                if rx1 < 0 or ry1 < 0 or rx2 > w or ry2 > h:
                    return False
                for ox1, oy1, ox2, oy2 in occupied_label_rects:
                    if not (rx2 <= ox1 or rx1 >= ox2 or ry2 <= oy1 or ry1 >= oy2):
                        return False
                return True

            # 4 candidate positions for label placement
            pos_above = (x_min, y_min - box_h, x_min + box_w, y_min)
            pos_below = (x_min, y_max, x_min + box_w, y_max + box_h)
            pos_inside_top = (x_min, y_min, x_min + box_w, y_min + box_h)
            pos_inside_bottom = (x_min, y_max - box_h, x_min + box_w, y_max)

            candidates = [pos_above, pos_below, pos_inside_top, pos_inside_bottom]

            chosen_rect = None
            for cand in candidates:
                if is_valid_position(cand):
                    chosen_rect = cand
                    break

            # Fallback strategy for highly crowded areas
            if chosen_rect is None:
                base_cand = pos_above if idx % 2 == 0 else pos_below
                rx1 = max(0, min(base_cand[0], w - box_w))
                ry1 = max(0, min(base_cand[1], h - box_h))
                chosen_rect = (rx1, ry1, rx1 + box_w, ry1 + box_h)

            rx1, ry1, rx2, ry2 = chosen_rect
            occupied_label_rects.append(chosen_rect)

            # Draw filled colored background rectangle
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, -1)

            # Calculate exact text position to center it inside the background box
            text_x = rx1 + padding_x
            text_y = ry1 + padding_y + text_h

            # Draw label text over the background box
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        return frame


def process_image(pipeline, image_path, output_path=None):
    """Processes a single image file, renders overlays, and saves the output if requested."""
    logging.info(f"Processing image file: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        logging.error(f"Failed to read image from path: {image_path}")
        return

    pipeline.run_ml_inference(frame)
    result_frame = pipeline.draw_faces(frame)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, result_frame)
        logging.info(f"Annotated image saved successfully to: {output_path}")

    cv2.imshow("Face Pipeline Demo (Image)", result_frame)
    logging.info("Press any key on the image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video_stream(pipeline, source, output_path=None, frame_skip=15):
    """Processes video file, RTSP, or webcam stream, renders overlays, and optionally records output."""
    logging.info(f"Opening media stream: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logging.error(f"Could not open video source: {source}")
        return

    # Fetch frame properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_writer = fps_in if (fps_in and fps_in > 0) else 25.0  # Fallback FPS for live streams/webcams

    video_writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps_writer, (width, height))
        logging.info(f"Video writer initialized. Output will be saved to: {output_path}")

    frame_count = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run heavy ML inference only every N frames
            if frame_count % frame_skip == 0:
                pipeline.run_ml_inference(frame)

            result_frame = pipeline.draw_faces(frame)

            # Calculate actual processing FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            fps = (1 / time_diff) if time_diff > 0 else 0.0
            prev_time = curr_time

            cv2.putText(result_frame, f"FPS: {int(fps)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Write annotated frame to video file if output path is specified
            if video_writer is not None:
                video_writer.write(result_frame)

            cv2.imshow("Face Pipeline Demo", result_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                break

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
            logging.info(f"Video file saved successfully to: {output_path}")
        cv2.destroyAllWindows()


def is_image_file(source_path):
    """Determines whether the given input source is a static image file."""
    if isinstance(source_path, str) and os.path.isfile(source_path):
        ext = os.path.splitext(source_path)[1].lower()
        return ext in IMAGE_EXTENSIONS
    return False


def main():
    parser = argparse.ArgumentParser(description="Real-Time Face Recognition & Age Pipeline with OpenCV and DeepFace")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image/video file, camera index (e.g., 0), or RTSP URL")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output file path to save annotated image/video result")
    parser.add_argument("--db", type=str, default="db",
                        help="Path to database folder containing identity images")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="ArcFace recognition distance threshold")
    parser.add_argument("--conf", type=float, default=0.65,
                        help="Minimum face detection confidence threshold")
    parser.add_argument("--min-face-size", type=int, default=40,
                        help="Minimum face bounding box width/height in pixels")
    parser.add_argument("--detector", type=str, default="yolov8",
                        help="Face detection backend (yolov8, retinaface, opencv, etc.)")
    parser.add_argument("--frame-skip", type=int, default=15,
                        help="Run ML inference every N frames for video streams")
    args = parser.parse_args()

    pipeline = FacePipeline(
        db_path=args.db,
        rec_threshold=args.threshold,
        min_conf=args.conf,
        detector_backend=args.detector,
        min_face_size=args.min_face_size
    )

    source = int(args.source) if args.source.isdigit() else args.source

    if is_image_file(source):
        process_image(pipeline, source, output_path=args.output)
    else:
        process_video_stream(pipeline, source, output_path=args.output, frame_skip=args.frame_skip)


if __name__ == "__main__":
    main()
