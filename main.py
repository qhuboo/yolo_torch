import os
from dotenv import load_dotenv
from roboflow import Roboflow
import torch
from ultralytics import YOLO
import supervision as sv
import cv2
from tqdm import tqdm

load_dotenv()


def inference_1(video_path):
    print("Inference 1 ...")
    frame_generator = sv.get_video_frames_generator(video_path)
    frame = next(frame_generator)

    # INFERENCE 1
    # This is the first inference test using the base YOLOv11x model
    model_base_x = YOLO("yolo11s.pt")

    result_1 = model_base_x.predict(
        frame, save=True, project="runs", name="inference_1"
    )[0]


def train():
    print("Beginning training ...")
    # This is to get the roboflow dataset and train.
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(20)
    dataset = version.download("yolov11")

    model_base_s = YOLO("yolo11s.pt")

    results = model_base_s.train(
        data="./football-players-detection-20/data.yaml",
        save=True,
        epochs=50,
        imgsz=1280,
        plots=True,
        device=0,
        batch=6,
        project="runs/detect",
    )


def inference_2(video_path):
    print("Inference 2 ...")
    # Now we use this new model to do inference again.
    model_trained = YOLO("./runs/detect/train/weights/best.pt")

    frame_generator = sv.get_video_frames_generator(video_path)
    frame = next(frame_generator)

    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(["#FF8C00", "#00BFFF", "#FF1493", "#FFD700"]),
        thickness=2,
    )

    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#FF8C00", "#00BFFF", "#FF1493", "#FFD700"]),
        text_color=sv.Color.from_hex("#000000"),
    )

    result_2 = model_trained.predict(
        frame, save=True, project="runs", name="inference_2"
    )[0]

    detections = sv.Detections.from_ultralytics(result_2)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(
            detections["class_name"], detections.confidence
        )
    ]

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    cv2.imwrite("./runs/annotated_frame.jpg", annotated_frame)


def inference_3(video_path):
    print("Inference 3 ...")
    # We do another inference with different annotations
    BALL_ID = 0

    model_trained = YOLO("./runs/detect/train/weights/best.pt")
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "FFD700"]), thickness=2
    )

    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"), base=25, height=21, outline_thickness=1
    )

    frame_generator_2 = sv.get_video_frames_generator(video_path)
    frame_2 = next(frame_generator_2)

    result_3 = model_trained.predict(
        frame_2, save=True, project="runs", name="inference_3"
    )[0]
    detections_2 = sv.Detections.from_ultralytics(result_3)

    ball_detections = detections_2[detections_2.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections_2[detections_2.class_id != BALL_ID]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections.class_id -= 1

    annotated_frame_2 = frame_2.copy()
    annotated_frame_2 = ellipse_annotator.annotate(
        scene=annotated_frame_2, etections=all_detections
    )
    annotated_frame_2 = triangle_annotator.annotate(
        scene=annotated_frame_2, detections=ball_detections
    )

    cv2.imwrite("./runs/annotated_frame_2.jpg", annotated_frame_2)


def tracking(video_path):
    print("Inference with tracking ...")
    # We will do inference again but this time with the video.
    BALL_ID = 0

    model_trained = YOLO("./runs/detect/train/weights/best.pt")
    ellipse_annotator_2 = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]), thickness=2
    )

    label_annotator_2 = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
        text_color=sv.Color.from_hex("#000000"),
        text_position=sv.Position.BOTTOM_CENTER,
    )

    triangle_annotator_2 = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"),
        base=25,
        height=21,
        outline_thickness=1,
    )

    tracker = sv.ByteTrack()

    frame_generator_3 = sv.get_video_frames_generator(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "./runs/video_tracked.mp4", fourcc, fps, (frame_width, frame_height)
    )

    for frame in frame_generator_3:
        result_tracking = model_trained.predict(frame, conf=0.3)[0]

        detections = sv.Detections.from_ultralytics(result_tracking)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections.class_id -= 1
        all_detections = tracker.update_with_detections(detections=all_detections)

        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator_2.annotate(
            scene=annotated_frame, detections=all_detections
        )
        annotated_frame = label_annotator_2.annotate(
            scene=annotated_frame, detections=all_detections, labels=labels
        )
        annotated_frame = triangle_annotator_2.annotate(
            scene=annotated_frame, detections=ball_detections
        )

        out.write(annotated_frame)
    out.release()


def create_crops(video_path):
    print("Generating player crops ...")
    # Getting training data for cluster model
    PLAYER_ID = 2
    STRIDE = 30

    model_trained = YOLO("./runs/detect/train/weights/best.pt")
    frame_generator_4 = sv.get_video_frames_generator(
        source_path=video_path, stride=STRIDE
    )

    crops = []
    for frame in tqdm(frame_generator_4, desc="collecting crops"):
        result = model_trained.predict(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    print(f"Total crops collected: {len(crops)}.")

    out_dir = "./runs/player_crops"
    os.makedirs(out_dir, exist_ok=True)

    for i, crop in enumerate(crops):
        filename = os.path.join(out_dir, f"crop_{i:04d}.jpg")
        cv2.imwrite(filename, crop)


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    video_path = "/home/lucas/Videos/soccer/clip_1000frames.mp4"

    # inference_1(video_path)
    # train()
    # inference_2(video_path)
    # inference_3(video_path)
    # tracking(video_path)
    create_crops(video_path)


if __name__ == "__main__":
    main()

