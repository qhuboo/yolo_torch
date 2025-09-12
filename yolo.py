import os
from dotenv import load_dotenv
from roboflow import Roboflow
import torch
from ultralytics import YOLO
import supervision as sv
import cv2

load_dotenv()


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    video_path = "/home/lucas/Videos/soccer/clip_1000frames.mp4"
    frame_generator = sv.get_video_frames_generator(video_path)
    frame = next(frame_generator)

    # INFERENCE 1
    # This is the first inference test using the base YOLOv11x model
    model_base_x = YOLO("yolo11s.pt")

    result_1 = model_base_x.predict(
        frame, save=True, project="runs", name="inference_1"
    )[0]

    # TRAINING 1
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

    # INFERENCE 2
    # Now we use this new model to do inference again.
    model_trained = YOLO("./runs/detect/train/weights/best.pt")

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

    # INFERENCE 3
    # We do another inference with different annotations
    BALL_ID = 0

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

    ball_detections = detections_2[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections[detections.class_id != BALL_ID]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections.class_id -= 1

    annotated_frame_2 = frame_2.copy()
    annotated_frame_2 = ellipse_annotator.annotate(
        scene=annotated_frame_2, detections=all_detections
    )
    annotated_frame_2 = triangle_annotator.annotate(
        scene=annotated_frame_2, detections=ball_detections
    )

    cv2.imwrite("./runs/annotated_frame_2.jpg", annotated_frame_2)


if __name__ == "__main__":
    main()
