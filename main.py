import os
from dotenv import load_dotenv
from roboflow import Roboflow
import torch
from ultralytics import YOLO
import supervision as sv
import cv2
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import numpy as np
import umap
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import base64
from io import BytesIO


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
        scene=annotated_frame_2, detections=all_detections
    )
    annotated_frame_2 = triangle_annotator.annotate(
        scene=annotated_frame_2, detections=ball_detections
    )

    cv2.imwrite("./runs/annotated_frame_2.jpg", annotated_frame_2)


def inference_with_player_tracking(video_path):
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


def create_player_crops(video_path):
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
    return crops


def extract_embeddings(crops):
    SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"

    EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to("cuda")
    EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

    BATCH_SIZE = 32

    crops = [sv.cv2_to_pillow(crop) for crop in crops]
    batches = chunked(crops, BATCH_SIZE)
    data = []

    with torch.no_grad():
        for batch in tqdm(batches, desc="embedding extraction"):
            inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to("cuda")
            outputs = EMBEDDINGS_MODEL(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            data.append(embeddings)
    data = np.concatenate(data)
    return data


def cluster_players_by_team(embeddings):
    REDUCER = umap.UMAP(n_components=3)
    CLUSTERING_MODEL = KMeans(n_clusters=2)

    projections = REDUCER.fit_transform(embeddings)
    clusters = CLUSTERING_MODEL.fit_predict(projections)

    return projections, clusters


def save_projection_plot_html(projections, clusters, crops):
    # inline helper: convert a crop (PIL image) to base64 string
    def pil_image_to_data_uri(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    # convert crops to base64-encoded URIs
    image_data_uris = {
        f"image_{i}": pil_image_to_data_uri(crop) for i, crop in enumerate(crops)
    }
    image_ids = np.array([f"image_{i}" for i in range(len(crops))])

    traces = []
    unique_clusters = np.unique(clusters)
    for unique_cluster in unique_clusters:
        mask = clusters == unique_cluster
        trace = go.Scatter3d(
            x=projections[mask][:, 0],
            y=projections[mask][:, 1],
            z=projections[mask][:, 2],
            mode="markers+text",  # hard-coded: markers+text
            text=clusters[mask],
            customdata=image_ids[mask],
            name=str(unique_cluster),
            marker=dict(size=6),
            hovertemplate="<b>class: %{text}</b><br>image ID: %{customdata}<extra></extra>",
        )
        traces.append(trace)

    # make cube axis range
    min_val = np.min(projections)
    max_val = np.max(projections)
    padding = (max_val - min_val) * 0.05
    axis_range = [min_val - padding, max_val + padding]

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=axis_range),
            yaxis=dict(title="Y", range=axis_range),
            zaxis=dict(title="Z", range=axis_range),
            aspectmode="cube",
        ),
        width=1000,
        height=1000,
        showlegend=True,
    )

    # embed chart HTML with custom JS for crop preview
    plotly_div = fig.to_html(
        full_html=False, include_plotlyjs=False, div_id="scatter-plot-3d"
    )
    javascript_code = f"""
    <script>
        function displayImage(imageId) {{
            var imageElement = document.getElementById('image-display');
            var placeholderText = document.getElementById('placeholder-text');
            var imageDataURIs = {image_data_uris};
            imageElement.src = imageDataURIs[imageId];
            imageElement.style.display = 'block';
            placeholderText.style.display = 'none';
        }}

        var chartElement = document.getElementById('scatter-plot-3d');
        chartElement.on('plotly_click', function(data) {{
            var customdata = data.points[0].customdata;
            displayImage(customdata);
        }});
    </script>
    """

    html_template = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                #image-container {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 200px;
                    height: 200px;
                    padding: 5px;
                    border: 1px solid #ccc;
                    background-color: white;
                    z-index: 1000;
                    box-sizing: border-box;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                }}
                #image-display {{
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }}
            </style>
        </head>
        <body>
            {plotly_div}
            <div id="image-container">
                <img id="image-display" src="" alt="Selected image" style="display: none;" />
                <p id="placeholder-text">Click a data point to display the image</p>
            </div>
            {javascript_code}
        </body>
    </html>
    """

    out_path = "./runs/player_clusters.html"
    with open(out_path, "w") as f:
        f.write(html_template)

    print(f"Interactive plot saved to {out_path}. Open it in your browser.")


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    video_path = "/home/lucas/Videos/soccer/clip_1000frames.mp4"

    inference_1(video_path)
    train()
    inference_2(video_path)
    inference_3(video_path)
    inference_with_player_tracking(video_path)
    crops = create_player_crops(video_path)
    embeddings = extract_embeddings(crops)
    projections, clusters = cluster_players_by_team(embeddings)
    save_projection_plot_html(projections, clusters, crops)


if __name__ == "__main__":
    main()

