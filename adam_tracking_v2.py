# Importing packages
from ultralytics import YOLO
import cv2
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from distance_calculation import distance_calculation
from sklearn.metrics import pairwise_distances


def convert_coords(coords: list):
    x_min, y_min, x_max, y_max = coords
    y_min, y_max = y_min * 1.35625, y_max * 1.35625
    x_min, x_max = x_min * 3.35, x_max * 3.35
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def resolve_lost_tracks(
    lost_tracks_prev: dict, lost_tracks_next: dict, metric: str = "cosine"
):
    prev_id_to_next_id = dict()

    if metric == "cosine":
        similarity_matrix = cosine_similarity(
            list(lost_tracks_prev.values()), list(lost_tracks_next.values())
        )
        threshold = 0.9999
    elif metric == "manhattan":
        similarity_matrix = np.negative(
            pairwise_distances(
                list(lost_tracks_prev.values()),
                list(lost_tracks_next.values()),
                metric="manhattan",
            )
        )
        threshold = -50

    else:
        raise ValueError("Input must be either 'cosine' or 'manhattan'")

    while not np.all(np.isinf(similarity_matrix)):
        max_index = np.argmax(similarity_matrix)
        max_row, max_col = np.unravel_index(max_index, similarity_matrix.shape)
        max_value = similarity_matrix[max_row, max_col]

        if max_value > threshold:
            prev_id_to_next_id[list(lost_tracks_prev.keys())[max_row]] = list(
                lost_tracks_next.keys()
            )[max_col]
        else:
            return prev_id_to_next_id
        similarity_matrix[max_row, :] = -np.inf
        similarity_matrix[:, max_col] = -np.inf

    return prev_id_to_next_id


def augment_and_resize_frame(frame, augment: bool = False):
    if augment:
        gamma = 0.5
        frame = cv2.pow(frame / 255.0, gamma)
        frame = (frame * 255).astype("uint8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.8  # Increase saturation
        hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180  # Increase hue by 10 degrees
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    frame = cv2.resize(frame, (640, 640))

    return frame


def adam_tracking_v2(
    model_path: str,
    input_video_path: str,
    output_video_path: str,
    augment: bool = False,
    display_distances: bool = True,
    verbose: bool = True,
    resolve_metric: str = "cosine",
):

    print(f"Beginning tracking on {input_video_path}")

    # Defining distance calculation object
    distance_calculator = distance_calculation()

    # Loading object detection model
    model = YOLO(model_path)

    # Reading input video
    input_video = cv2.VideoCapture(input_video_path)

    # Preparing output video parameters
    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initializing dictionaries
    prev_frame_predictions = {}
    distances = {}

    for frame_number in range(1, total_frames):

        input_video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        _, original_next_frame = input_video.read()

        next_frame = augment_and_resize_frame(original_next_frame, augment=augment)

        results = model.track(
            source=next_frame,
            persist=True,
            tracker="botsort.yaml",
            mode="track",
            conf=0.5,
            verbose=False,
            max_det=32,
        )

        next_frame_predictions = dict(
            zip(
                results[0].boxes.id.int().tolist(), results[0].boxes.xyxy.int().tolist()
            )
        )

        next_frame_predictions = {
            k: next_frame_predictions[k] for k in sorted(next_frame_predictions)
        }

        lost_tracks_next = {
            id: coords for id, coords in next_frame_predictions.items() if id > 32
        }
        lost_tracks_prev = {
            id: coords
            for id, coords in prev_frame_predictions.items()
            if id not in lost_tracks_next.keys()
        }

        if lost_tracks_next and lost_tracks_prev:
            prev_id_to_next_id = resolve_lost_tracks(
                lost_tracks_prev, lost_tracks_next, metric=resolve_metric
            )
            for prev_id, next_id in prev_id_to_next_id.items():
                next_frame_predictions[prev_id] = next_frame_predictions.pop(next_id)

        for id in list(
            set(next_frame_predictions.keys()).intersection(
                set(prev_frame_predictions.keys())
            )
        ):
            if id not in distances:
                distances[id] = 0
            else:
                distances[id] += distance_calculator.cal_distance(
                    prev_frame_predictions[id], next_frame_predictions[id]
                )

        # Write frame to video
        for id, box in next_frame_predictions.items():
            x_min, y_min, x_max, y_max = convert_coords(box)
            label = str(id)
            cv2.rectangle(
                original_next_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
            )
            cv2.putText(
                original_next_frame,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            if display_distances:
                if id in distances:
                    cv2.putText(
                        original_next_frame,
                        str(round(distances[id])),
                        ((x_min + x_max) // 2, (y_min + y_max) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )
        output_video.write(original_next_frame)
        prev_frame_predictions = next_frame_predictions

        if verbose:
            print("Processed frame", frame_number, "of", total_frames)

    print(f"Finished tracking. Output video written to {output_video_path}")
    output_video.release()


if __name__ == "__main__":

    adam_tracking_v2(
        model_path="./sheep_tracking/models/best.pt",
        input_video_path="./sheep_tracking/input_videos/natural/sheeps_30sec.mp4",
        output_video_path="./sheep_tracking/output_videos/test_no_distances.mp4",
        augment=True,
        display_distances=False,
        verbose=True,
        resolve_metric="manhattan",
    )
