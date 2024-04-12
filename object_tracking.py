from ultralytics import YOLO
import cv2
import torch
import numpy as np
from sklearn.metrics import pairwise_distances

from augment_frame import apply_augmentation, apply_contrast_change
from distance_calculation import distance_calculation


def pad_image(image, target_shape):
    """
    Pad the image with black pixels to match the target shape.

    Parameters:
        image (numpy.ndarray): Input image array.
        target_shape (tuple): Target shape (height, width) to resize the image to.

    Returns:
        numpy.ndarray: Padded image array.
    """
    h, w = image.shape[:2]
    pad_height = target_shape[0] - h
    pad_width = target_shape[1] - w
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded_image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return padded_image


def track(
    model_path,
    input_video_path,
    output_video_path,
    pjt=np.inf,
    apply_augment: bool = False,
):
    """
    Apply object tracking to input video using custom minimum pairwise joining algorithim

    Parameters:
        model_path (str): Path to Yolov8 model .pt file
        input_video_path (str): Path to input video
        output_video_path (str): Path to output video
        pjt (float64): Pairwise joining threshold, set to lower value for more strict pairing
        apply_augment (bool): set to True to augment frames before model prediction
    """

    model = YOLO(model_path)
    distance_calculator = distance_calculation(px=1280, py=664)

    # Preparing output video parameters
    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracked_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Reading initial frame
    first_frame = vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, initial_frame = vidcap.read()

    # Augmenting initial frame
    augmented_intial_frame = (
        apply_contrast_change(apply_augmentation(initial_frame))
        if apply_augment
        else initial_frame
    )

    # Predicting initial frame
    intitial_frame_predictions = model.predict(
        augmented_intial_frame, max_det=32, conf=0.5
    )
    initial_frame_box_coords = (
        [torch.round(sheep.boxes.xyxy) for sheep in intitial_frame_predictions][0]
        .int()
        .tolist()
    )
    initial_frame_sheep_images = [
        augmented_intial_frame[y_b:y_t, x_l:x_r]
        for x_l, y_b, x_r, y_t in initial_frame_box_coords
    ]

    # Initializing last seen dictionary. Key is label. Value is the corresponding image the last time that label was seen.
    id_to_lastseen_img = {}
    for id, image in enumerate(initial_frame_sheep_images):
        id_to_lastseen_img[id] = {
            "image": image,
            "bounding-box coords": initial_frame_box_coords[id],
            "distance traveled": 0,
            "lagging distance": 0,
        }

    # Iterate through the frames of the input video
    for frame_number in range(1, total_frames // 4):

        # Read frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, next_frame = vidcap.read()

        if frame_number % 1 == 0:

            # Augment frame
            augmented_next_frame = (
                apply_contrast_change(apply_augmentation(next_frame))
                if apply_augment
                else next_frame
            )

            # Predict next frame
            next_frame_predictions = model.predict(
                augmented_next_frame, max_det=32, conf=0.5, verbose=False
            )

            # Get next frame images
            coords = (
                [torch.round(sheep.boxes.xyxy) for sheep in next_frame_predictions][0]
                .int()
                .tolist()
            )
            next_frame_sheep_images = [
                augmented_next_frame[y_b:y_t, x_l:x_r] for x_l, y_b, x_r, y_t in coords
            ]

            # Last seen images as a list
            last_seen_sheep_images = []
            for id in id_to_lastseen_img.keys():
                last_seen_sheep_images.append(id_to_lastseen_img[id]["image"])

            # Padding last_seen and next frame images
            max_height = max(
                [
                    image.shape[0]
                    for image in next_frame_sheep_images + last_seen_sheep_images
                ]
            )
            max_width = max(
                [
                    image.shape[1]
                    for image in next_frame_sheep_images + last_seen_sheep_images
                ]
            )

            padded_last_seen = [
                pad_image(image, (max_height, max_width))
                for image in last_seen_sheep_images
            ]
            padded_next_frame = [
                pad_image(image, (max_height, max_width))
                for image in next_frame_sheep_images
            ]

            # Calculating img distances between each of the last seen images and the next frame iamges
            flat_last_seen = [frame.flatten() for frame in padded_last_seen]
            flat_next_frame = [frame.flatten() for frame in padded_next_frame]

            distance_matrix = pairwise_distances(
                flat_last_seen, flat_next_frame, metric="manhattan"
            )

            # Reassign IDs
            count_resassigned = 0

            while not np.all(np.isinf(distance_matrix)):
                min_index = np.argmin(distance_matrix)
                min_row, min_col = np.unravel_index(min_index, distance_matrix.shape)
                min_value = distance_matrix[min_row][min_col]
                if min_value < pjt:
                    # Calculate "world" distance
                    last_x_min, last_y_min, last_x_max, last_y_max = id_to_lastseen_img[
                        min_row
                    ]["bounding-box coords"]
                    next_x_min, next_y_min, next_x_max, next_y_max = coords[min_col]

                    # print([last_x_min, last_y_max, last_x_max, last_y_min])
                    # print([next_x_min, next_y_max, next_x_max, next_y_min])

                    world_distance = distance_calculator.cal_distance(
                        [last_x_min, last_y_max, last_x_max, last_y_min],
                        [next_x_min, next_y_max, next_x_max, next_y_min],
                    )
                    # print(world_distance)
                    id_to_lastseen_img[min_row]["distance traveled"] += world_distance
                    # Assign new "last_seen" image to id
                    id_to_lastseen_img[min_row]["image"] = next_frame_sheep_images[
                        min_col
                    ]
                    # Assign new "last_seen" coords to id
                    id_to_lastseen_img[min_row]["bounding-box coords"] = coords[min_col]

                    count_resassigned += 1
                distance_matrix[min_row, :] = np.inf
                distance_matrix[:, min_col] = np.inf
                pjt += 100000
            print(
                f"{len(next_frame_sheep_images)} detected in frame {frame_number} \t {count_resassigned} given new labels"
            )

        if frame_number % 10 == 0:
            for id in id_to_lastseen_img.keys():
                id_to_lastseen_img[id]["lagging distance"] = id_to_lastseen_img[id][
                    "distance traveled"
                ]

        # Write frame to video with bounding boxes and labels
        for id in id_to_lastseen_img.keys():
            box = id_to_lastseen_img[id]["bounding-box coords"]
            x_min, y_min, x_max, y_max = box
            label = str(id)
            cv2.rectangle(next_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                next_frame,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                next_frame,
                str(round(id_to_lastseen_img[id]["lagging distance"])),
                ((x_min + x_max) // 2, (y_min + y_max) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        tracked_video.write(next_frame)

    tracked_video.release()


if __name__ == "__main__":
    track(
        model_path="./sheep_tracking/models/best.pt",
        input_video_path="./sheep_tracking/input_videos/evaluation_1.mp4",
        output_video_path="./sheep_tracking/output_videos/evaluation_1_tracked.mp4",
    )
