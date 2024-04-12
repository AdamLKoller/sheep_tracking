import cv2
import numpy as np


def get_subvideo(video_path, output_video_path, start_time, duration):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)

    fourcc = int(vidcap.get(cv2.CAP_PROP_FOURCC))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_number in range(start_frame, min(end_frame, total_frames)):
        ret, frame = vidcap.read()
        if ret:
            img = frame[500:1500, 256:2400]
            pts = np.array([[1100, 0], [1600, 266], [2400, 266], [2400, 0]])
            pts2 = np.array([[1175, 0], [580, 0], [880, 70]])
            pts = pts.reshape((-1, 1, 2))
            pts2 = pts2.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], (0, 0, 0))
            cv2.fillPoly(img, [pts2], (0, 0, 0))
            out.write(img)

    vidcap.release()
    out.release()

if __name__ == '__main__':
    video_path = "./sheeps1.mp4"
    output_video_path = "./sheeps_5min.mp4"
    start_time = 3600
    duration = 300

    get_subvideo(
        video_path=video_path,
        output_video_path=output_video_path,
        start_time=start_time,
        duration=duration,
    )