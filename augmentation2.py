import cv2
import numpy as np
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

def apply_augmentation(frame):
    # # Change brightness
    # frame = cv2.convertScaleAbs(frame, alpha=1, beta=10)  # Increase brightness

    # Change exposure (gamma correction)
    gamma = 0.5
    frame = cv2.pow(frame / 255.0, gamma)
    frame = (frame * 255).astype('uint8')

    # # Convert to YUV color space
    # yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # # Apply CLAHE to Y channel (luminance)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])

    # # Convert back to BGR
    # frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Change saturation and hue
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.8  # Increase saturation
    hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180  # Increase hue by 10 degrees
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame

def apply_contrast_change(frame):
    # Convert frame to float
    frame = frame.astype('float64') / 255

    # Global equalize
    frame_rescale = exposure.equalize_hist(frame)

    # Equalization
    frame_eq = exposure.equalize_adapthist(frame_rescale, clip_limit=0.03)

    # Convert frame back to uint8
    frame_eq = (frame_eq * 255).astype('uint8')

    return frame_eq

# Open video file
video_capture = cv2.VideoCapture('./sheep_tracking/input_videos/natural/evaluation_5.mp4')

# Get video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)

# # Define codec and create VideoWriter object
# output_video = cv2.VideoWriter('output_video4_noresize.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (640, 640))

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter('./sheep_tracking/input_videos/augmented/evaluation_5.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Define codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'FMP4'
# output_video = cv2.VideoWriter('output_video4.mp4', fourcc, fps, (640, 640))


# Calculate the number of frames to process for 10 seconds of video
# frames_to_process = int(10 * fps)

# for i in range(frames_to_process):
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame to 640x640
    # frame = cv2.resize(frame, (640, 640))

    # Apply augmentation
    augmented_frame = apply_augmentation(frame)

    # Apply contrast change
    contrast_changed_frame = apply_contrast_change(augmented_frame)

    # Write contrast changed frame to output video
    output_video.write(contrast_changed_frame)

    #cv2.imshow('Contrast Changed Video', contrast_changed_frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# Release resources
video_capture.release()
output_video.release()
cv2.destroyAllWindows()