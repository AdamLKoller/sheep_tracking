from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
import cv2

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