import os
import glob

def convert_xywhn_to_xyxy(file_path, img_width=2144, img_height=868):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            # Convert the bounding box coordinates from xywhn to xyxy
            x_center, y_center, width, height = map(float, parts[1:5])
            x1 = (x_center - width / 2) * img_width
            y1 = (y_center + height / 2) * img_height
            x2 = (x_center + width / 2) * img_width
            y2 = (y_center - height / 2) * img_height
            # Write the new line to the file
            f.write(f"{parts[0]} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {' '.join(parts[5:])}\n")

# Get a list of all txt files in the result directory
txt_files = glob.glob('./sheep_tracking/output_videos/output_files/evaluation_5/*.txt')

# Convert the bounding boxes in all txt files
for txt_file in txt_files:
    convert_xywhn_to_xyxy(txt_file)