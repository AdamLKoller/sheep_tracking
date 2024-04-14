import cv2
import os

# Directory containing the video file
video_file = "./sheep_tracking/input_videos/natural/evaluation_5.mp4"

# Directory containing the text files with bounding box coordinates
txt_directory = "./sheep_tracking/output_videos/output_files/evaluation_5/corrected_ids"

# Output directory for the new video with bounding boxes
output_video = "./sheep_tracking/output_videos/evaluation_5_tracked.mp4"

# Read the video file
cap = cv2.VideoCapture(video_file)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


# List files in directory
file_names = os.listdir(txt_directory)


# Extract numerical indices from file names
indices = [int(file.split("_")[0]) for file in file_names]

# Combine file names and indices into tuples
file_indices = list(zip(file_names, indices))

# Sort tuples based on indices
sorted_files_indices = sorted(file_indices, key=lambda x: x[1])

# Extract sorted file names
sorted_files = [file[0] for file in sorted_files_indices]

# Iterate through each text file
for txt_file in sorted_files:
    if txt_file.endswith(".txt"):
        frame_number = int(txt_file.split("_")[0])  # Extract frame number from file name
        frame_path = os.path.join(txt_directory, txt_file)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if video ends
        
        # Read bounding box coordinates from the text file and draw them on the frame
# Read bounding box coordinates from the text file and draw them on the frame
    with open(frame_path, 'r') as f:
        for line in f:
            if line.strip():  # Check if the line is not empty
                _, x1, y1, x2, y2, id = line.strip().split(" ")
                x1, y1, x2, y2 = map(int, [float(x1), float(y1), float(x2), float(y2)])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                font_scale = 0.5
                cv2.putText(frame, id, (x1+2, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

        
        # Write the frame with bounding boxes to the output video
        out.write(frame)

# Release video capture and writer
cap.release()
out.release()

print("Bounding boxes drawn and saved to", output_video)