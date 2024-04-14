import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import re

# Directory containing the files
ori_directory = "./sheep_tracking/output_videos/output_files/evaluation_5/original_ids"
reid_dir = "./sheep_tracking/output_videos/output_files/evaluation_5/corrected_ids"

files = [f for f in os.listdir(ori_directory)]# if f.startswith("output_video4_") and f.endswith(".txt")]
title = ["label", "xtop", "ytop", "xbot", "ybot", "id"]

def map_id_cos_sim(cos_sim_matrix, lost_track1, lost_track2):
    unmapped_rows = set(range(len(lost_track1)))  # Initialize with all rows
    id_mapping_dict_cos_sim = {}

    while not np.all(np.isinf(cos_sim_matrix)):
        # print("\n",i)
        # print(cos_sim)
        # print("max cos sim" ,np.argmax(cos_sim))
        max_index = np.argmax(cos_sim_matrix)
        max_row, max_col = np.unravel_index(max_index, cos_sim_matrix.shape)
        id_mapping_dict_cos_sim[lost_track2.id.iloc[max_col]] = lost_track1.id.iloc[max_row]
        unmapped_rows.discard(max_row)  # Remove the mapped row index
        print("cos sim",lost_track1.id.iloc[max_row], lost_track2.id.iloc[max_col])
        cos_sim_matrix.iloc[max_row, :] = -np.inf
        cos_sim_matrix.iloc[:, max_col] = -np.inf
    # when lost_track1 > lost_track2, that means there are obj(s) missing in frame2
    # return the object from frame1 that wasn't matched to frame2, to add that missing obj to the frame2 detections

    # print(type(id_mapping_dict_cos_sim.values()), id_mapping_dict_cos_sim.values(), list(id_mapping_dict_cos_sim.values()) )
    # missing_detections = lost_track1.id[~list(id_mapping_dict_cos_sim.values())]
    # return id_mapping_dict_cos_sim, missing_detections

    # Create a list of unmapped rows
    unmapped_rows = lost_track1[~lost_track1['id'].isin(id_mapping_dict_cos_sim.values())]
    # Convert the list to a DataFrame
    unmapped_df = pd.DataFrame(unmapped_rows) 
    # print("abcd", id_mapping_dict_cos_sim)   
    return id_mapping_dict_cos_sim, unmapped_df

def map_id_manhattan(distance_matrix, lost_track1, lost_track2):
    while not np.all(np.isinf(distance_matrix)):
        # print(distance_matrix)
        # print("min manhattan", np.argmin(distance_matrix))
        min_index = np.argmin(distance_matrix)
        min_row, min_col = np.unravel_index(min_index, distance_matrix.shape)
        id_mapping_manhattan[lost_track2.id.iloc[min_col]] = lost_track1.id.iloc[min_row]
        print("manhattan",lost_track1.id.iloc[min_row], lost_track2.id.iloc[min_col])
        distance_matrix[min_row, :] = np.inf
        distance_matrix[:, min_col] = np.inf

# Iterate over consecutive pairs of files
print(len(files))
for i in range(1, len(files)):
# for i in range(1, 192):

    prev_file_path = f"{ori_directory}/evaluation_5_{i}.txt"
    next_file_path = f"{ori_directory}/evaluation_5_{i+1}.txt"

    print(prev_file_path)

    # prev_file_path = os.path.join(directory, prev_file)
    # next_file_path = os.path.join(directory, next_file)

    print("Mapping IDs between", prev_file_path, "and", next_file_path)

    # Read data from the files
    df_prev = pd.read_csv(prev_file_path, header=None, names=title, sep=" ")

    df_next = pd.read_csv(next_file_path, header=None, names=title, sep=" ")

    print(df_prev)
    print(df_next)
    # Identify lost tracks in the next file
    lost_tracks_next = df_next[df_next['id'] > 32]

    # Identify lost tracks in the previous file
    ids_in_next = df_prev['id'].isin(df_next['id'])
    lost_tracks_prev = df_prev[~ids_in_next]

    # print("ids_in_next", ids_in_next)

    print("1voi", lost_tracks_prev)
    print("2cho", lost_tracks_next)
    if not lost_tracks_prev.empty:
        if not lost_tracks_next.empty:
        # Compute mappings between lost tracks
            cos_sim_matrix = pd.DataFrame(cosine_similarity(lost_tracks_prev.iloc[:, 1:-1], lost_tracks_next.iloc[:, 1:-1]))
            # distance_matrix = pairwise_distances(lost_tracks_prev.iloc[:, 1:-1], lost_tracks_next.iloc[:, 1:-1], metric='manhattan')

            # id_mapping_cos_sim = {}
            # id_mapping_manhattan = {}

            # map_id_cos_sim(cos_sim_matrix, id_mapping_cos_sim)
            # map_id_manhattan(distance_matrix, id_mapping_manhattan)
            
            id_mapping_dict_cos_sim, missing_detections = map_id_cos_sim(cos_sim_matrix, lost_tracks_prev, lost_tracks_next)
            # map_id_manhattan(distance_matrix, lost_tracks_prev, lost_tracks_next)
            print("id_dict", type(id_mapping_dict_cos_sim), id_mapping_dict_cos_sim)
            for value in id_mapping_dict_cos_sim.values():
                if not isinstance(value, int):
                    break
            # Apply mappings to next file
            # extra incorrect detections (ie, 33 detections instead of 32) that are not mapped will have ID as NaN
            missing_ids = lost_tracks_next.id.unique()
            df_next.loc[df_next['id'].isin(missing_ids), 'id'] = df_next['id'].map(id_mapping_dict_cos_sim)

            df_next = pd.concat([df_next, missing_detections])
            # drop the extra incorrect detections
            df_next.dropna(inplace=True)

            # Convert ID column to integer type
            df_next['id'] = df_next['id'].astype(int)




            # df_next.loc['id'] = df_next['id'].map(id_mapping_cos_sim)
            # df_next['id'] = df_next['id'].map(id_mapping_manhattan)
            print("missing", missing_detections)

            print("finale", df_next)
        elif lost_tracks_next.empty: # next_frame does detects less than 32 objects
            df_next = pd.concat([df_next, lost_tracks_prev])
        
        if df_next.id.isnull().sum():
            break

    # Write to new file
    output_file = f"{reid_dir}/{i+1}_reID.txt"
    print(output_file)
    with open(output_file, 'w') as f:
        f.write(df_next.to_csv(index=False, sep=" ", header=False))
    # print("conmeo")
print("All files processed.")