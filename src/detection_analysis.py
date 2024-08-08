import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider
from scipy.spatial.distance import cdist
from tqdm import tqdm
from scipy.signal import butter, filtfilt

""" This script contains functions to load and analyze tracking data from a JSON file. 
It also contains a class to plot the tracking data on top of the video frames. """

def load_tracking_data(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
    except:
        raise FileNotFoundError(f"File not found: {json_path}")  

    video_path = json_path.replace('.json', '.mp4')

    # Initialize lists to store x and y coordinates of bounding box centers
    x_coords = []
    y_coords = []
    frame_ids = []
    track_ids = []
    widths = []
    heights = []

    # Loop through the data to extract bounding box centers
    for frame in data:
        frame_id = frame['frame_id']
        for detection in frame['detections']:
            bbox = detection['bbox']
            try:
                track_id = detection['track_id'][0]  # track_id is a list with a single element
            except:
                # track_id = -1  # If track_id is not provided, skip
                continue

            # Calculate center of the bounding box
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            x_coords.append(center_x)
            y_coords.append(center_y)
            frame_ids.append(frame_id)
            track_ids.append(track_id)
            widths.append(width)
            heights.append(height)

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Sanity check on frame count
    json_frame_count = max(frame_ids)+1 # Frame IDs are zero-based
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if json_frame_count != video_frame_count:
        # Raise warning but let script continue
        print(f"Warning: Frame count mismatch: JSON file has {json_frame_count} frames, video has {video_frame_count} frames.")

    cap.release()

    # return frame_ids, track_ids, x_coords, y_coords
    return np.array(frame_ids), np.array(track_ids), np.array(x_coords), np.array(y_coords), np.array(widths), np.array(heights)


# %% Signal Processing
def remove_outliers(signal, threshold_abs, threshold_factor, window_size, debug=False):
    """
    Apply a standard deviation-based filter to remove outliers from the signal.
    
    Parameters:
    signal (array-like): Input signal to be filtered.
    threshold_factor (float): Factor to determine the outlier threshold based on running standard deviation.
    window_size (int): Window size for computing running standard deviation.
    debug (bool): Flag to indicate if debug plot should be generated.
    
    Returns:
    array-like: Filtered signal with implausible values corrected.
    array-like: Mask indicating the removed values.
    list: List of tuples indicating start and length of outlier sequences.
    """
    signal = np.asarray(signal)
    filtered_signal = signal.copy()
    implausible_sequences = []
    mask_removed = np.zeros_like(signal, dtype=bool)

    # Initialize running statistics
    running_mean = []
    running_std = []

    prev_valid_idx = 0

    pbar = tqdm(total=len(signal), desc='Processing signal')

    for i in range(len(signal)):
        if i == 0:
            running_mean.append(signal[i])
            running_std.append(0)
        else:
            # Use a window of last validated samples
            valid_window = filtered_signal[max(0, prev_valid_idx - window_size):prev_valid_idx]

            current_mean = np.mean(valid_window)
            current_std = np.std(valid_window)

            running_mean.append(current_mean)
            running_std.append(current_std)

            threshold = max(threshold_factor * current_std, threshold_abs) # Threshold to remove outliers

            if np.abs(signal[i] - filtered_signal[i-1]) > threshold:
                # Outlier detected
                mask_removed[i] = True
                # if i < len(signal) and np.abs(signal[i] - filtered_signal[start_idx-1]) > threshold:
                #     # As long as the signal is above the threshold, remove the value
                #     mask_removed[i] = True
                #     i += 1
                # implausible_sequences.append((start_idx, i - start_idx)) 
                # MAYBE ADD MAX LENGTH OF IMPLAUSIBLE SEQUENCE
                # Predict the missing values based on polyfit using last 10 samples
                x = np.arange(i-11, i)

                # Fit a polynomial to the last 10 samples before start_idx
                p = np.polyfit(np.arange(i-11, i-1), filtered_signal[i - 11:i-1], 1)
                filtered_signal[i] = np.polyval(p, x[-1]) # Predict the next value based on the polynomial fit

                # Debug plot for the polyfit
                # if debug:
                    # plt.plot(signal, 'o', label='Original Data')
                    # plt.plot(x, filtered_signal[i - 11:i], 'o', label='Filtered Data')
                    # plt.plot(x, np.polyval(p, x), label='Polyfit')
                    # plt.xlabel('Sample Index')
                    # plt.ylabel('Signal Value')
                    # plt.title('Polyfit for Missing Values')
                    # plt.legend()
                    # plt.show()

                # if i < len(signal):
                    
                #     if i - start_idx > 1:
                        

                #         # filtered_signal[start_idx:i] = np.interp(
                #         #     range(start_idx, i), [start_idx-1, i], [filtered_signal[start_idx-1], signal[i]]
                #         # )
                #     else:
                #         filtered_signal[start_idx:i] = filtered_signal[start_idx-1]
                if i < len(signal):
                    valid_window = filtered_signal[max(0, i - window_size + 1):i]
                    running_mean[-1] = np.mean(valid_window)
                    running_std[-1] = np.std(valid_window)
            else:
                prev_valid_idx = i

        pbar.update(1)

    if debug:
        # Plot the signal and the removed values
        plt.figure(figsize=(14, 8))
        plt.plot(signal, label='Original signal', color='blue')
        plt.plot(filtered_signal, label='Filtered signal', color='green')
        plt.plot(np.arange(len(signal))[mask_removed], signal[mask_removed], 'rx', label='Removed values')

        # Plot the threshold region
        running_mean = np.array(running_mean)
        running_std = np.array(running_std)
        # For each sample, take either std or abs thhreshold, whichever is larger
        lower_bound = filtered_signal - np.maximum(threshold_factor * running_std, threshold_abs)
        upper_bound = filtered_signal + np.maximum(threshold_factor * running_std, threshold_abs)

        plt.fill_between(np.arange(len(signal)), lower_bound, upper_bound, color='grey', alpha=0.5, label='Threshold Region')

        plt.legend()
        plt.xlabel('Sample Index')
        plt.ylabel('Signal Value')
        plt.title('Outlier Detection and Correction with Running Standard Deviation')
        plt.show()
    
    return filtered_signal, mask_removed, implausible_sequences

def compute_pairwise_distance(frame_ids, y_coords):
    dist_per_frame = []
    min_dist = []
    unique_frame_ids = np.unique(frame_ids)

    pbar = tqdm(total=len(frame_ids), desc='Processing Frames')

    # Get pairwise distances between bounding box centers for each frame
    for idx, frame_id in enumerate(np.unique(frame_ids)):
        idc = np.where(frame_ids == frame_id)[0].astype(int)
        x = np.zeros(np.size(idc)) # Set to 0, because we only need the y-coordinates
        y = y_coords[idc]
        centers = np.column_stack((np.zeros(np.size(x)), y)) # Set to 0, because we only need the y-coordinates

        x_min = 0
        y_min = np.min(y)

        # Compute distance between lowest value y-coordinate and all others
        distances = cdist(centers, [[x_min, y_min]])

        # Find the min distance between bounding box centers, excluding 0
        if np.size(distances[distances > 0]) > 0:
            min_distance = np.min(distances[distances > 0])
        else:
            min_distance = 0

        dist_per_frame.append(distances) # rows = frame count
        min_dist.append(min_distance) # rows = frame count

        pbar.update(1)

    pbar.close()

    return min_dist, unique_frame_ids

def zero_phase_filter(data, wn=0.1, order=5):
    b, a = butter(order, wn, btype='low')
    y = filtfilt(b, a, data)
    return y

def detect_changeovers(distance, start_thd=20, start_confirm_frames=25, confirm_thd=150, end_thd=20, end_confirm_frames=5):
    """
    Detect changeovers in the distance signal. Distance signal must be filtered such that the changeover pattern is not 
    influence by YOLO misdetections or tracking errors.

    Parameters:
    distance (array-like): Distance signal to detect changeovers.
    start_thd (float): Threshold for detecting start of changeover.
    start_confirm_frames (int): Number of frames to confirm start of changeover.
    confirm_thd (float): Threshold for confirming start of changeover.
    end_thd (float): Threshold for detecting end of changeover.
    end_confirm_frames (int): Number of frames to confirm end of changeover.

    Returns:
    list: List of tuples indicating start and end of changeover sequences.
    """
    distance = np.asarray(distance)
    changeovers = []
    
    in_changeover = False
    start_counter = 0
    end_counter = 0
    

    for i in range(len(distance)):
        if not in_changeover:
            if distance[i] > start_thd and distance[i] > distance[i-1]: # Check if distance is increasing
                start_counter += 1
                if start_counter >= start_confirm_frames and distance[i] > confirm_thd: # Distance monotonically increasing
                    in_changeover = True
                    changeovers.append(('start', i - start_confirm_frames + 1))
                    start_counter = 0
            else:
                start_counter = 0
        else:
            if distance[i] < end_thd:
                end_counter += 1
                if end_counter >= end_confirm_frames:
                    in_changeover = False
                    changeovers.append(('end', i - end_confirm_frames + 1))
                    end_counter = 0
            else:
                end_counter = 0
    
    return changeovers

# Append times to changeovers
def restructure_changeovers(changeovers, changeover_times):
    if len(changeovers) != len(changeover_times):
        raise ValueError("Length of changeovers and changeover_times must be equal.")
    
    out = []
    
    for i in range(0, len(changeovers), 2):
        if i + 1 < len(changeovers):
            
            start = changeovers[i]
            end = changeovers[i + 1]
            if start[0] == 'start' and end[0] == 'end':
                changeover_number = (i // 2) + 1
                result = {'changeover_number': changeover_number,
                          'data': []}

                changeover = {
                    'start_frame': start[1],
                    'end_frame': end[1],
                    'duration': end[1] - start[1] + 1,
                    'start_time': changeover_times[i],
                    'end_time': changeover_times[i + 1]
                }
                result['data'] = changeover

            out.append(result)

    return out

def normalize_rider_movement(y_distance):
    """Normalizes the rider movement (height, y-coordinate) w.r.t. bounding box height.
    Physical rider height is assumed to be constant and equal for all riders, and is as such used as a reference.
    """
