import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.spatial.distance import cdist
from tqdm import tqdm
from scipy.signal import butter, filtfilt

""" This script contains functions to load and analyze cyclist tracking data (YOLO output) from a JSON file.

To-do:
- Set minimum changeover duration to avoid early end detection caused by tracking problems. Requires link to video (convert frames to time via FPS).


 """

def load_tracking_data(json_path):
    """
    Returns:
        array-like: frame_ids
        array-like: track_ids
        array-like: x_coords
        array-like: y_coords
        array-like: widths
        array-like: heights
    """
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

def zero_phase_filter(data, wn=0.1, order=2, debug=False):
    """ Apply a zero-phase Butterworth filter to the input data. Forward-backward filtering is used to avoid phase shift, hence only works
    for offline processing and not in real-time applications.
    
    - Debug plot shows the original and filtered data, as well as interpolated values in case of NaN."""
    # Handle NaN values
    data = np.asarray(data).flatten()
    # Interpolate NaN values
    nans, x = np.isnan(data), lambda z: z.nonzero()[0]
    print(f"nans: {nans}, x: {x}")
    data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    # Warning if NaN values are present
    if np.sum(np.isnan(data)) > 0:
        print("Warning: NaN values are present in the data. Interpolated between adjacent values.")

    # Apply zero-phase Butterworth filter
    b, a = butter(order, wn, btype='low')
    y = filtfilt(b, a, data)

    if debug:
        plt.figure(figsize=(14, 8))
        plt.plot(data, label='Original Data', color='blue')
        plt.plot(y, label='Filtered Data', color='green')
        plt.plot(np.arange(len(data))[nans], data[nans], 'rx', label='Interpolated Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Signal Value')
        plt.title('Zero-phase Butterworth Filter')
        plt.legend()
        plt.show()

    return y

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


def detect_changeovers(distance, start_thd=20, start_confirm_frames=25, confirm_thd=150, end_thd=20, end_confirm_frames=5, video_fps=50, duration_min=2,
                       duration_fixed=None):
    """
    Detect changeovers in the distance signal. Distance signal must be filtered such that the changeover pattern is not 
    influenced by YOLO misdetections or tracking errors.

    TODO: Add option to set minimum or fixed duration of changeover to avoid early end detection caused by tracking problems.

    Parameters:
    distance (array-like): Distance signal to detect changeovers.
    start_thd (float): Threshold for detecting start of changeover.
    start_confirm_frames (int): Number of frames to confirm start of changeover.
    confirm_thd (float): Threshold for confirming start of changeover.
    end_thd (float): Threshold for detecting end of changeover.
    end_confirm_frames (int): Number of frames to confirm end of changeover.
    video_fps (int): Frames per second of the video.
    duration_min (int) [s]: Minimum duration of a changeover [s].
    duration_fixed (int) [s]: Fixed duration of a changeover [s]. Overrides duration_min. If None, duration is computed based on descending distance.

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
    """Restructure changeover data to include start and end times."""

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

def normalize_rider_movement(y_dist, frame_ids_unique, h_bboxes, frame_ids, h_rider=1, debug=False):
    """Normalizes the rider movement (height, y-coordinate) w.r.t. bounding box height.
    Physical rider height is assumed to be constant and equal for all riders, and is as such used as a reference.

    Parameters:
        y_dist (array-like): Distance signal to detect changeovers. Indices in frame_ids_unique.
        frame_ids_unique (array-like): Unique frame IDs.
        h_bboxes (array-like): Heights of bounding boxes for each data point. Indices correspond to frame_ids.
        frame_ids (array-like): Frame IDs for each data point.

    Next steps:
        - Input changeover times
        - Get maximum height per changeover
        - Normalize y_dist per changeover 
    """
    # Filter bbox height
    # h_bboxes = zero_phase_filter(h_bboxes, wn=0.1, order=2, debug=True)
    
    # Get max bbox height per frame
    h_bboxes_max_per_frame = []
    # print(frame_ids)
    for frame_id in frame_ids_unique:
        idx = np.where(frame_ids == frame_id)[0]
        if len(idx) > 0:
            h_bboxes_max_per_frame.append(np.max(h_bboxes[idx]))
        else:
            h_bboxes_max_per_frame.append(np.nan)

    h_bboxes_max_per_frame_filt = zero_phase_filter(h_bboxes_max_per_frame, wn=0.1, order=2, debug=True)

    # Normalize y_dist. Avoid division by zero
    # y_dist_norm = np.zeros_like(y_dist)
    # for i, frame_id in enumerate(frame_ids_unique):
        # y_dist_norm[i] = y_dist[i] / h_bboxes_max_per_frame_filt[i]
    y_dist_norm = y_dist / h_bboxes_max_per_frame_filt

    print(f"y_dist_norm: {y_dist_norm}")

    # Plot y_dist, height and normalized y_dist
    if debug:
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Bbox Y-distance and Bbox Height')

        gs = fig.add_gridspec(2,2)

        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
        ax3 = fig.add_subplot(gs[:,1])

        ax1.plot(frame_ids_unique, y_dist, label='Distance', color='blue')
        ax1.set_ylim([0, 400])
        
        # Create a secondary y-axis for the normalized distance
        ax12 = ax1.twinx()
        ax12.plot(frame_ids_unique, y_dist_norm, label='Normalized Distance', color='red')
        ax12.set_ylim([0, 4])
        ax12.set_ylabel('Normalized Distance')

        ax2.set_ylabel('Distance (px)')
        ax2.plot(frame_ids, h_bboxes, label='Bounding Box Height', color='green')
        ax2.plot(frame_ids_unique, h_bboxes_max_per_frame_filt, label='Max Bounding Box Height', color='orange')
        ax2.set_ylabel('Bounding Box Height (px)')
        ax2.set_xlabel('Frame Index')

        plt.legend()
        
        from plotting_tools import VideoPlotterGeneric

        # VideoPlotterGeneric(fig, [ax1, ax12, ax2], ax3, video_path=r'output\C1319-20240701_112129_yolov8m-trackcycling-03-output.mp4', x_type='frame').show()
        # Video path is hardcoded for now
        VideoPlotterGeneric(fig, [ax1, ax12, ax2], ax3, video_path=r'media\VID-PI-EL-BEL-240412-NC-Milton-TP-R1.MP4', x_type='frame').show()

