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

    # Loop through the data to extract bounding box centers
    for frame in data:
        frame_id = frame['frame_id']
        for detection in frame['detections']:
            bbox = detection['bbox']
            try:
                track_id = detection['track_id'][0]  # track_id is a list with a single element
            except:
                track_id = np.NaN
            # Calculate center of the bounding box
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            x_coords.append(center_x)
            y_coords.append(center_y)
            frame_ids.append(frame_id)
            track_ids.append(track_id)

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
    return np.array(frame_ids), np.array(track_ids), np.array(x_coords), np.array(y_coords)


# %% VideoPlotter Tool
"""Creates a figure showing the track coordinates on a left subplot and the
corresponding video frame on a right subplot. Interactive (seeking).
Should be updated to give subplots as input instead of creating inside."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider

class VideoPlotterTracks:
    """Class for tool to plot the tracking data on top of the video frames.

    Args:
        track_ids (list): List of track IDs.
        positions (np.array): Array of shape (n, 4) where n is the number of detections.
            Each row contains the frame ID, x-coordinate, y-coordinate, and track ID.
        x_coords (list): List of x-coordinates of bounding box centers.
        y_coords (list): List of y-coordinates of bounding box centers.
        video_path (str): Path to the video file.
        
    Methods:
        setup_figure: Setup the figure and subplots.
        setup_slider: Setup the slider for frame seeking.
        update: Update the plot and video frame based on the slider value.
        update_frame: Update the video frame based on the frame index.
        on_click: Event handler for mouse clicks on the plot.
        connect_events: Connect the event handlers to the figure.
        show: Show the figure.
    """
    def __init__(self, frame_ids, track_ids, x_coords, y_coords, video_path):
        self.frame_ids = frame_ids
        self.track_ids = track_ids
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_id = video_path.split('/')[-1]
        
        self.setup_figure()
        self.setup_slider()
        self.connect_events()
        
        # self.update(0)
        # plt.show()
        # self.cap.release()

    def setup_figure(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.gs = gridspec.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[1, 1])
        plt.subplots_adjust(left=0.1, bottom=0.25, wspace=0.4, hspace=0.5)

        self.ax1 = self.fig.add_subplot(self.gs[0, 0])
        self.ax2 = self.fig.add_subplot(self.gs[1, 0], sharex=self.ax1)

        # Set title
        self.fig.suptitle(f'Tracking Data for Video: {self.video_id}')

        colors = plt.cm.jet(np.linspace(0, 1, len(np.unique(self.track_ids))))
        for idx, track_id in enumerate(np.unique(self.track_ids)):
            mask = self.track_ids == track_id

            track_frames = self.frame_ids[mask]
            track_x = self.x_coords[mask]
            track_y = self.y_coords[mask]
        
            self.ax1.plot(track_frames, track_x, marker='o', linestyle='', color=colors[idx], label=f'Track ID {track_id}')
            self.ax2.plot(track_frames, track_y, marker='o', linestyle='', color=colors[idx], label=f'Track ID {track_id}')

        self.ax1.set_xlabel('Frame Count')
        self.ax1.set_ylabel('X [px]')
        self.ax1.set_title('X-Coordinate of Bbox Centers')
        self.line1, = self.ax1.plot([0, 0], [min(self.x_coords), max(self.x_coords)], 'r-', lw=2)

        self.ax2.set_xlabel('Frame Count')
        self.ax2.set_ylabel('Y [px]')
        self.ax2.set_title('Y-Coordinate of Bbox Centers')
        self.line2, = self.ax2.plot([0, 0], [min(self.y_coords), max(self.y_coords)], 'r-', lw=2)

        handles, labels = self.ax1.get_legend_handles_labels()
        self.legend = self.fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.45, 0.5), fontsize='small')
        self.ax2.legend().set_visible(False)

        self.ax3 = self.fig.add_subplot(self.gs[:, 1])
        self.ax3.axis('off')
        self.im = self.ax3.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

        # Draw xy-coordinate frame at left top corner of video frame
        # Draw x-axis
        plt.arrow(0, 0, 50, 0, head_width=10, head_length=10, fc='red', ec='red')
        self.ax3.text(60, 0, 'x', fontsize=12, color='red')
        # Draw y-axis
        plt.arrow(0, 0, 0, 50, head_width=10, head_length=10, fc='red', ec='red')
        self.ax3.text(5, 60, 'y', fontsize=12, color='red')

    def setup_slider(self):
        axcolor = 'lightgoldenrodyellow'
        self.axframe = plt.axes([0.6, 0.1, 0.35, 0.03], facecolor=axcolor)
        self.slider = Slider(self.axframe, 'Frame', 0, self.frame_count - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update)

    def update(self, val):
        frame_idx = int(self.slider.val)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            self.im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        self.line1.set_xdata([frame_idx, frame_idx])
        self.line2.set_xdata([frame_idx, frame_idx])
        
        xlim = self.ax1.get_xlim()
        ylim1 = self.ax1.get_ylim()
        ylim2 = self.ax2.get_ylim()
        x_range = xlim[1] - xlim[0]
        
        if not xlim[0] < frame_idx < xlim[1]:
            self.ax1.set_xlim([frame_idx - x_range / 2, frame_idx + x_range / 2])
            
        self.fig.canvas.draw_idle()

    def update_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.line1.set_xdata([frame_idx, frame_idx])
        self.line2.set_xdata([frame_idx, frame_idx])
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes in [self.ax1, self.ax2]:
            frame_idx = int(event.xdata)
            self.update_frame(frame_idx)
            self.slider.set_val(frame_idx)

    def connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def show(self):
        self.update(0)
        plt.draw()
        plt.show()
        # self.cap.release()

# Example usage:
# plotter = VideoPlotter(frame_ids, track_ids, x_coords, y_coords, frame_count, video_path)

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

            threshold = max(threshold_factor * current_std, threshold_abs)

            if np.abs(signal[i] - filtered_signal[i-1]) > threshold:
                mask_removed[i] = True
                start_idx = i
                while i < len(signal) and np.abs(signal[i] - filtered_signal[start_idx-1]) > threshold:
                    mask_removed[i] = True
                    i += 1
                implausible_sequences.append((start_idx, i - start_idx))
                if i < len(signal):
                    if i - start_idx > 1:
                        filtered_signal[start_idx:i] = np.interp(
                            range(start_idx, i), [start_idx-1, i], [filtered_signal[start_idx-1], signal[i]]
                        )
                    else:
                        filtered_signal[start_idx:i] = filtered_signal[start_idx-1]
                if i < len(signal):
                    valid_window = filtered_signal[max(0, i - window_size + 1):i + 1]
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