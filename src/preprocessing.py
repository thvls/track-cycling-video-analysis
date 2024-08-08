# preprocessing.py
"""Preprocessing functions for tracking data.
Run this script to test or debug the preprocessing steps used in the main tool."""

import numpy as np
import detection_analysis as da
import matplotlib.pyplot as plt

def plot_data(frame_ids, track_ids, x_coords, y_coords):
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('Track Data')

    gs = fig.add_gridspec(2, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax1.plot(frame_ids, x_coords, marker='.', linestyle='', label='Original')
    ax2.plot(frame_ids, y_coords, marker='.', linestyle='', label='Original')

    plot_track_numbers(frame_ids, track_ids, x_coords, ax1)
    plot_track_numbers(frame_ids, track_ids, y_coords, ax2)

    ax1.set_ylabel('X Coordinate')
    ax2.set_xlabel('Frame ID')
    ax2.set_ylabel('Y Coordinate')

    plt.legend()
    plt.grid(True)
    plt.show()

    return fig, ax1, ax2

def remove_nan_tracks(frame_ids, track_ids, x_coords, y_coords):
    """Remove tracks with NaN values in x or y coordinates."""
    mask = np.isnan(track_ids)
    frames_removed = frame_ids[mask]
    print(f"Removed {np.sum(mask)} tracks with NaN values.")
    print(f"Frames removed: {frames_removed}")

    return frame_ids[~mask], track_ids[~mask], x_coords[~mask], y_coords[~mask]

def compute_stdevs(frame_ids, x_coords, y_coords):
    """Computes the standard deviation of x and y coordinates for each frame.
    Output is an array of stdevs for each frame."""

    stdevs_x = []
    stdevs_y = []
    stdevs_frames = np.unique(frame_ids)

    for frame in stdevs_frames:
        mask = frame_ids == frame
        x_coords_frame = x_coords[mask]
        y_coords_frame = y_coords[mask]

        stdev_x = np.std(x_coords_frame)
        stdev_y = np.std(y_coords_frame)

        stdevs_x.append(stdev_x)
        stdevs_y.append(stdev_y)

    return stdevs_x, stdevs_y, stdevs_frames

def get_last_track_indices(frame_ids, track_ids):
    """Get the last index of each track ID in the frame_ids array.
    
    Args:
    frame_ids (np.array): Array of frame IDs.
    track_ids (np.array): Array of track IDs.
    
    Returns:
    last_indices (list): List of the last indices of each track ID. Size is equal to the number of unique track IDs."""
    unique_tracks = np.unique(track_ids)
    last_indices = []
    last_frames = []

    for track_id in unique_tracks:
        track_indices = np.where(track_ids == track_id)[0]
        last_index = track_indices[-1]
        last_indices.append(last_index)
        last_frames.append(frame_ids[last_index])

    return last_indices, last_frames

def get_first_track_indices(frame_ids, track_ids):
    """Get the first index of each track ID in the frame_ids array.
    
    Args:
    frame_ids (np.array): Array of frame IDs.
    track_ids (np.array): Array of track IDs.
    
    Returns:
    first_indices (list): List of the first indices of each track ID. Size is equal to the number of unique track IDs."""
    unique_tracks = np.unique(track_ids)
    first_indices = []
    first_frames = []

    for track_id in unique_tracks:
        track_indices = np.where(track_ids == track_id)[0]
        first_index = track_indices[0]
        first_indices.append(first_index)
        first_frames.append(frame_ids[first_index])

    print(f"First frames: {first_frames}")

    return first_indices, first_frames

def merge_tracks(frame_ids, track_ids, x_coords, y_coords, debug=False):
    from detection_analysis import zero_phase_filter

    """Merge tracks that are close to each other.
    First assesses whether tracks are distinguishable within a frame by checking the standard deviation of the values in a frame (window).
    If this exceeds a threshold, tracks are distuinguishable and we can attempt to merge consecutive tracks.
    
    Merging happens by computing a polyfit through the last x,y coordinates of the track and comparing the distance to the next appearing tracks.
    The distance must be lower than a threshold to merge the tracks.
    """

    # TO-DO: These constants should be given as input parameters!
    POLYFIT_LEN = 10 # Length of the polyfit on last x,y coordinates of the track
    FRAME_SEARCH_WINDOW_NEXT_TRACK = 5 # Search window for the next appearing tracks to potentially merge with
    MERGE_THD_DIST_X = 100 # Distance with next track must be lower than this to merge based on x
    MERGE_THD_DIST_Y = 100
    MERGE_THD_STDEV_X = 150 # Stdev must be higher than this to merge based on x
    MERGE_THD_STDEV_Y = 150
    MIN_TRACK_LENGTH_COMPARE = 20 # Minimum next track length to potentially merge with
    
    unique_tracks = np.unique(track_ids)
    unique_frames = np.unique(frame_ids)

    stdevs_x_orig, stdevs_y_orig, stdevs_frames = compute_stdevs(frame_ids, x_coords, y_coords)
    # Filter the stdevs to avoid large jumps based on outliers
    stdevs_x = zero_phase_filter(stdevs_x_orig)
    stdevs_y = zero_phase_filter(stdevs_y_orig)

    first_track_idc, first_track_frames = get_first_track_indices(frame_ids, track_ids)
    last_track_idc, _ = get_last_track_indices(frame_ids, track_ids)

    if debug == True:
        print(f"Tracks start at following indices: {first_track_idc}")
        print(f"Tracks end at following indices: {last_track_idc}")

    # Track merging result: array that links original track and merged track
    unique_tracks_merged = unique_tracks.copy()
    track_ids_merged = track_ids.copy()
    
    # Store the polyfit constants for each track, to plot afterwards. It shall be of equal size as the unique_tracks_merged
    polyfit_constants_x = np.zeros((len(unique_tracks_merged), 3)) # 3 coefficients for the quadratic fit
    polyfit_constants_y = np.zeros((len(unique_tracks_merged), 3))

    if debug == True:
        # Plot the stdevs and mark the first and last points of each track
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Raw data and first and last point of tracks (X). Standard Deviation of X and Y Coordinates')
        gs = fig.add_gridspec(3, 1)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

        ax1.plot(frame_ids, x_coords, marker='.', linestyle='', label='Original')
        ax2.plot(frame_ids, y_coords, marker='.', linestyle='', label='Original')

        ax1.plot(frame_ids[first_track_idc], x_coords[first_track_idc], marker='x', linestyle='', color='black', label='First Point')
        ax2.plot(frame_ids[first_track_idc], y_coords[first_track_idc], marker='x', linestyle='', color='black', label='First Point')

        ax1.plot(frame_ids[last_track_idc], x_coords[last_track_idc], marker='x', linestyle='', color='red', label='Last Point')
        ax2.plot(frame_ids[last_track_idc], y_coords[last_track_idc], marker='x', linestyle='', color='red', label='Last Point')

        ax3.plot(unique_frames, stdevs_x, marker='.', linestyle='', label='Stdev X Filtered')
        ax3.plot(unique_frames, stdevs_x_orig, marker='.', linestyle='', label='Stdev X Original')
        ax3.plot(unique_frames, stdevs_y, marker='.', linestyle='', label='Stdev Y Filtered')
        ax3.plot(unique_frames, stdevs_y_orig, marker='.', linestyle='', label='Stdev Y Original')

        ax1.set_ylabel('X Coordinate')
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('Y Coordinate')
        ax3.set_ylabel('Stdev')

        plot_track_numbers(frame_ids, track_ids, x_coords, ax1)
        plot_track_numbers(frame_ids, track_ids, y_coords, ax2)

        plt.legend()
        plt.grid(True)
        plt.show()
   
    for i, track_id in enumerate(unique_tracks_merged): 
        print(f"Unique tracks: {unique_tracks_merged}")

        # Iterate over the merged tracks, because otherwise multiple track merging will not happen correctly
        # e.g. 1-2-3-4-5 where 123 should merge, then becomes 1-1-2-4-5 instead of 1-1-1-4-5
        if debug == True:
            print(f"Track ID {track_id}")

        # Get current track data
        track_indices = np.where(track_ids_merged == track_id)[0]
        x_coords_track = x_coords[track_indices]
        y_coords_track = y_coords[track_indices]
        frame_ids_track = frame_ids[track_indices]

        frame_last = frame_ids_track[-1] # Last frame of this track

        if debug == True:
            print(f"Last frame of this track: {frame_last}")

        mask_frame_search_window = np.logical_and(frame_ids >= frame_last, frame_ids <= frame_last + FRAME_SEARCH_WINDOW_NEXT_TRACK) # Search window for next appearing tracks
        frame_ids_search_window = frame_ids[mask_frame_search_window] # Frames in the search window
        frame_ids_search_window_unique = np.unique(frame_ids_search_window)

        # Check if tracks start in the search window
        next_tracks = []
        for next_frame in frame_ids_search_window_unique:
            if next_frame in first_track_frames:
                next_track_idc = first_track_idc[first_track_frames.index(next_frame)]
                next_track_id = track_ids[next_track_idc]
                # Get track length
                next_track_indices = np.where(track_ids == next_track_id)[0]
                next_track_length = len(next_track_indices)
                if next_track_length >= MIN_TRACK_LENGTH_COMPARE:
                    next_tracks.append(next_track_id)

        if debug == True:
            print(f"Potential tracks to merge with {track_id}: {next_tracks}")

        # If the stdev exceeds a threshold, the tracks are distinguishable, so we can attempt to merge with the next track(s)
        stdevs_cur_frame_idx = np.where(stdevs_frames == frame_last)[0][0]

        if stdevs_x[stdevs_cur_frame_idx] >= MERGE_THD_STDEV_X or stdevs_y[stdevs_cur_frame_idx] >= MERGE_THD_STDEV_Y:

            if len(x_coords_track) >= POLYFIT_LEN: # There must be at least the polyfit length of coordinates
                polyfit_x_in = np.arange(len(x_coords_track[-POLYFIT_LEN:]))
                polyfit_x_val = x_coords_track[-POLYFIT_LEN:]
                p_x = np.polyfit(polyfit_x_in, polyfit_x_val, 2)
                polyfit_constants_x[i] = p_x

                polyfit_y_in = np.arange(len(y_coords_track[-POLYFIT_LEN:]))
                polyfit_y_val = y_coords_track[-POLYFIT_LEN:]
                p_y = np.polyfit(polyfit_y_in, polyfit_y_val, 2)
                polyfit_constants_y[i] = p_y

                # For the next appearing tracks, compare the distance to the polyfit
                distances_x = []
                distances_y = []

                # Compute distances with next tracks
                for _, next_track_id in enumerate(next_tracks):
                    # Get the x, y coordinates of the next track
                    # next_track_id = unique_tracks[j]
                    next_track_indices = np.where(track_ids == next_track_id)[0]
                    frame_next = frame_ids[next_track_indices[0]]

                    if frame_next - frame_last > 10:
                        distances_x.append(np.inf)
                        distances_y.append(np.inf)
                        if debug == True:
                            print(f"Merging attempt failed. Track ID {track_id} is too many frames apart from track ID {next_track_id}.")
                        break # Frames are too far apart
                
                    x_coords_next = x_coords[next_track_indices[0]]
                    y_coords_next = y_coords[next_track_indices[0]]

                    # Distance between the last point of the polyfit of the current track with the first point of the next track
                    fit_x = np.polyval(p_x, polyfit_x_in[-1]) # Get last point of polyfit
                    fit_y = np.polyval(p_y, polyfit_y_in[-1])

                    if debug == True:
                        print(f"Polyfit X: {p_x}")
                        print(f"Frames indices: {frame_ids[track_indices[-POLYFIT_LEN:]]}")
                        print(f"Polyfit values: {np.polyval(p_x, polyfit_x_in[-POLYFIT_LEN:])}")

                    distance_x = abs(fit_x - x_coords_next)
                    distance_y = abs(fit_y - y_coords_next)

                    distances_x.append(distance_x)
                    distances_y.append(distance_y)

                if len(distances_x) > 0:
                    print(f"Distances between track {track_id} and track(s) {next_tracks} (x): {distances_x, distances_y}")

                    # Get the track ID with the smallest distance
                    mask_closest_track_x = np.argmin(distances_x)
                    mask_closest_track_y = np.argmin(distances_y)

                    closest_track_x = next_tracks[mask_closest_track_x]
                    closest_track_y = next_tracks[mask_closest_track_x]

                    if debug == True:
                        print(f"Track ID {track_id} is closest to track ID {closest_track_x} (x, {distances_x[mask_closest_track_x]}) and {closest_track_y} (y, {distances_y[mask_closest_track_y]}).")
                        print(f"Stdev X: {stdevs_x[stdevs_cur_frame_idx]}, Stdev Y: {stdevs_y[stdevs_cur_frame_idx]}")

                    # MERGING HAPPENS HERE
                    # Overwrite the 'next' track ID with the current track ID
                    # Caution: This will result in the 'next' track ID disappearing from unique_tracks_merged, hence this one will not be iterated over
                    if distances_x[mask_closest_track_x] <= MERGE_THD_DIST_X and stdevs_x[stdevs_cur_frame_idx] > MERGE_THD_STDEV_X:
                        if debug == True:
                            print(f"Merging track {track_id} with track {closest_track_x} (x).")
                        # Merge the tracks
                        unique_tracks_merged[unique_tracks == closest_track_x] = track_id # look in original array (non-merged), because we must replace the original track_id
                        track_ids_merged[track_ids == closest_track_x] = track_id # look in original array (non-merged), because we must replace the original track_id
                    if distances_y[mask_closest_track_y] <= MERGE_THD_DIST_Y and stdevs_y[stdevs_cur_frame_idx] > MERGE_THD_STDEV_Y:
                        if debug == True:
                            print(f"Merging track {track_id} with track {closest_track_y} (y).")
                        # Merge the tracks
                        unique_tracks_merged[unique_tracks == closest_track_y] = track_id
                        track_ids_merged[track_ids == closest_track_y] = track_id

    if debug == True:
        for i, track_id in enumerate(unique_tracks):

            if track_id != unique_tracks_merged[i]:
                print(f"Track ID {track_id} is merged with track ID {unique_tracks_merged[i]}.")
        print(f"Track merge results: {unique_tracks_merged}")

        # Plot the merged tracks
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Merged Tracks')
        gs = fig.add_gridspec(3, 1)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

        # Plot original on top
        for i, track_id in enumerate(unique_tracks):
            mask = track_ids == track_id
            track_indices = np.where(mask)[0]
            ax1.plot(frame_ids[track_indices], x_coords[track_indices], marker='.', linestyle='', label=f'Track ID {track_id}')
            ax1.plot(frame_ids[track_indices[-POLYFIT_LEN:]], np.polyval(polyfit_constants_x[i], np.arange(len(track_indices[-POLYFIT_LEN:]))), linestyle='-', label='Polyfit X', color='red')
                        
        plot_track_numbers(frame_ids, track_ids, x_coords, ax1)

        # Plot merged tracks
        for track_id in np.unique(track_ids_merged):
            mask = track_ids_merged == track_id
            track_indices = np.where(mask)[0]
            ax2.plot(frame_ids[track_indices], x_coords[track_indices], marker='.', linestyle='', label=f'Track ID {track_id}')
            # ax2.plot(frame_ids[track_indices[-POLYFIT_LEN:]], np.polyval(polyfit_constants_x[i], np.arange(len(track_indices[-POLYFIT_LEN:]))), linestyle='-', label='Polyfit X', color='red')
        
        plot_track_numbers(frame_ids, track_ids, y_coords, ax2)
        
        # Plot stdev
        ax3.plot(unique_frames, stdevs_x, marker='.', linestyle='', label='Stdev X')
        ax3.plot(unique_frames, stdevs_y, marker='.', linestyle='', label='Stdev Y')
        ax3.set_ylabel('Stdev')

        ax1.set_ylabel('X Coordinate')
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('X Coordinate')

        plt.legend()
        plt.grid(True)
        plt.show()

    return frame_ids, track_ids_merged, x_coords, y_coords

def plot_track_numbers(frame_ids, track_ids, coords, ax):
    """Plot the track number on an existing subplot (ax) at the provided coordinates.
    Frame ID on x-axis."""
    for i, track_id in enumerate(np.unique(track_ids)):
        mask = track_ids == track_id
        track_indices = np.where(mask)[0]
        ax.text(frame_ids[track_indices[0]], coords[track_indices[0]], f'{track_id}', fontsize=8, color='black')

    return

# %% FILL DISCONTINUITIES
from scipy.interpolate import interp1d

def detect_discontinuities(frame_ids, threshold=1):
    """Returns the start and end frame IDs of the discontinuities in the frame_ids array.
    The interpolation shall happen between these points. They are the start and end points of existing tracks."""
    mask_start = np.where(np.diff(frame_ids) > threshold)[0]
    mask_end = np.where(np.diff(frame_ids) > threshold)[0] + 1

    discontinuities_start = frame_ids[mask_start]
    discontinuities_end = frame_ids[mask_end]

    start_end_pairs_value = list(zip(discontinuities_start, discontinuities_end))
    start_end_pairs_index = list(zip(mask_start, mask_end))
    print(f"Discontinuities detected at frame values: {start_end_pairs_value}")
    print(f"Discontinuities detected at indices: {start_end_pairs_index}")

    return start_end_pairs_value, start_end_pairs_index

def fill_discontinuities(frame_ids, track_ids, x_coords, y_coords, debug=False):
    """Fill the discontinuities in the frame_ids array by linear interpolation between the start and end points of the discontinuities.
    
    Args:
    frame_ids (np.array): Array of frame IDs.
    track_ids (np.array): Array of track IDs.
    x_coords (np.array): Array of x coordinates.
    y_coords (np.array): Array of y coordinates.
    
    Returns:
    filled_frame_ids (np.array): Array of frame IDs with filled discontinuities.
    filled_track_ids (np.array): Array of track IDs with filled discontinuities.
    filled_x_coords (np.array): Array of x coordinates with filled discontinuities.
    filled_y_coords (np.array): Array of y coordinates with filled discontinuities
    """

    filled_frame_ids = frame_ids # For filling the discontinuities, insert into the existing arrays
    filled_x_coords = x_coords
    filled_y_coords = y_coords
    filled_track_ids = track_ids

    unique_tracks = np.unique(track_ids)

    for track_id in unique_tracks:
        track_indices = np.where(track_ids == track_id)[0]
        track_frame_ids = frame_ids[track_indices]
        track_x_coords = x_coords[track_indices]
        track_y_coords = y_coords[track_indices]

        _, start_end_pairs = detect_discontinuities(track_frame_ids)
        
        if debug == True:
            print(f"Track frame IDs {track_frame_ids}")
            print(f"Discontinuities detected for track {track_id}: {track_frame_ids[start_end_pairs]}")

        for start, end in start_end_pairs:
            # Interpolation range
            interp_range = np.arange(track_frame_ids[start], track_frame_ids[end])
            
            # Linear interpolation
            interp_func_x = interp1d([track_frame_ids[start], track_frame_ids[end]], [track_x_coords[start], track_x_coords[end]], kind='linear')
            interp_func_y = interp1d([track_frame_ids[start], track_frame_ids[end]], [track_y_coords[start], track_y_coords[end]], kind='linear')
            interp_x_coords = interp_func_x(interp_range)
            interp_y_coords = interp_func_y(interp_range)

            # Insert the interpolated values into the lists
            # start is relative to this track, but we need to insert the interpolated values into the global arrays. Insert into the existing arrays
            filled_frame_ids = np.insert(filled_frame_ids, track_frame_ids[start+1], interp_range)
            filled_x_coords = np.insert(filled_x_coords, track_frame_ids[start+1], interp_x_coords)
            filled_y_coords = np.insert(filled_y_coords, track_frame_ids[start+1], interp_y_coords)
            filled_track_ids = np.insert(filled_track_ids, track_frame_ids[start+1], [track_id] * len(interp_range))          

            if debug == True:
                print(f"Discontinuity detected for track {track_id} between frames {track_frame_ids[start]} and {track_frame_ids[end]}.")
                print(f"Interpolated x: {interp_x_coords}")
                print(f"Interpolated y: {interp_y_coords}")
                print(f"Filled frames: {filled_frame_ids}")

    if debug == True:
        # Plot before and after. Mark filled points.
        fig = plt.figure(figsize=(12, 6))

        gs = fig.add_gridspec(2, 1)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

        ax1.plot(filled_frame_ids, filled_x_coords, marker='x', linestyle='', color='red', label='Filled')
        ax2.plot(filled_frame_ids, filled_y_coords, marker='x', linestyle='', color='red', label='Filled')

        ax1.plot(frame_ids, x_coords, marker='.', linestyle='', label='Original')
        ax2.plot(frame_ids, y_coords, marker='.', linestyle='', label='Original')

        plot_track_numbers(frame_ids, track_ids, x_coords, ax1)
        plot_track_numbers(frame_ids, track_ids, y_coords, ax2)

        ax1.set_ylabel('X Coordinate')
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('Y Coordinate')

        plt.legend()
        plt.grid(True)
        plt.show()

    # Print amount of discontinuities filled
    print(f"Filled {len(filled_frame_ids) - len(frame_ids)} frames.")

    return np.array(filled_frame_ids), np.array(filled_track_ids), np.array(filled_x_coords), np.array(filled_y_coords)

def remove_outliers_first_sample(frame_ids, track_ids, x_coords, y_coords, threshold_x=300, threshold_y=300, window_size=5, debug=False):
    """Remove tracks where the first sample is further than the threshold from all other tracks in the same frame or frames before (window_size).
    To be performed after merging tracks and filling discontinuities.
    
    Args:
    frame_ids (np.array): Array of frame IDs.
    track_ids (np.array): Array of track IDs.
    x_coords (np.array): Array of x coordinates.
    y_coords (np.array): Array of y coordinates.
    threshold_x (int): Threshold for x coordinate: larger distance = rejected.
    threshold_y (int): Threshold for y coordinate: larger distance = rejected.
    window_size (int): Number of frames before the first frame to check for other tracks.
    debug (bool): If True, plots the raw and cleaned data    

    Returns:
    frame_ids_cleaned (np.array): Array of frame IDs with removed outliers.
    track_ids_cleaned (np.array): Array of track IDs with removed outliers.
    x_coords_cleaned (np.array): Array of x coordinates with removed outliers.
    y_coords_cleaned (np.array): Array of y coordinates with removed outliers.
        
    """
    unique_tracks = np.unique(track_ids)
    invalid_tracks = []
    first_sample_per_track = []

    for i, track_id in enumerate(unique_tracks):
        track_indices = np.where(track_ids == track_id)[0]

        # Skip if the track is empty
        if len(track_indices) == 0:
            continue

        first_sample_per_track.append(track_indices[0])
        
        if i < 4:
            continue # Skip initial tracks to provide starting point

        first_index = track_indices[0]
        first_frame = frame_ids[first_index] # Frame of the first appearance of the current track ID
        
        # Get the x, y coordinates of the first appearance of the current track ID
        first_x, first_y = x_coords[first_index], y_coords[first_index]
        # Get the x, y coordinates of all other tracks in the current frame and frames before
        mask_first_frame = frame_ids == first_frame
        for j in range(1, window_size+1):
            mask_first_frame = np.logical_or(mask_first_frame, frame_ids == first_frame - j)

        # Exclude the current track
        mask_first_frame[first_index] = False

        tracks_in_frame = track_ids[mask_first_frame]

        # If no other tracks in these frames, skip
        if len(tracks_in_frame) == 0:
            if debug == True:
                print(f"No other tracks in frame {first_frame} and {window_size} frames before.")
            continue
        
        # Remove the invalid tracks from the mask
        tracks_in_frame = tracks_in_frame[~np.isin(tracks_in_frame, invalid_tracks)]

        other_x = x_coords[mask_first_frame]
        other_y = y_coords[mask_first_frame]
        
        # Calculate the distance between the first point of the current track and all other tracks
        distances_x = np.abs(first_x - other_x)
        distances_y = np.abs(first_y - other_y)

        if debug == True:
            print(f"Checking track ID {track_id}...")
            print(f"First frame of this track: {first_frame}")
            print(f"Other tracks in frame: {tracks_in_frame}")
            print(f"Other tracks in frame after removing invalid tracks: {tracks_in_frame}")
            print(f"Distances x: {distances_x}")
            print(f"Distances y: {distances_y}")

        # Check if the first point of the current track is further than the threshold from all other tracks
        if np.all(distances_x > threshold_x) or np.all(distances_y > threshold_y):
            if debug == True:
                print(f"Track ID {track_id} is removed.") 
            invalid_tracks.append(track_id)          
    
    mask_invalid = np.isin(track_ids, invalid_tracks)

    if debug == True:
        # Plot raw and cleaned data
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Outlier Removal Based on First Sample Distance')

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        colors = plt.cm.jet(np.linspace(0, 1, len(np.unique(track_ids))))
        
        ax1.plot(frame_ids, x_coords, marker='.', linestyle='', label='Original')
        ax2.plot(frame_ids, y_coords, marker='.', linestyle='', label='Original')

        ax1.plot(frame_ids[mask_invalid], x_coords[mask_invalid], marker='.', linestyle='', color='red', label='Removed')
        ax2.plot(frame_ids[mask_invalid], y_coords[mask_invalid], marker='.', linestyle='', color='red', label='Removed')

        # Plot first sample per track
        ax1.plot(frame_ids[first_sample_per_track], x_coords[first_sample_per_track], marker='x', linestyle='', color='black', label='First Sample')
        ax2.plot(frame_ids[first_sample_per_track], y_coords[first_sample_per_track], marker='x', linestyle='', color='black', label='First Sample')

        plot_track_numbers(frame_ids, track_ids, x_coords, ax1)
        plot_track_numbers(frame_ids, track_ids, y_coords, ax2)

        ax1.set_ylabel('X Coordinate')
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('Y Coordinate')

        plt.legend()
        plt.grid(True)
        plt.show()
    
    print(f"Outlier rejection removed following tracks: {invalid_tracks}")

    return frame_ids[~mask_invalid], track_ids[~mask_invalid], x_coords[~mask_invalid], y_coords[~mask_invalid]

if __name__ == '__main__':
    # frame_ids, track_ids, x_coords, y_coords, _, _ = da.load_tracking_data(r'output\VID-PI-EL-BEL-240412-NC-Milton-TP-R1_20240730_100328_yolov8s-trackcycling-04.json')
    # frame_ids, track_ids, x_coords, y_coords, _, _ = da.load_tracking_data(r'output\C1319-20240701_112129_yolov8m-trackcycling-03-output.json')
    frame_ids, track_ids, x_coords, y_coords, _, _ = da.load_tracking_data(r'output\sample_yolov8s-output.json')
    _, track_ids_merged, _, _ = merge_tracks(frame_ids, track_ids, x_coords, y_coords, debug=True)
    frame_ids_fill_x, track_ids_fill, x_coords_fill, y_coords_fill = fill_discontinuities(frame_ids, track_ids_merged, x_coords, y_coords, debug=True)
    frame_ids_clean, track_ids_clean, x_coords_clean, y_coords_clean = remove_outliers_first_sample(frame_ids_fill_x, track_ids_fill, x_coords_fill, y_coords_fill, debug=True)

# %% OBSOLETE
# Save the x-coordinate data of track to json
# import json
# def save_track_data_to_json(frame_ids, track_ids, x_coords, y_coords, json_out='track_data.json'):
#     track_data = {
#         'frame_ids': frame_ids.tolist(),
#         'track_ids': track_ids.tolist(),
#         'x_coords': x_coords.tolist(),
#         'y_coords': y_coords.tolist()
#     }

#     with open(json_out, 'w') as file:
#         json.dump(track_data, file)

#     return