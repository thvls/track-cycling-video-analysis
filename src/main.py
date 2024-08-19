"""
Workflow:
- Load video file
- Run object detection
- Load resulting tracking data
- Preprocess tracking data to improve changeover detection consistency: merge (clearly distinguishable) track ID, remove outliers, fill discontinuities
- Detect changeovers
- Plot and save changeover detection results
- Extract changeover clips

TO-DO:
- Update local debug (extra print and visual output) flags to global DEBUG_ON flag
"""

# %%
from cyclist_tracking import run_cyclist_tracking
import detection_analysis
import preprocessing
import plotting_tools
from datetime import datetime
import os
import cv2
import json
from matplotlib.widgets import Slider
from extract_changeover_clips import extract_changeover_clips
import numpy as np
import matplotlib.pyplot as plt
import plotting_tools

def select_video_file(sample_path = 'media/sample.mp4'):
    # Select video file
    path_vid_input = input("Enter the path to the video file: ")
    path_vid_sample = sample_path

    if not path_vid_input or not os.path.exists(path_vid_input):
        use_sample = input(f"No corresponding video found. Use sample video: {path_vid_sample}? (y/n, default: y)")

        if use_sample.lower() == 'y' or not use_sample:
            path_vid_input = path_vid_sample
            print(f"Video used: {path_vid_input}")
        else: # Break
            print("No video selected. Exiting...")
            exit()
    
    return path_vid_input

def select_object_detection_json_file(path_vid_input):
    run_object_detection = True

    # If already done, ask to rerun object detection
    # Look for files containing video name in 'output' directory
    vid_id = os.path.splitext(os.path.basename(path_vid_input))[0]
    if os.path.exists('output'):
        vid_files = [f for f in os.listdir('output') if vid_id in f and f.endswith('.mp4')]
        json_files = [f for f in os.listdir('output') if vid_id in f and f.endswith('.json')]
        print(f"Existing files corresponding to video analysis found: {vid_files}")
    else:
        os.makedirs('output')
        vid_files = []
        json_files = []

    # Select latest json file
    if json_files:
        if len(json_files) == 1:
            path_YOLO_out_json = os.path.join('output', json_files[0])
        elif len(json_files) > 1: 
            # If multiple json files exist, prompt user to select which one to use
            print("Multiple JSON files found. Select which one to use.")
            # json_files.sort()
            # path_YOLO_out_json = os.path.join('output', json_files[-1])
            for i, f in enumerate(json_files):
                print(f"{i}: {f}")
            json_file_idx = int(input("Enter the index of the JSON file to use: "))
            path_YOLO_out_json = os.path.join('output', json_files[json_file_idx])
        
        path_YOLO_out_vid = path_YOLO_out_json.replace('.json', '.mp4')

        if os.path.exists(path_YOLO_out_json):
            rerun = input("YOLO objection detection already run on this video. Rerun? (y/n, default: n)")

            if rerun.lower() != 'y':
                run_object_detection = False

        if not os.path.exists(path_YOLO_out_vid):
            print(f"No corresponding YOLO output video found. Using original for visualization.")
            path_YOLO_out_vid = path_vid_input

        print(f"Result JSON file: {path_YOLO_out_json}")
        print(f"Result video file: {path_YOLO_out_vid}")

    return path_YOLO_out_vid, path_YOLO_out_json, run_object_detection

def restructure_changeovers(changeovers, changeover_times):
        # Print amount of changeovers
        print(f"Detected {len(changeovers) // 2} changeovers.")

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

def plot_tune_changeover_detection(path_YOLO_out_vid, frame_ids, track_ids, x_coords, y_coords, changeovers):
    """
    THIS MUST BE UPDATED TO USE VIDEOPLOTTER CLASS.

    - Plots X, Y coordinates and distance between riders
    - Shows video frame corresponding to frame selected by tracer in the plots
    - Shows detected changeovers
    - Sliders to tune changeover detection thresholds
    """
    now_string = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load video to show frames
    cap = cv2.VideoCapture(path_YOLO_out_vid)

    fig = plt.figure(figsize=(12, 6))
    # Set figure title
    fig.suptitle(f'Changeover Detection Tuning. Video: {path_YOLO_out_vid}')

    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[:, 1])

    # Plot x- and y-coordinates
    ax1.plot(frame_ids, x_coords, '.', label='X-Coordinate')
    # ax1.set_xlabel('Frame no.')
    ax1.set_ylabel('Position [px]')
    ax1.set_title('X-Coordinates')

    ax2.plot(frame_ids, y_coords, '.', label='Y-Coordinate')
    # ax2.set_xlabel('Frame no.')
    ax2.set_ylabel('Position [px]')
    ax2.set_title('Y-Coordinates')

    plotting_tools.plot_track_numbers(frame_ids, track_ids, x_coords, ax1)
    plotting_tools.plot_track_numbers(frame_ids, track_ids, y_coords, ax2)

    # Plot filtered distance 
    # HOW DOES THIS WORK?? The variable y_dist_min_clean_filt is not defined in this scope
    ax3.plot(frame_ids_unique, y_dist_min_clean_filt, label='Filtered Distance')
    ax3.plot(frame_ids_unique, y_dist_min, label='Raw Distance')
    ax3.set_xlabel('Frame no.')
    ax3.set_ylabel('Distance [px]')
    ax3.set_title('Filtered Distance')

    # Plot video frame
    frame = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_plot = ax4.imshow(img)
    ax4.axis('off')
    ax4.set_title(f'Frame {frame}')

    # Draw xy-coordinate frame at left top corner of video frame
    # Draw x-axis
    plt.arrow(0, 0, 100, 0, head_width=10, head_length=10, fc='red', ec='red')
    ax4.text(110, 0, 'x', fontsize=12, color='red')
    # Draw y-axis
    plt.arrow(0, 0, 0, 100, head_width=10, head_length=10, fc='red', ec='red')
    ax4.text(5, 110, 'y', fontsize=12, color='red')

    # Add tracer line linked to current frame
    tracer_line1 = ax1.axvline(frame, color='b', linestyle='--', label='Current frame')
    tracer_line2 = ax2.axvline(frame, color='b', linestyle='--')
    tracer_line3 = ax3.axvline(frame, color='b', linestyle='--')

    changeover_markers = []

    # Add changeover markers
    def update_changeover_markers(changeovers, axes):
        nonlocal changeover_markers

        # Remove existing changeover markers if the variable exists
        if changeover_markers is not None:
            for marker in changeover_markers:
                marker.remove()
            changeover_markers.clear()

        for changeover in changeovers:
            changeover_type, frame = changeover
            if changeover_type == 'start':
                for ax in axes:
                    marker = ax.axvline(frame, color='green', linestyle='--', label='Start')
                    changeover_markers.append(marker)
            elif changeover_type == 'end':
                for ax in axes:
                    marker = ax.axvline(frame, color='red', linestyle='--', label='End')
                    changeover_markers.append(marker)
        return changeover_markers
    
    changeover_markers = update_changeover_markers(changeovers, [ax1, ax2, ax3])

    # Show legends
    # Only show 1 instance of each marker in the legend
    plt.sca(ax1) # Set active axis to ax1
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=[handles[1], handles[2], handles[3]], labels=['Current frame', 'Start', 'End'], loc='upper right') # 0 is the data itself

    # for ax in [ax1, ax2, ax3]:
    #     handles, labels = ax.get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     ax.legend(by_label.values(), by_label.keys())

    
    # Add input fields for changeover detection thresholds
    # Thresholds: min_distance, max_distance, min_frames, max_frames
    # Slider for start_thd

    # Get ax4 position
    ax_start_thd = plt.axes([0.6, 0.1, 0.3, 0.02])
    start_thd_slider = Slider(ax_start_thd, 'Start Threshold', valmin=0, valmax=100, valinit=15, valstep=1)
    end_thd_slider = Slider(plt.axes([0.6, 0.05, 0.3, 0.02]), 'End Threshold', valmin=0, valmax=100, valinit=50, valstep=1)
    
    # Update changeover detection based on slider values
    def update(val):
        global start_thd_val, end_thd_val, changeover_markers
        start_thd_val = start_thd_slider.val
        end_thd_val = end_thd_slider.val

        changeovers = detection_analysis.detect_changeovers(y_dist_min_clean_filt, start_thd=start_thd_val, start_confirm_frames=25, confirm_thd=150, end_thd=end_thd_val, end_confirm_frames=5)

        # Update changeover markers
        changeover_markers = update_changeover_markers(changeovers, [ax1, ax2, ax3])

        fig.canvas.draw()

    start_thd_slider.on_changed(update)
    end_thd_slider.on_changed(update)

    # When user clicks in left subplots, move tracer line to that x-coordinate
    def on_click(event):
        if event.inaxes in [ax1, ax2, ax3]:
            frame_idx = int(event.xdata)
            
            # Set tracer lines to new x-coordinate
            tracer_line1.set_xdata([frame_idx, frame_idx])
            tracer_line2.set_xdata([frame_idx, frame_idx])
            tracer_line3.set_xdata([frame_idx, frame_idx])

            # Update video frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, img = cap.read()

            # Plot new frame
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_plot.set_data(img)
                ax4.set_title(f'Frame {frame_idx}')
                ax4.axis('off')
                fig.canvas.draw() # Necessary to update the plot

    fig.canvas.mpl_connect('button_press_event', on_click) # Connect click-on-plot event to function

    plt.show()

    # Save figure if user wants
    save_fig = input("Save figure? (y/n, default: n) ")
    # Replace video extension with png. Be robust against caps
    if save_fig.lower() == 'y':
        fig_append_string = f'_changeover_detection_{now_string}.png'
        fig_name = os.path.basename(path_YOLO_out_vid).replace('.mp4', fig_append_string).replace('.MP4', fig_append_string)
        fig_path = os.path.join('output', fig_name)
        fig.savefig(fig_path)
        print(f"Figure saved to {fig_path}")

    return changeovers

if __name__ == '__main__':
    # %% LOAD VIDEO AND DATA

    # Datetime string for saving files
    now = datetime.now()
    now_string = now.strftime("%Y%m%d_%H%M%S")
   
    path_vid_input = select_video_file()

    # Select object detection JSON file
    path_YOLO_out_vid, path_YOLO_out_json, run_object_detection = select_object_detection_json_file(path_vid_input)

    if run_object_detection:
        path_YOLO_out_vid, path_YOLO_out_json = run_cyclist_tracking(vid_file=path_vid_input, model_id = 'yolov8n-trackcycling-04.pt', model_mode='track')    
    
    frame_ids, track_ids, x_coords, y_coords, _, heights = detection_analysis.load_tracking_data(path_YOLO_out_json)
    
    # Open video
    cap = cv2.VideoCapture(path_YOLO_out_vid)

    # Create plotter object for initial visualization
    plotter = plotting_tools.VideoPlotterTracks(frame_ids, track_ids, x_coords, y_coords, path_YOLO_out_vid)
    plotter.show()

    # %% PREPROCESSING AND CHANGEOVER DETECTION
    frame_ids_orig, track_ids_orig, x_coords_orig, y_coords_orig = frame_ids, track_ids, x_coords, y_coords

    # Preprocess the data
    _, track_ids, _, _ = preprocessing.merge_tracks(frame_ids, track_ids, x_coords, y_coords, debug=False)
    frame_ids, track_ids, x_coords, y_coords = preprocessing.remove_outliers_first_sample(frame_ids, track_ids, x_coords, y_coords, debug=False)
    frame_ids, track_ids, x_coords, y_coords = preprocessing.fill_discontinuities(frame_ids, track_ids, x_coords, y_coords)
    preprocessing.plot_data(frame_ids, track_ids, x_coords, y_coords)

    # Changeover detection
    y_dist_min, frame_ids_unique = detection_analysis.compute_pairwise_distance(frame_ids, y_coords)
    y_dist_min_clean_filt = detection_analysis.zero_phase_filter(y_dist_min, wn=0.1, order=4) # THIS SHOULD ALL BE COMBINED INSIDE DETECT_CHANGEOVERS()!
    changeovers = detection_analysis.detect_changeovers(y_dist_min_clean_filt)

    changeovers = plot_tune_changeover_detection(path_YOLO_out_vid, frame_ids, track_ids, x_coords, y_coords, changeovers)

    # %% (DEVELOPMENT) NORMALIZE RIDER MOVEMENT
    detection_analysis.normalize_rider_movement(y_dist_min_clean_filt, frame_ids_unique, heights, frame_ids_orig, debug=True) # Original frame_ids bc heights is not yet modified (no input to track merging)

    # Plot the average bounding box height per frame
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('BBox Height')

    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[0:-2, 1], sharex=ax1)
    ax5 = fig.add_subplot(gs[-2:, 1])

    ax1.plot(frame_ids, x_coords, marker='.', linestyle='')
    ax2.plot(frame_ids, y_coords, marker='.', linestyle='')
    ax3.plot(frame_ids_unique, y_dist_min_clean_filt, marker='.', linestyle='')
    ax4.plot(frame_ids_orig, heights, marker='.', linestyle='') # Plot original frame IDs because height has not been modified
    
    plotting_tools.plot_track_numbers(frame_ids, track_ids, x_coords, ax1)
    plotting_tools.plot_track_numbers(frame_ids, track_ids, y_coords, ax2)

    ax1.set_ylabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax3.set_ylabel('Distance')
    ax4.set_ylabel('Height')
    ax3.set_xlabel('Frame ID')

    plt.legend()
    plt.grid(True)

    plotter = plotting_tools.VideoPlotterGeneric(fig, [ax1, ax2, ax3, ax4], ax5, video_path=path_YOLO_out_vid, x_type='frame')
    plotter.show()

    # %% RESTRUCTURE CHANGEOVERS TO INCLUDE ALL DATA AND SAVE RESULTS
    if not cap:
        cap = cv2.VideoCapture(path_YOLO_out_vid)

    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Check if framerate is constant
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print("Static frame rate check. Check whether video duration is correct.")
    print(f"Frame rate: {fps} fps")
    print(f"Frame count: {frame_count}")
    print(f"Expected duration for constant {fps} fps: {duration} s")
    
    # Get start and end times of changeovers
    changeover_times = []
    for changeover in changeovers:
        changeover_type, frame = changeover
        time = frame / fps
        changeover_times.append(time)

    # Append times to changeovers
    changeovers_out = restructure_changeovers(changeovers, changeover_times)
    
    # Save changeovers to JSON
    # Create changeovers folder if doesn't exist
    if not os.path.exists('output/changeovers'):
        os.makedirs('output/changeovers')
        
    vid_name_with_ext = os.path.basename(path_YOLO_out_vid)
    json_name_changeovers = vid_name_with_ext.replace('.mp4', f'_changeovers_{now_string}.json')
    json_path_changeovers = os.path.join('output/changeovers', json_name_changeovers)

    if input(f"Save results to {json_path_changeovers}? (y/n, default: n) ").lower() == 'y':
        with open(json_path_changeovers, 'w') as f:
            json.dump(changeovers_out, f, indent=4)

        print(f"Results saved to {json_path_changeovers}")
    else:
        print("Changeover timings not saved.")

    # %% Extract changeover videos
    # Ask user to export changeover clips
    if input("Extract changeover clips? (y/n, default: n) ").lower() == 'y':
        if not os.path.exists(json_path_changeovers):
            json_path_changeovers = input("Enter the path to the changeover JSON file: ")
        
        if json_path_changeovers:
            # Load changeover data
            with open(json_path_changeovers, 'r') as f:
                changeover_data = json.load(f)

            # Extract changeover videos
            clip_output_path = 'output/changeovers'

            extract_changeover_clips(json_path_changeovers, path_vid_input, clip_output_path, buffer_before=2, buffer_after=1)
        else:
            print("No changeover JSON file found. Exiting...")