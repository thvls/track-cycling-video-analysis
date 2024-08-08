import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider

# %% VideoPlotter Tool
"""Creates a figure showing the track coordinates on a left subplot and the
corresponding video frame on a right subplot. Interactive (seeking).
Should be updated to give subplots as input instead of creating inside."""

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

# %% Generic Video Plotter
class VideoPlotterGeneric:
    """Tool that adds an interactive video subplot to an existing corresponding data figure.

    Instructions:
        - Use the show method to display the figure.
        - Do not call plt.show() before or after calling the show method.

    Args:
        fig (matplotlib.figure.Figure): Figure object containing the data plots.
        axs_data (list): List of axis objects containing the data plots.
        ax_video (matplotlib.axes.Axes): Axis object to contain the video frame.
        video_path (str): Path to the video file.

    Methods (external):
        show: Show the figure.       
    
    Updates to-do:
        - Define translation function from data x-axis to video frame index.
    """
    def __init__(self, fig, axs_data, ax_video, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_id = video_path.split('/')[-1].split('\\')[-1]
        self.fig = fig
        self.axs_data = axs_data
        self.tracers = []
        self.current_x = 0
        # Get maximum value of the axes
        self.max_x = max([ax.get_xlim()[1] for ax in axs_data])
        
        self.setup_video(ax_video)
        self.add_tracers()
        self.connect_events()
                
        # self.update(0)
        # plt.show()
        # self.cap.release()

    def map_x_to_frame(self, x):
        """Map the x-coordinate of the data plot to the corresponding video frame index."""
        frame_idx = int(x * self.fps)
        return frame_idx

    def setup_video(self, ax_video):
        """Add the video to the provided axis."""

        ax_video.axis('off')
        self.im = ax_video.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()

        if ret:
            self.im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        self.fig.canvas.draw_idle()

    def add_tracers(self):
        """Add tracer lines to the existing data plots, showing the current video frame."""
        frame_idx = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        for ax in self.axs_data:
            line = ax.axvline(x=frame_idx, color='red')
            self.tracers.append(line)

    def update(self, x):
        self.current_x = x
        frame_idx = self.map_x_to_frame(x)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            self.im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        for tracer in self.tracers:
            tracer.set_xdata([x, x])
        
        self.fig.canvas.draw_idle()

    def update_frame(self, x):
        self.current_x = x
        frame_idx = self.map_x_to_frame(x)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        for tracer in self.tracers:
            tracer.set_xdata([x, x])
        
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        for ax in self.axs_data:
            if event.inaxes == ax:
                x = int(event.xdata)
                self.update_frame(x)
                break

    def on_key_press(self, event):
        if event.key == 'left':
            new_x = max(0, self.current_x - 1)
            self.update_frame(new_x)
        elif event.key == 'right':
            new_x = min(self.max_x, self.current_x + 1)
            self.update_frame(new_x)

    def connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def show(self):
        self.update(0)
        plt.draw()
        plt.show()