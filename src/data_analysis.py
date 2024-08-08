import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from detection_analysis import zero_phase_filter

def moving_average(data, window=30):
    """Input a list of data and return the moving average of the data"""
    return data.rolling(window=window, min_periods=1).mean()

def normalized_power(power_data, window=30):
    power_rolling = pd.Series(power_data).rolling(window=window, min_periods=1).mean()
    np_power = (np.mean(power_rolling ** 4)) ** 0.25
    return np_power

# Load the JSON data
file_path_power_data = 'data/pursuit_times.json'
with open(file_path_power_data, 'r') as file:
    data = json.load(file)

file_path_changeover_times = 'output\changeovers\C1319-20240701_112129_yolov8m-trackcycling-03-output_changeovers_20240730_131940.json'
with open(file_path_changeover_times, 'r') as file:
    changeover_data = json.load(file)

# Extract the changeover times for each rider
changeover_times = []
for co in changeover_data:
    changeover_times.append({
        'changeover_number': co['changeover_number'],
        'start_time': co['data']['start_time'],
        'end_time': co['data']['end_time']
    })

# Initialize a dictionary to hold power data for each rider
rider_power_data = {}

for entry in data:
    athlete_id = entry['athlete']['athleteId']
    if 'sensorData' in entry:
        sensor_data = entry['sensorData']
        for sensor_id, sensor_records in sensor_data.items():
            for record in sensor_records:
                if record.get('sensorValueName') == 'AveragePower':
                    timestamp = record.get('timestamp')
                    power_value = record.get('sensorValue')
                    if athlete_id not in rider_power_data:
                        rider_power_data[athlete_id] = []
                    rider_power_data[athlete_id].append((timestamp, power_value))

# Convert each rider's power data to a DataFrame
rider_dfs_t = {}
output_metrics = []

# Temporal data analysis
POWER_FILTER_WN = 0.15
POWER_FILTER_ORDER = 2

for athlete_id, power_data in rider_power_data.items():
    df = pd.DataFrame(power_data, columns=['Timestamp', 'Power'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    t_start = df['Timestamp'].iloc[0]
    df['Time'] = (df['Timestamp'] - t_start).dt.total_seconds()

    df['Power_MA_30s'] = moving_average(df['Power'], window=30)
    df['Power_MA_10s'] = moving_average(df['Power'], window=10)
    df['Power_MA_3s'] = moving_average(df['Power'], window=3)
    df['Power_Filt'] = zero_phase_filter(df['Power'], wn=POWER_FILTER_WN, order=POWER_FILTER_ORDER)
    
    total_energy = (df['Power'].sum() * (df['Timestamp'].diff().dt.total_seconds().fillna(0) / 3600)).sum()
    avg_power = df['Power'].mean()
    norm_power = normalized_power(df['Power'])
    
    rider_dfs_t[athlete_id] = df

    output_metrics.append({
        'athlete_id': athlete_id,
        'average_power': avg_power,
        'normalized_power': norm_power,
        'total_energy': total_energy
    })

output_metrics_df = pd.DataFrame(output_metrics)

# Changeover analysis
changeover_metrics = []
CO_ANALYSIS_WINDOW_POST_START = 6 # seconds

for idx, (athlete_id, df) in enumerate(rider_dfs_t.items()):
    for co in changeover_times:
        co_df = df[(df['Time'] >= co['start_time']) & (df['Time'] <= co['start_time'] + CO_ANALYSIS_WINDOW_POST_START)] # Get the data for the changeover
        # total_energy = (co_df['Power'].sum() * (co_df['Time'].diff().dt.total_seconds().fillna(0) / 3600)).sum()
        avg_power = co_df['Power'].mean()
        norm_power = normalized_power(co_df['Power'])
        changeover_metrics.append({
            'athlete_id': athlete_id,
            'changeover_number': co['changeover_number'],
            'average_power': avg_power,
            'normalized_power': norm_power,
            # 'total_energy': total_energy
        })

changeover_metrics_df = pd.DataFrame(changeover_metrics)

athlete_ids = list(rider_dfs_t.keys())

# Show the output metrics
print(output_metrics_df)
print(changeover_metrics_df)

# %% Power Filter Tuning Tool
"""A tool to help tune the Butterworth filter parameters for the power data.
Plots the power data and allows the user to adjust the filter parameters (using sliders) and see the effect on the filtered data."""

from matplotlib.widgets import Slider
import scipy.signal as signal

# Initial filter parameters
initial_wn = POWER_FILTER_WN
initial_order = POWER_FILTER_ORDER
initial_athlete_idx = 0
initial_athlete = athlete_ids[initial_athlete_idx]

df = rider_dfs_t[initial_athlete]

fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)
l1, = plt.plot(df['Time'], df['Power'], label='Raw Power', linestyle='-', linewidth=1, color='blue')
l2, = plt.plot(df['Time'], df['Power_Filt'], label='Filtered Power', linestyle='-', linewidth=1, color='red')
l3, = plt.plot(df['Time'], df['Power_MA_10s'], label='10s MA', linestyle='--', linewidth=1, color='green')
plt.xlabel('Time')
plt.ylabel('Power (Watts)')
plt.title(f'Power Filter Tuning Tool - Athlete {initial_athlete}')
plt.legend()
plt.grid(True)

# Define sliders for filter parameters
ax_wn = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_order = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_athlete = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

s_wn = Slider(ax_wn, 'Cutoff Freq (wn)', 0.01, 0.5, valinit=initial_wn, valstep=0.01)
s_order = Slider(ax_order, 'Order', 1, 10, valinit=initial_order, valstep=1)
s_athlete = Slider(ax_athlete, 'Athlete', 0, len(athlete_ids) - 1, valinit=initial_athlete_idx, valstep=1)

# Update function
def update(val):
    wn = s_wn.val
    order = int(s_order.val)
    athlete_id = athlete_ids[int(s_athlete.val)]
    df = rider_dfs_t[athlete_id]

    ax.set_title(f'Power Filter Tuning Tool - Athlete {athlete_id}')

    df['Power_Filt'] = zero_phase_filter(df['Power'], wn, order)

    # Update the plot data
    l1.set_xdata(df['Time'])
    l1.set_ydata(df['Power'])
    l2.set_xdata(df['Time'])
    l2.set_ydata(df['Power_Filt'])
    l3.set_xdata(df['Time'])
    l3.set_ydata(df['Power_MA_10s'])

    # Adjust the x and y limits to fit the new data
    ax.relim()
    ax.autoscale_view()
    
    fig.canvas.draw_idle()

# Register the update function with each slider
s_wn.on_changed(update)
s_order.on_changed(update)
s_athlete.on_changed(update)

plt.show()

# %% Power filter debug plot
# Plot raw and filtered power data for each rider on one subplot

fig = plt.figure(figsize=(12, 6))
fig.suptitle('Power Filter Debug Plot')

gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
ax4 = fig.add_subplot(gs[1, 1], sharex=ax1)
ax5 = fig.add_subplot(gs[:, 2]) # Subplot to show the video

ax1.plot(rider_dfs_t[athlete_ids[0]]['Time'], rider_dfs_t[athlete_ids[0]]['Power'], label='Raw Power', linestyle='-', linewidth=1, color='blue')
ax1.plot(rider_dfs_t[athlete_ids[0]]['Time'], rider_dfs_t[athlete_ids[0]]['Power_Filt'], label='Filtered Power', linestyle='-', linewidth=1, color='red')
ax1.plot(rider_dfs_t[athlete_ids[0]]['Time'], rider_dfs_t[athlete_ids[0]]['Power_MA_10s'], label='10s MA', linestyle='--', linewidth=1, color='green')

ax2.plot(rider_dfs_t[athlete_ids[1]]['Time'], rider_dfs_t[athlete_ids[1]]['Power'], label='Raw Power', linestyle='-', linewidth=1, color='blue')
ax2.plot(rider_dfs_t[athlete_ids[1]]['Time'], rider_dfs_t[athlete_ids[1]]['Power_Filt'], label='Filtered Power', linestyle='-', linewidth=1, color='red')
ax2.plot(rider_dfs_t[athlete_ids[1]]['Time'], rider_dfs_t[athlete_ids[1]]['Power_MA_10s'], label='10s MA', linestyle='--', linewidth=1, color='green')

ax3.plot(rider_dfs_t[athlete_ids[2]]['Time'], rider_dfs_t[athlete_ids[2]]['Power'], label='Raw Power', linestyle='-', linewidth=1, color='blue')
ax3.plot(rider_dfs_t[athlete_ids[2]]['Time'], rider_dfs_t[athlete_ids[2]]['Power_Filt'], label='Filtered Power', linestyle='-', linewidth=1, color='red')
ax3.plot(rider_dfs_t[athlete_ids[2]]['Time'], rider_dfs_t[athlete_ids[2]]['Power_MA_10s'], label='10s MA', linestyle='--', linewidth=1, color='green')

ax4.plot(rider_dfs_t[athlete_ids[3]]['Time'], rider_dfs_t[athlete_ids[3]]['Power'], label='Raw Power', linestyle='-', linewidth=1, color='blue')
ax4.plot(rider_dfs_t[athlete_ids[3]]['Time'], rider_dfs_t[athlete_ids[3]]['Power_Filt'], label='Filtered Power', linestyle='-', linewidth=1, color='red')
ax4.plot(rider_dfs_t[athlete_ids[3]]['Time'], rider_dfs_t[athlete_ids[3]]['Power_MA_10s'], label='10s MA', linestyle='--', linewidth=1, color='green')

ax1.set_title('Athlete 1'), ax2.set_title('Athlete 2'), ax3.set_title('Athlete 3'), ax4.set_title('Athlete 4')
ax1.set_ylabel('Power (Watts)'), ax2.set_ylabel('Power (Watts)'), ax3.set_ylabel('Power (Watts)'), ax4.set_ylabel('Power (Watts)')
ax1.grid(True), ax2.grid(True), ax3.grid(True), ax4.grid(True)

# Plot the video
ax5.axis('off')

from detection_analysis import VideoPlotterGeneric

vidplotter = VideoPlotterGeneric(fig, [ax1, ax2, ax3, ax4], ax5, 'media\C1319.MP4')

vidplotter.show()

# %% Output Plot
plt.figure(figsize=(12, 6))
colors = plt.get_cmap('tab10', len(rider_dfs_t))  # Use a colormap to get distinct colors

for idx, (athlete_id, df) in enumerate(rider_dfs_t.items()):
    color = colors(idx)
    plt.plot(df['Time'], df['Power'], label=f'Athlete {athlete_id}', linestyle='-', linewidth=0.5, color=color)
    plt.plot(df['Time'], df['Power_MA_10s'], label=f'Athlete {athlete_id} (10s MA)', linestyle='--', linewidth=0.2, color=color)
    plt.plot(df['Time'], df['Power_Filt'], label=f'Athlete {athlete_id} (BW Filt)', linestyle='-', linewidth=2, color=color)
    for co in changeover_times:
        plt.axvline(x=co['start_time'], color='green', linestyle='--')
        plt.axvline(x=co['start_time']+CO_ANALYSIS_WINDOW_POST_START, color='red', linestyle='--')

plt.xlabel('Time')
plt.ylabel('Power (Watts)')
plt.title('Comparison of Power Outputs Between Riders')
plt.legend()
plt.grid(True)

# %% Output Plot with Video

fig = plt.figure(figsize=(12, 6))
fig.suptitle('Comparison of Power Outputs Between Riders')

colors = plt.get_cmap('tab10', len(rider_dfs_t))  # Use a colormap to get distinct colors

gs = fig.add_gridspec(1, 3)

ax1 = fig.add_subplot(gs[0, :-1])
ax2 = fig.add_subplot(gs[0, 2])

for idx, (athlete_id, df) in enumerate(rider_dfs_t.items()):
    color = colors(idx)
    ax1.plot(df['Time'], df['Power'], label=f'Athlete {athlete_id}', linestyle='-', linewidth=0.5, color=color)
    ax1.plot(df['Time'], df['Power_MA_10s'], label=f'Athlete {athlete_id} (10s MA)', linestyle='--', linewidth=0.2, color=color)
    ax1.plot(df['Time'], df['Power_Filt'], label=f'Athlete {athlete_id} (BW Filt)', linestyle='-', linewidth=2, color=color)
    for co in changeover_times:
        ax1.axvline(x=co['start_time'], color='green', linestyle='--')
        ax1.axvline(x=co['start_time']+CO_ANALYSIS_WINDOW_POST_START, color='red', linestyle='--')

ax1.set_xlabel('Time')
ax1.set_ylabel('Power (Watts)')
ax1.legend()
ax1.grid(True)

vidplotter = VideoPlotterGeneric(fig, [ax1], ax2, 'media\C1319.MP4')

vidplotter.show()

