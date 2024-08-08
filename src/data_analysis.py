import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the JSON data
file_path = 'data/pursuit_times.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract power sensor data
power_data = []
for entry in data:
    if 'sensorData' in entry:
        sensor_data = entry['sensorData']
        for sensor_id, sensor_records in sensor_data.items():
            for record in sensor_records:
                if record.get('sensorValueName') == 'AveragePower':
                    timestamp = record.get('timestamp')
                    power_value = record.get('sensorValue')
                    power_data.append((timestamp, power_value))

# Convert to DataFrame for plotting
df_power = pd.DataFrame(power_data, columns=['Timestamp', 'Power'])
df_power['Timestamp'] = pd.to_datetime(df_power['Timestamp'])

# Plotting power data over time
plt.figure(figsize=(12, 6))
plt.plot(df_power['Timestamp'], df_power['Power'], label='Power')
plt.xlabel('Time')
plt.ylabel('Power (Watts)')
plt.title('Power Output Over Time')
plt.legend()
plt.grid(True)
plt.show()
