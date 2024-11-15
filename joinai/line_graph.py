import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import threading
from flask import Flask, render_template, Response

from main import csv_file_path

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Convert the 'Timestamp' to a datetime object
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Set the 'Timestamp' column as the index (optional, for better plotting)
data.set_index('Timestamp', inplace=True)

# Function to plot and save the graph
def plot_and_save_graph():
    plt.figure(figsize=(12, 8))

    # Plotting each feed
    plt.plot(data.index, data['Feed 1 Count'], marker='o', label='Feed 1', color='blue')
    plt.plot(data.index, data['Feed 2 Count'], marker='o', label='Feed 2', color='orange')
    plt.plot(data.index, data['Feed 3 Count'], marker='o', label='Feed 3', color='green')
    plt.plot(data.index, data['Feed 4 Count'], marker='o', label='Feed 4', color='red')

    # Labeling the graph
    plt.xlabel('Timestamp')
    plt.ylabel('Number of People Counted')
    plt.title('People Count from Multiple Feeds Over Time')

    # Formatting the x-axis to display timestamps clearly
    plt.xticks(rotation=45, ha='right')

    # Adding a legend to differentiate the feeds
    plt.legend()

    # Save the graph as a PNG file with a timestamp in the filename
    snapshot_filename = f'people_count_snapshot_{int(time.time())}.png'
    plt.tight_layout()
    plt.savefig(snapshot_filename)
    plt.close()  # Close the figure to free up memory

    print(f"Graph saved as {snapshot_filename}")

# Interval in seconds for saving snapshots
interval_seconds = 30  # Set your desired interval

# Loop to continuously plot and save the graph
try:
    while True:
        plot_and_save_graph()
        time.sleep(interval_seconds)  # Wait for the specified interval
except KeyboardInterrupt:
    print("Snapshot saving stopped.")
