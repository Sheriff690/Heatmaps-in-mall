import os
import csv
from datetime import datetime
from threading import Timer

import pdfkit
import cv2
from ultralytics import YOLO
from tracker import Tracker
import cvzone
from flask import Flask, render_template, Response, request, send_file, redirect, url_for, flash
from flask_socketio import SocketIO, emit
import numpy as np
from scipy.ndimage import gaussian_filter
import threading
import time
import matplotlib.pyplot as plt
import io
from weasyprint import HTML
import base64

app = Flask(__name__)
socketio = SocketIO(app)
model = YOLO('yolov8s.pt')  # Choose a smaller model if needed for faster inference

HEATMAPS_DIR = "static/heatmaps/daily"
RECORDINGS_DIR = "static/recordings/daily"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
dwell_times = {}
person_timestamps = {}

# Define the line for entry/exit
LINE_Y = 200  # Y-coordinate of the horizontal line
LINE_COLOR = (0, 255, 0)  # Green line color
LINE_THICKNESS = 2


# Read class names from file
try:
    with open("coco.txt", "r") as my_file:
        data = my_file.read()
    class_list = data.split("\n")
except FileNotFoundError:
    print("Error: coco.txt file not found.")
    exit()


# Placeholder variables for the models and trackers
tracker = Tracker()
person_coordinates = [[] for _ in range(4)]  # List for storing coordinates for each feed
heatmap_imgs = [None] * 4  # List to hold heatmap images for four feeds
person_count = [0] * 4


from PIL import Image
def generate_heatmap(feed_index):
    global person_coordinates, heatmap_imgs

    if heatmap_imgs[feed_index] is not None:
        print(f"Heatmap for feed {feed_index} is available.")

        # Ensure heatmap is a valid NumPy array before returning
        if isinstance(heatmap_imgs[feed_index], np.ndarray):
            return heatmap_imgs[feed_index]  # Return the NumPy array
        else:
            print(f"Error: Heatmap for feed {feed_index} is not a NumPy array.")
            return None
    else:
        print(f"No heatmap image available for feed {feed_index}.")
        return None


# # Function to save heatmap
# def save_heatmap(heatmap_image, feed_index, frame_number):
#     file_path = f'heatmaps/feed_{feed_index}_frame_{frame_number}.png'  # Example file path
#     heatmap_image.save(file_path)  # Save the image
#     print(f"Heatmap saved to {file_path}.")  # Optional: Log the saving



# This global dictionary will store entry and exit times for tracked persons
tracked_people = {}
dwell_times = []
last_save_time = 0

def process_frame(frame, model, tracker, class_list, person_coordinates, feed_index, frame_number):
    global person_count, last_save_time
    # Labels for each feed (adjust as per your feed sources)
    camera_labels = ['LG Camera 1', 'LG Camera 2', 'UG Camera 1', 'UG Camera 2']

    results = model.predict(frame)  # Get predictions from the model
    boxes = results[0].boxes.data  # Get bounding boxes (x1, y1, x2, y2)

    detected_persons = []
    # Reset the person count for the current feed before processing
    person_count[feed_index] = 0
    new_person_coords = []  # Temporary list for coordinates of the current feed

    person_coordinates[feed_index] = new_person_coords
    # Track current time for dwell time calculation
    current_time = time.time()

    for box in boxes:  # Loop through detected boxes
        x1, y1, x2, y2, conf, class_id = box  # Unpack the box
        if class_list[int(class_id)] == 'person':  # Check if detected class is 'person'
            detected_persons.append([int(x1), int(y1), int(x2), int(y2)])  # Append bounding box as integer
            cx = (int(x1) + int(x2)) // 2  # Calculate center x
            cy = (int(y1) + int(y2)) // 2  # Calculate center y
            new_person_coords.append((cx, cy))  # Store coordinates for this feed
            person_count[feed_index] += 1  # Increment the count for this feed

    # Update the list of person coordinates for this feed
    person_coordinates[feed_index] = new_person_coords

    # Use tracker to update and get IDs for detected persons
    bbox_id = tracker.update(detected_persons)  # Update tracker with detected persons
    current_detected_ids = []  # Keep track of currently detected IDs

    # for bbox in bbox_id:
    #     x3, y3, x4, y4, id = bbox  # Get person tracking ID
    #     current_detected_ids.append(id)  # Append current ID to list of detected IDs
    #
    #     # If this ID is new, record entry time
    #     if id not in tracked_people:
    #         tracked_people[id] = {'entry_time': current_time}
    #
    #     # Draw the bounding box and ID
    #     cx = (x3 + x4) // 2
    #     cy = (y3 + y4) // 2
    #     cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
    #     cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
    #     cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox  # Get person tracking ID
        current_detected_ids.append(id)  # Append current ID to list of detected IDs

        # If this ID is new, record entry time
        if id not in tracked_people:
            tracked_people[id] = {'entry_time': current_time}

        # Draw the bounding box and ID
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)  # Draw a circle at the center of the detected person
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Draw bounding box

        # Draw a smaller ID tag
        cv2.putText(frame, f'{id}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


    # Check for persons who are no longer detected and calculate dwell time for them
    to_remove = []
    for person_id in tracked_people.keys():
        if person_id not in current_detected_ids:  # Person has left the frame
            # Log the exit time and calculate the dwell time
            exit_time = current_time
            entry_time = tracked_people[person_id]['entry_time']
            dwell_time = (exit_time - entry_time)

            tracked_people[person_id]['exit_time'] = exit_time
            tracked_people[person_id]['dwell_time'] = dwell_time

            # Append to the list of dwell times
            dwell_times.append(dwell_time)
            # Mark for removal from tracking dictionary (since they've left)
            to_remove.append(person_id)

    # Remove persons who have exited
    for person_id in to_remove:
        del tracked_people[person_id]

    # Emit total visitors count after updating the current feed
    total_visitors = sum(person_count)
    socketio.emit('update_visitor_count', {'total_visitors': total_visitors})

    # Calculate average dwell time
    average_dwell_time = calculate_average_dwell_time()
    socketio.emit('update_average_dwell_time', {'average_dwell_time': average_dwell_time})

    # Generate the heatmap for this feed
    heatmap = generate_heatmap(feed_index)  # Get the heatmap image
    if heatmap is not None and heatmap.any():  # Check if the heatmap was generated
        current_time = time.time()

        # Check if 20 seconds have passed since the last save
        if current_time - last_save_time >= 20:
            save_heatmap(heatmap, feed_index, frame_number)  # Save the heatmap image
            last_save_time = current_time  # Update last save time

    # Add green label for the current camera feed
    cv2.putText(frame, camera_labels[feed_index], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def calculate_average_dwell_time():
    if len(dwell_times) == 0:  # No dwell times available yet
        return 0.0  # Return 0 if no dwell times have been recorded

    # Calculate the sum of all dwell times and divide by the number of entries
    total_dwell_time = sum(dwell_times)
    average_dwell_time = total_dwell_time / len(dwell_times)

    # Ensure the average is not zero by applying some minimum threshold for display
    average_dwell_time = round(average_dwell_time, 6)  # Show 6 decimal places for small values
    # print(average_dwell_time)
    return average_dwell_time  # Return the dwell time without rounding too early

# Global dictionary to track the time each person is first and last seen

def generate_frames(video_sources):
    frame_number = 0

    # Define the codec and create VideoWriter objects for saving the video streams
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_files = [cv2.VideoWriter(os.path.join(RECORDINGS_DIR, f'output_feed_{i}.mp4'), fourcc, 20.0, (510, 250)) for i in range(len(video_sources))]

    # Open video sources
    caps = [cv2.VideoCapture(source) for source in video_sources]
    if not all(cap.isOpened() for cap in caps):
        print("Error: Could not open video file(s).")
        exit()

    while True:
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (510, 250))  # Resize frame to fit in grid
            processed_frame = process_frame(frame, model, tracker, class_list, person_coordinates, i, frame_number)

            # Write the processed frame to the corresponding output file
            out_files[i].write(processed_frame)
            frames.append(processed_frame)
        frame_number += 1

        if len(frames) < 4:
            break  # If not enough frames, exit the loop

        # Create a 2x2 grid of the frames
        top_row = cv2.hconcat(frames[:2])
        bottom_row = cv2.hconcat(frames[2:])
        composite_frame = cv2.vconcat([top_row, bottom_row])

        # Encode the composite frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', composite_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the VideoWriter objects
    for out in out_files:
        out.release()

    # Release the video captures
    for cap in caps:
        cap.release()

# def generate_frames(video_sources, feed_index):
#     frame_number = 0
#     # Open the video source for the specific feed index
#     cap = cv2.VideoCapture(video_sources[feed_index])
#
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_sources[feed_index]}.")
#         exit()
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = cv2.resize(frame, (510, 250))  # Resize to fit in the layout
#         # Process the frame (person detection, tracking, etc.)
#         processed_frame = process_frame(frame, model, tracker, class_list, person_coordinates, feed_index, frame_number)
#         frame_number += 1
#
#         # Encode frame to JPEG format
#         ret, buffer = cv2.imencode('.jpg', processed_frame)
#         frame = buffer.tobytes()
#
#         # Return the frame for the current feed
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
#     cap.release()



##Creating heatmaps from four feeds
def create_heatmap():
    global heatmap_imgs, person_coordinates
    camera_labels = ['LG Camera 1', 'LG Camera 2', 'UG Camera 1', 'UG Camera 2']

    while True:
        time.sleep(5)  # Update heatmap every few seconds
        for i in range(4):
            if person_coordinates[i]:  # Check if there are coordinates for the current feed
                print(f"Creating heatmap for feed {i}...")  # Debug print
                heatmap_data = np.zeros((500, 1020))  # Assuming heatmap size is 500x1020

                # Accumulate coordinates for the current feed
                for (x, y) in person_coordinates[i]:
                    if x < 1020 and y < 500:
                        heatmap_data[y][x] += 1

                # Apply Gaussian filter to smooth the heatmap
                heatmap_data = gaussian_filter(heatmap_data, sigma=30)

                # Normalize the data to a range of 0-255
                normalized_heatmap = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
                normalized_heatmap = np.uint8(normalized_heatmap)

                # Apply the color map without any conversion
                heatmap_imgs[i] = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

                # Add green label for the current heatmap
                cv2.putText(heatmap_imgs[i], camera_labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Reset coordinates for the next heatmap generation
                person_coordinates[i] = []  # Clear the list for the next frame


wkhtmltopdf_path = r'C:\Program Files\wkhtmltopdf\bin'

if not os.path.exists(HEATMAPS_DIR):
    os.makedirs(HEATMAPS_DIR)


def save_heatmap(heatmap, feed_name, frame_number):
    # Check if the heatmap is a valid NumPy array before proceeding
    if isinstance(heatmap, np.ndarray):
        # Convert from OpenCV's default BGR format to RGB before saving, as PIL expects RGB
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Convert to a PIL image
        heatmap_image = Image.fromarray(heatmap_rgb)

        # Construct a file name based on feed name, frame number, and current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{feed_name}_heatmap_{frame_number}_{timestamp}.png"

        # Define the full path to save the heatmap
        file_path = os.path.join(HEATMAPS_DIR, file_name)

        # Save the heatmap as an image
        heatmap_image.save(file_path)
        print(f"Saved heatmap for {feed_name} at frame {frame_number} to {file_path}")
    else:
        print(f"Error: Invalid heatmap format for {feed_name} at frame {frame_number}")


# def save_bar_graph(person_count, output_file="static/graphs/bar_graph.png"):
#     camera_labels = ['LG Camera 1', 'LG Camera 2', 'UG Camera 1', 'UG Camera 2']  # Adjust labels as per your feeds
#     plt.figure(figsize=(10, 6))  # Set the figure size
#
#     # Create a bar chart
#     plt.bar(camera_labels, person_count, color=['blue', 'green', 'red', 'purple'])
#
#     plt.title('People Detected by Camera Feeds')
#     plt.xlabel('Camera Feed')
#     plt.ylabel('Number of People Detected')
#
#     # Save the graph to a file
#     plt.savefig(output_file)
#     plt.close()  # Close the plot to free up memory
#     print(f"Bar graph saved to {output_file}")

def save_bar_graph(person_count):
    # Create a bar graph using Matplotlib
    labels = ['LG Camera 1', 'LG Camera 2', 'UG Camera 1', 'UG Camera 2']
    fig, ax = plt.subplots()
    ax.bar(labels, person_count, color=['blue', 'green', 'red', 'purple'])

    # Labeling the graph
    ax.set_xlabel('Camera Feeds')
    ax.set_ylabel('People Count')
    ax.set_title('People Count by Camera Feeds')

    # Define the path to save the image
    graph_dir = os.path.join(app.static_folder, 'graphs')
    os.makedirs(graph_dir, exist_ok=True)  # Create directory if it doesn't exist

    graph_path = os.path.join(graph_dir, 'people_count_snapshot.png')

    # Save the plot as a PNG image
    plt.savefig(graph_path)
    plt.close()  # Close the plot to free memory

    # Return the path of the saved image
    return graph_path

@app.route('/')
def index():
    global dwell_times
    if len(dwell_times) > 0:
        average_dwell_time = sum(dwell_times) / len(dwell_times)
    else:
        average_dwell_time = 0
    # Dynamically calculate total visitors
    total_visitors = sum(person_count)
    return render_template('index.html', total_visitors=total_visitors, average_dwell_time=average_dwell_time)

# @socketio.on('connect')
# def handle_connect():
#     # Send current visitor count when the client connects
#     total_visitors = sum(person_count)
#     emit('update_count', {'total_visitors': total_visitors})

@socketio.on('connect')
def handle_connect():
    global dwell_times
    # Send current visitor count when the client connects
    total_visitors = sum(person_count)
    print(f"Total visitors to send: {total_visitors}")  # Debug line
    emit('update_count', {'total_visitors': total_visitors})

    # Calculate average dwell time and emit it
    if len(dwell_times) > 0:
        average_dwell_time = sum(dwell_times) / len(dwell_times)
    else:
        average_dwell_time = 0  # Default to 0 if no dwell times are recorded
    print(f"Average dwell time to send: {average_dwell_time}")  # Debug line
    emit('update_dwell_time', {'average_dwell_time': average_dwell_time})


@app.route('/video_feed')
def video_feed():
    video_sources = ['vidp1.mp4', 'vidp4.mp4', 'vidp2.mp4', 'vidp3.mp4']  # List of video files
    return Response(generate_frames(video_sources), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed/<int:feed_index>')
# def video_feed(feed_index):
#     video_sources = ['vidp1.mp4','vidp2.mp4','vidp3.mp4', 'vidp4.mp4' ]  # List of video files
#     return Response(generate_frames(video_sources, feed_index), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/export_report')
def export_report():
    with app.app_context():
        total_visitors = sum(person_count)
        image_path = os.path.join(app.static_folder, 'graphs', 'people_count_snapshot_1728631081.png')

        # Generate the bar graph and get its file path
        bar_graph_path = save_bar_graph(person_count)
        formatted_lineGraph_path = f"file:///{os.path.abspath(image_path).replace('\\', '/')}"

        # Ensure the path is not None and file exists
        if bar_graph_path and os.path.exists(bar_graph_path):
            formatted_image_path = f"file:///{os.path.abspath(bar_graph_path).replace('\\', '/')}"
        else:
            formatted_image_path = None

        # Heatmap path (for demonstration purposes, ensure the file exists)
        heatmap_image_path = os.path.join(app.static_folder, 'heatmaps/daily', '0_heatmap_7_20241016_152444.png')
        heatmap_image_path_formatted = f"file:///{os.path.abspath(heatmap_image_path).replace('\\', '/')}"

        goals = [
            {"name": "Increase Foot Traffic", "status": "On Track", "progress": "70%"},
            {"name": "Improve Customer Satisfaction", "status": "At Risk", "progress": "45%"},
            {"name": "Boost Sales", "status": "Completed", "progress": "100%"},
            {"name": "Enhance Marketing Campaign", "status": "Not Started", "progress": "0%"},
        ]

        # Generate HTML content for the report with Bootstrap styling
        report_html = f"""
        <html>
        <head>
            <title>Business Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="container">
            <h1 class="text-center my-4">Business Report</h1> <!-- Centered heading -->

            <h2>Business Insights</h2>
            <table class="table table-striped table-bordered">
                <thead class="table-light">
                    <tr>
                        <th>Goal</th>
                        <th>Status</th>
                        <th>Progress</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Append goals to the HTML
        for goal in goals:
            report_html += f"""
                <tr>
                    <td>{goal['name']}</td>
                    <td>{goal['status']}</td>
                    <td>{goal['progress']}</td>
                </tr>
            """

        report_html += """
                </tbody>
            </table>

            <!-- Bootstrap grid for side-by-side graphs -->
            <div class="row my-4">
                <div class="col-md-6"> <!-- Bar Graph -->
                    <h2>People Count by Camera Feeds</h2>
        """

        # Conditionally add the bar graph image if it exists
        if formatted_image_path:
            report_html += f"""
                    <img src="{formatted_image_path}" alt="bar-graph" class="img-fluid"/>
                </div>
                <div class="col-md-6"> <!-- Line Graph -->
                    <h2>Line Graph of the Four Feeds</h2>
                    <p class="lead">
                        This shows the number of people detected based on time frames. At each time instance, the total number 
                        of people counted from each camera feed is plotted. The data is taken from a csv file, which involves 
                        saving the total number of people from each feed at regular time intervals.
                    </p>
                    <img src="{formatted_lineGraph_path}" alt="line-graph" class="img-fluid"/>
                </div>
            </div> <!-- End of row -->
            """

        # Total visitors count placed just below the graphs
        report_html += f"""
            <div class="text-center">
                <h3>Total Visitors: {total_visitors}</h3>
            </div>

            <h2>Heatmaps</h2>
            <img src="{heatmap_image_path_formatted}" alt="heatmap" class="img-fluid my-4"/>
        </body>
        </html>
        """

        # PDFKit configuration and options
        config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
        options = {
            'enable-local-file-access': '',
        }

        # Generate the PDF from the HTML string
        pdf_file = 'business_report.pdf'
        pdfkit.from_string(report_html, pdf_file, configuration=config, options=options)

        # Send the generated PDF to the user
        response = send_file(pdf_file, as_attachment=True)

        return response



# @app.route('/heatmap_feed')
# def heatmap_feed():
#     return Response(generate_heatmap(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/heatmap_feed/<int:feed_index>')
def heatmap_feed(feed_index):
    def generate_heatmap():
        global heatmap_imgs
        while True:
            if heatmap_imgs[feed_index] is not None:
                ret, buffer = cv2.imencode('.jpg', heatmap_imgs[feed_index])
                heatmap_frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + heatmap_frame + b'\r\n')
            time.sleep(0.1)

    return Response(generate_heatmap(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route to handle report generation
def get_latest_heatmap():
    # Get the list of files in the heatmaps directory
    heatmap_files = [os.path.join(HEATMAPS_DIR, f) for f in os.listdir(HEATMAPS_DIR) if f.endswith('.jpg') or f.endswith('.png')]

    # Sort the files by modification time in descending order
    if heatmap_files:
        latest_heatmap = max(heatmap_files, key=os.path.getmtime)
        return latest_heatmap
    else:
        return None


# Define the path to the CSV file
csv_file_path = 'person_count_data.csv'

# Function to write person counts to a CSV file
def write_to_csv():
    global person_count

    # Get the current timestamp
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')

    # Check if all counts are zero
    if all(count == 0 for count in person_count):
        print(f"Skipping CSV write at {current_time} as all counts are zero.")
        return  # Skip writing if all counts are zero

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the timestamp and person counts for all feeds
        writer.writerow([current_time] + person_count)

    print(f"Data written to CSV at {current_time}: {person_count}")

# Function to start the timer and write data every 20 seconds
def start_periodic_csv_write():
    write_to_csv()  # Write the data immediately
    Timer(20, start_periodic_csv_write).start()  # Schedule the next write after 20 seconds

# Create the CSV file and add the headers if it doesn't exist
def create_csv_file():
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers for the CSV file
        headers = ['Timestamp', 'Feed 1 Count', 'Feed 2 Count', 'Feed 3 Count', 'Feed 4 Count']
        writer.writerow(headers)
    print(f"CSV file created with headers at {csv_file_path}")


REPORTS_DIR = "static/reports"  # Directory to save the reports
os.makedirs(REPORTS_DIR, exist_ok=True)

@app.route('/reports')
def view_reports():
    # Get the list of generated reports
    report_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.pdf')]
    return render_template('reports.html', report_files=report_files)

# Route to display generated heatmaps
@app.route('/heatmaps')
def view_heatmaps():
    # Get a list of generated heatmaps
    heatmaps = os.listdir(HEATMAPS_DIR)  # Ensure HEATMAPS_DIR is defined and points to the correct directory
    return render_template('heatmaps.html', heatmaps=heatmaps)

# Route to display the bar graph of people counted in each feed
@app.route('/people_count_graph')
def people_count_graph():
    global person_count
    camera_labels = ['LG Camera 1', 'LG Camera 2', 'UG Camera 1', 'UG Camera 2']

    # Create a bar plot
    fig, ax = plt.subplots()
    ax.bar(camera_labels, person_count, color=['blue', 'orange', 'green', 'red'])

    ax.set_xlabel('Camera Feeds')
    ax.set_ylabel('People Count')
    ax.set_title('Number of People Detected in Each Camera Feed')

    # Save the plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Send the image as a response
    return send_file(img, mimetype='image/png')


if __name__ == "__main__":
    print("Starting Flask app...")
    heatmap_thread = threading.Thread(target=create_heatmap)
    heatmap_thread.daemon = True
    heatmap_thread.start()
    # create_csv_file() #If it does not exist
    # start_periodic_csv_write()
    socketio.run(app,debug=True, allow_unsafe_werkzeug=True)
