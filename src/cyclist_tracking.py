from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import json
from tqdm import tqdm # Progress bar

def run_cyclist_tracking(vid_file=None, model_id='yolov8s-trackcycling-04.pt', model_mode='predict'):
    """
    Tracks cyclists in a video using YOLOv8 for object detection and BotSort for tracking.
    
    Prompts the user for the video file path and the mode (prediction or tracking).
    Outputs an annotated video and a JSON file with tracking data.
    
    Inputs:
    - vid_file: Path to the input video file.
    - model_mode: Mode selection for prediction ('predict') or tracking ('track').
    
    Outputs:
    - vid_out: Annotated video file saved in the 'output' directory.
    - json_out: JSON file with detection/tracking results.
    """

    if vid_file == None:
        vid_file = 'media/sample.mp4'
        print(f"Video used: {vid_file}")
    vid_id = os.path.splitext(os.path.basename(vid_file))[0]

    cap = cv2.VideoCapture(vid_file)

    vid_width = int(cap.get(3))
    vid_height = int(cap.get(4))
    vid_size = (vid_width, vid_height)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    # Load model
    model_files = [f for f in os.listdir('src/') if f.endswith('.pt')]
    print(f"Model files found: {model_files}")

    # Models are found in subfolders src/
    model_id = f"src/{model_id}"

    # If model doesn't exist, throw error
    if not os.path.exists(model_id):
        raise FileNotFoundError(f"Model '{model_id}' not found. Please upload it to the current directory.")

    print(f"Model used: {model_id}")

    model_name = os.path.splitext(model_id)[0] # Remove extension
    model_name = model_name.split('/')[1] # Remove 'src/' from model name

    model = YOLO(model_id)

    # Get current datetime to save in video file name
    now = datetime.now()
    now_string = now.strftime("%Y%m%d_%H%M%S")

    # Process video
    # Make 'output' directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    vid_out = f"output/{vid_id}_{now_string}_{model_name}.mp4"
    json_out = vid_out.replace('.mp4', '.json')
    print(f"Output video: {vid_out}")
    print(f"Output JSON: {json_out}")

    results_list = [] # json output

    # Video object to write output
    out = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc('M','J','P','G'), vid_fps, vid_size)

    # Progress bar
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Processing Video')

    # Process each frame of the video
    frame_count = 0
    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Prepare results for saving
        frame_results = {
            'frame_id': frame_count,
            'detections': []
        }

        # Run the model
        if model_mode == 'predict':
            results = model.predict(frame, conf=0.25, classes=[0])
            boxes = results[0].boxes.xywh.cpu()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf.item()
                    class_id = box.cls.item()
                    detection = {
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    }
                    frame_results['detections'].append(detection)

        elif model_mode == 'track':
            results = model.track(frame, tracker='botsort.yaml', classes=[0], persist=True, show=True)
            boxes = results[0].boxes.xywh.cpu()
            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            except:
                track_ids = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf.item()
                    class_id = box.cls.item()
                    try:
                        track_id = box.id.int().cpu().tolist()
                    except:
                        track_id = []
                    detection = {
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'track_id': track_id
                    }
                    frame_results['detections'].append(detection)
        else:
            raise ValueError("Invalid model mode. Please choose 'predict' or 'track'.")
        results_list.append(frame_results)

        # Display the annotated frame
        annotated_frame = results[0].plot()
        
        # Write to video file
        out.write(annotated_frame)

        frame_count += 1
        pbar.update(1)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video file
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save results to JSON file
    with open(json_out, 'w') as f:
        json.dump(results_list, f, indent=4)

    print(f"Results saved to {json_out}")
    
    return vid_out, json_out

if __name__ == '__main__':
    run_cyclist_tracking()