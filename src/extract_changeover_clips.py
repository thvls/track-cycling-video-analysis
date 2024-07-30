import os
import json
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip # REQUIRES SETUPTOOLS 69.0.2 OR EARLIER

# Function to trim video based on start and end frames
def extract_changeover_clips(json_path, video_path, output_path, buffer_before=2, buffer_after=1):
    """
    Extracts video clips based on changeover data.

    Inputs:
    - json_path: Path to JSON file containing changeover data.
    - video_path: Path to video file.
    - output_path: Path to save extracted video clips.
    - buffer_before: Number of seconds to include before the changeover.
    - buffer_after: Number of seconds to include after the changeover.

    Changeover json structure example:
    [
        {
            "changeover_number": 1,
            "data": {
                "start_time": 100,
                "end_time": 200
        }
    ]

    """

    # Create output path if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Iterate over changeovers and trim video
    for changeover in data:
        no = changeover['changeover_number']
        changeover_data = changeover['data']
        start_time = changeover_data['start_time'] - buffer_before
        end_time = changeover_data['end_time'] + buffer_after 
        
        # Trim video and save
        video_name = os.path.basename(video_path)
        # Extract extension
        video_name = os.path.splitext(video_name)[0]

        outname = f"{output_path}/{video_name}_changeover_{no}.mov"
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=outname)

if __name__ == "__main__":
    # Example usage
    json_path = input('Enter path to changeover JSON file: ')
    video_path = input('Enter path to video file: ')
    output_path = 'output/changeovers'
    extract_changeover_clips(json_path, video_path, output_path)
