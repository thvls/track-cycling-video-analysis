## Requirements

- Python 3.11+
- Other dependencies listed in `requirements.txt`

## Installation

1. Ensure you have Python 3.11 or higher installed.
2. Clone the repository:
    ```sh
    git clone https://github.com/thvls/track-cycling-video-analysis
    cd track-cycling-video-analysis
    ```
3. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run main.py
You will be prompted to enter the path to the video file. If no path is provided, a sample video will be used.
Workflow:
- Load video file
- Run object detection
- Load resulting tracking data
- Preprocess tracking data to improve changeover detection consistency: merge (clearly distinguishable) track ID, remove outliers, fill discontinuities
- Detect changeovers
- Plot and save changeover detection results
- Extract changeover clips