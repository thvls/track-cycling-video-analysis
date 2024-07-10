from cyclist_tracking import run_cyclist_detection

if __name__ == '__main__':

    vid = input("Enter the path to the video file: ")

    run_cyclist_detection(vid_file=vid, model_mode='track')