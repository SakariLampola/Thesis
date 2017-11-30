# -*- coding: utf-8 -*-
"""
Analyze video frame by frame detecting objects and estimating states

USAGE
python AnalyzeVideo.py --video videos/CarsOnHighway001.mp4

Created on Wed Nov 29 13:44:02 2017
@author: Sakari Lampola
"""

import argparse
import cv2
import ImageObjectDetection as iod

def analyze_video(videofile):
    """
    Analyzes a video frame by frame, detecting objects with locations
    """
    cap = cv2.VideoCapture(videofile)
    i_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("Frame ", i_frame)
            i_frame = i_frame + 1
            detected_objects = iod.detect_objects(frame, 0.2)
            for detected_object in detected_objects:
                # display the prediction
                label = "{}: {:.2f}%".format(detected_object.name, \
                         detected_object.confidence * 100)
                print("--- {}".format(label))
                cv2.rectangle(frame, (detected_object.x_min, detected_object.y_min), \
                              (detected_object.x_max, detected_object.y_max), 255, 2)
                ytext = detected_object.y_min - 15 if detected_object.y_min - 15 > 15 \
                    else detected_object.y_min + 15
                cv2.putText(frame, label, (detected_object.x_min, ytext), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # parse the video file
    AP = argparse.ArgumentParser()
    AP.add_argument("-v", "--video", required=True, help="path to input video")
    ARGS = vars(AP.parse_args())
    VIDEOFILE = ARGS["video"]
    # spawn the analyzer
    analyze_video(VIDEOFILE)
    