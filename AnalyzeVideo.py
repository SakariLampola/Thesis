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
import ImageClasses as ic
import ObjectDetection as od

def analyze_video(videofile):
    """
    Analyzes a video frame by frame, detecting objects with locations
    """
    # create an empty world
    world = ic.ImageWorld()
    time = 0.0
    # open the video
    video = cv2.VideoCapture(videofile)
    fps = video.get(cv2.CAP_PROP_FPS)
    time_step = 1.0 / fps
    i_frame = 1
    # loop every frame in history
    while video.isOpened():
        ret, frame = video.read()
        frame_copy = frame
        if ret:
            time = time + time_step
            print("Time {0:<.2f}".format(time))
            # make a measurement by detecting objects
            detected_objects = od.detect_objects(frame, 0.0)
            # display measurements
            for detected_object in detected_objects:
                print("---{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}".format(
                    detected_object.name, detected_object.confidence,
                    detected_object.x_min, detected_object.x_max,
                    detected_object.y_min, detected_object.y_max))
                label = "{}: {:.2f}%".format(detected_object.name, \
                         detected_object.confidence * 100)
                cv2.rectangle(frame, (detected_object.x_min, detected_object.y_min), \
                              (detected_object.x_max, detected_object.y_max), 255, 2)
                ytext = detected_object.y_min - 15 if detected_object.y_min - 15 > 15 \
                    else detected_object.y_min + 15
                cv2.putText(frame, label, (detected_object.x_min, ytext), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.imshow('Measurements', frame)
            # update the world model
            world.update(time, detected_objects)
            # display the world with filtered objects
            for detected_object in detected_objects:
                label = "{}: {:.2f}%".format(detected_object.name, \
                         detected_object.confidence * 100)
                cv2.rectangle(frame_copy, (detected_object.x_min, detected_object.y_min), \
                              (detected_object.x_max, detected_object.y_max), 255, 2)
                ytext = detected_object.y_min - 15 if detected_object.y_min - 15 > 15 \
                    else detected_object.y_min + 15
                cv2.putText(frame_copy, label, (detected_object.x_min, ytext), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.imshow('Model', frame_copy)
            i_frame = i_frame + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # parse the video file
    AP = argparse.ArgumentParser()
    AP.add_argument("-v", "--video", required=True, help="path to input video")
    ARGS = vars(AP.parse_args())
    VIDEOFILE = ARGS["video"]
    # spawn the analyzer
    analyze_video(VIDEOFILE)
    