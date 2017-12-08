# -*- coding: utf-8 -*-
"""
Analyze video frame by frame detecting objects and estimating states

USAGE
python AnalyzeVideo.py --video videos/CarsOnHighway001.mp4

Created on Wed Nov 29 13:44:02 2017
@author: Sakari Lampola
"""

import time
import argparse
import cv2
import ImageClasses as ic
import ObjectDetection as od

def analyze_video(videofile):
    """
    Analyzes a video frame by frame, detecting objects with locations
    """
    # open log files
    log_file = open("log.txt", "w")
    trace_file = open("trace.txt", "w")
    trace_file.write("time,zx,zy,x,y,vx,vy,ax,ay,zxs,zys,xs,ys,vsx,vsy,asx,asy\n")
    # create an empty world
    world = ic.ImageWorld()
    current_time = 0.0
    # open the video
    video = cv2.VideoCapture(videofile)
    fps = video.get(cv2.CAP_PROP_FPS)
    time_step = 1.0 / fps
    i_frame = 1
    # loop every frame in history
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame_copy = frame.copy()

            log_file.write("----------------------------------------------\n")
            log_file.write("Time {0:<.2f}\n".format(current_time))
            log_file.write("Measurements:\n")

            print("----------------------------------------------")
            print("Time {0:<.2f}".format(current_time))
            print("Measurements:")

            # make a measurement by detecting objects
            detected_objects = od.detect_objects(frame, 0.2)
            
            # display measurements
            for detected_object in detected_objects:
                x_min, x_max, y_min, y_max = detected_object.bounding_box()

                log_file.write("---{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}\n".format(
                    ic.CLASS_NAMES[detected_object.idx], detected_object.confidence,
                    x_min, x_max, y_min, y_max))

                print("---{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}".format(
                    ic.CLASS_NAMES[detected_object.idx], detected_object.confidence,
                    x_min, x_max, y_min, y_max))

                label = "{}: {:.2f}%".format(ic.CLASS_NAMES[detected_object.idx], \
                         detected_object.confidence * 100)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 255, 2)
                ytext = y_min - 15 if y_min - 15 > 15 else y_min + 15
                cv2.putText(frame, label, (x_min, ytext), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.imshow('Measurements', frame)

            # update the world model
            world.update(current_time, detected_objects, log_file, trace_file)

            # display the world with filtered objects
            log_file.write("World:\n")
            print("World:")

            for world_object in world.world_objects:
                x_min, x_max, y_min, y_max = world_object.bounding_box()

                log_file.write("---{0:d} {1:s} {2:6.2f} {3:4d} {4:4d} {5:4d} {6:4d}\n".format(
                    world_object.id, world_object.name, world_object.confidence,
                    x_min, x_max, y_min, y_max))

                print("---{0:d} {1:s} {2:6.2f} {3:4d} {4:4d} {5:4d} {6:4d}".format(
                    world_object.id, world_object.name, world_object.confidence,
                    x_min, x_max, y_min, y_max))

                label = "{0:d} {1:s}: {2:.2f}%".format(world_object.id, world_object.name, \
                         world_object.confidence * 100)
                cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), 255, 2)
                ytext = y_min - 15 if y_min - 15 > 15 else y_min + 15
                cv2.putText(frame_copy, label, (x_min, ytext), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.imshow('Model', frame_copy)
            i_frame = i_frame + 1
            current_time = current_time + time_step

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(1.0)

    video.release()
    cv2.destroyAllWindows()
    log_file.close()
    trace_file.close()

if __name__ == "__main__":
    # parse the video file
    AP = argparse.ArgumentParser()
    AP.add_argument("-v", "--video", required=True, help="path to input video")
    ARGS = vars(AP.parse_args())
    VIDEOFILE = ARGS["video"]
    # spawn the analyzer
    analyze_video(VIDEOFILE)
    