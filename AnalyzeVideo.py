# -*- coding: utf-8 -*-
"""
Analyze video frame by frame detecting objects and estimating states

USAGE
python AnalyzeVideo.py --video videos/CarsOnHighway001.mp4

Created on Wed Nov 29 13:44:02 2017
@author: Sakari Lampola
"""

import argparse
import time
import cv2
import ImageClasses as ic
import ObjectDetection as od

def analyze_video(videofile):
    """
    Analyzes a video frame by frame, detecting objects with locations
    """
    # Open log and trace files
    log_file = open("log.txt", "w")
    trace_file = open("trace.txt", "w")
    trace_file.write("time,id,x_min_p,x_max_p,y_min_p,y_max_p,")
    trace_file.write("x_min_m,x_max_m,y_min_m,y_max_m\n")

    # Create an empty world
    world = ic.ImageWorld()

    # Open the video and initialize follow-up vaiables
    video = cv2.VideoCapture(videofile)
    fps = video.get(cv2.CAP_PROP_FPS)
    time_step = 1.0 / fps
    current_time = 0.0
    i_frame = 1

    # Loop every frame in the video
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame_copy = frame.copy() # Make a copy to present image world model

            log_file.write("----------------------------------------------\n")
            log_file.write("Time {0:<.2f}\n".format(current_time))
            log_file.write("Object detections:\n")

            print("----------------------------------------------")
            print("Time {0:<.2f}".format(current_time))
            print("Object detections:")

            # Detect objects in the current frame
            detected_objects = od.detect_objects(frame, current_time, 0.2)

            # display detected objects
            for detected_object in detected_objects:

                log_file.write("---{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}\n".format(
                    ic.CLASS_NAMES[detected_object.class_type], detected_object.confidence,
                    detected_object.x_min, detected_object.x_max, \
                    detected_object.y_min, detected_object.y_max))

                print("---{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}".format(
                    ic.CLASS_NAMES[detected_object.class_type], detected_object.confidence, \
                    detected_object.x_min, detected_object.x_max, \
                    detected_object.y_min, detected_object.y_max))

                label = "{}: {:.2f}%".format(ic.CLASS_NAMES[detected_object.class_type], \
                         detected_object.confidence * 100)
                cv2.rectangle(frame, (detected_object.x_min, detected_object.y_min), \
                              (detected_object.x_max, detected_object.y_max), 255, 2)
                ytext = detected_object.y_min - 15 if detected_object.y_min - 15 > 15 \
                    else detected_object.y_min + 15
                cv2.putText(frame, label, (detected_object.x_min, ytext), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.imshow('Object detections', frame)

            # update the world model
            world.update(current_time, detected_objects, log_file, trace_file)

            # display the world with filtered objects
            log_file.write("Image objects:\n")
            print("Image objects:")

            for image_object in world.image_objects:

                log_file.write("---{0:d} {1:s} {2:6.2f} {3:4d} {4:4d} {5:4d} {6:4d}\n".format(
                    image_object.id, image_object.name, image_object.confidence,
                    image_object.x_min, image_object.x_max, \
                    image_object.y_min, image_object.y_max))

                print("---{0:d} {1:s} {2:6.2f} {3:4d} {4:4d} {5:4d} {6:4d}".format(
                    image_object.id, image_object.name, image_object.confidence,
                    image_object.x_min, image_object.x_max, \
                    image_object.y_min, image_object.y_max))

                label = "{0:d} {1:s}: {2:.2f}%".format(image_object.id, image_object.name, \
                         image_object.confidence * 100)
                cv2.rectangle(frame_copy, (image_object.x_min, image_object.y_min), \
                              (image_object.x_max, image_object.y_max), 255, 2)
                ytext = image_object.y_min - 15 if image_object.y_min - 15 > 15 \
                    else image_object.y_min + 15
                cv2.putText(frame_copy, label, (image_object.x_min, ytext), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.imshow('Image objects', frame_copy)
            i_frame = i_frame + 1
            current_time = current_time + time_step

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

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
    