# -*- coding: utf-8 -*-
"""
Analyze a video frame by frame, detecting objects and estimating states

USAGE
python AnalyzeVideo.py --video videos/CarsOnHighway001.mp4

Created on Wed Nov 29 13:44:02 2017
@author: Sakari Lampola
"""

import argparse
import numpy as np
#import time
import cv2
import ImageClasses as ic
import ObjectDetection as od

def analyze_video(videofile):
    """
    Open the video file and step it frame by frame
    """
    # Open log and trace files
    log_file = open("log.txt", "w")
    trace_file = open("trace.txt", "w")
    trace_file.write("time,id,x_min_p,x_max_p,y_min_p,y_max_p,")
    trace_file.write("x_min_m,x_max_m,y_min_m,y_max_m,")
    trace_file.write("x_min_c,x_max_c,y_min_c,y_max_c,distance\n")

    # Open the video and initialize follow-up variables
    video = cv2.VideoCapture(videofile)

    width = int(video.get(3))
    height = int(video.get(4))
    size_ratio = 1000 / width
    fps = video.get(cv2.CAP_PROP_FPS)
   
    time_step = 1.0 / fps
    current_time = 0.0
    i_frame = 1

    log_file.write("Frames per second = {0:.2f}\n".format(fps))
    log_file.write("Frame width = {0:d}\n".format(width))
    log_file.write("Frame height = {0:d}\n".format(height))

    # Create an empty world
    world = ic.ImageWorld(width, height)

    # Loop every frame in the video
    mode_step = True
    while video.isOpened():
        ret, frame_detected_objects = video.read()
        if ret:
            frame_image_objects = frame_detected_objects.copy() # Make a copy to present image objects

            # Detect objects in the current frame
            detected_objects = od.detect_objects(frame_detected_objects, \
                                                 current_time, ic.CONFIDENFE_LEVEL)

            log_file.write("----------------------------------------------\n")
            log_file.write("Time {0:<.2f}, frame {1:d}\n".format(current_time, i_frame))
            log_file.write("Detected objects ({0:d}):\n".format(len(detected_objects)))

            # Display detected objects
            for detected_object in detected_objects:

                log_file.write("---{0:d} {1:s} {2:6.2f} {3:4d} {4:4d} {5:4d} {6:4d}\n".format( \
                    detected_object.id, \
                    ic.CLASS_NAMES[detected_object.class_type], \
                    detected_object.confidence, \
                    detected_object.x_min, detected_object.x_max, \
                    detected_object.y_min, detected_object.y_max))

                label = "{0:d} {1:s}: {2:.2f}".format(detected_object.id, \
                         ic.CLASS_NAMES[detected_object.class_type], \
                         detected_object.confidence)
                cv2.rectangle(frame_detected_objects, \
                              (detected_object.x_min, detected_object.y_min), \
                              (detected_object.x_max, detected_object.y_max), 255, 2)
                ytext = detected_object.y_min - 15 if detected_object.y_min - 15 > 15 \
                    else detected_object.y_min + 15
                cv2.putText(frame_detected_objects, label, (detected_object.x_min, ytext), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) , 2)

            time_label_detected_objects = "Time {0:<.2f}, frame {1:d}, detected objects".format(\
                                                current_time, i_frame)
            cv2.putText(frame_detected_objects, time_label_detected_objects, (10,20), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

            # update the world model
            world.update(current_time, detected_objects, log_file, trace_file)

            # display the world with updated image objects
            log_file.write("Image objects ({0:d}):\n".format(len(world.image_objects)))

            for image_object in world.image_objects:

                log_file.write("---{0:d} {1:s} {2:6.2f} {3:7.2f} {4:7.2f} {5:7.2f} {6:7.2f}\n".format(
                    image_object.id, image_object.name, image_object.confidence,
                    image_object.x_min, image_object.x_max, \
                    image_object.y_min, image_object.y_max))

                label = "{0:d} {1:s}: {2:.2f}".format(image_object.id, image_object.name, \
                         image_object.confidence)
                cv2.rectangle(frame_image_objects, (int(image_object.x_min), \
                              int(image_object.y_min)), (int(image_object.x_max), \
                              int(image_object.y_max)), image_object.color, 2)

                ytext = int(image_object.y_min) - 15 if int(image_object.y_min) - 15 > 15 \
                    else int(image_object.y_min) + 15
                cv2.putText(frame_image_objects, label, (int(image_object.x_min), ytext), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            time_label_image_objects = "Time {0:<.2f}, frame {1:d}, image objects".format(current_time,\
                                             i_frame)
            cv2.putText(frame_image_objects, time_label_image_objects, (10,20), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            
            frame_detected_objects = cv2.resize(frame_detected_objects, (0, 0), None, size_ratio, size_ratio)
            frame_image_objects = cv2.resize(frame_image_objects, (0, 0), None, size_ratio, size_ratio)
            frame_current = np.concatenate((frame_detected_objects, frame_image_objects), axis=1)
            
            if i_frame == 1:
                frame_previous = frame_current.copy()

            frame_final = np.concatenate((frame_previous, frame_current), axis=0)
            cv2.imshow(videofile, frame_final)
            
            frame_previous = frame_current.copy()

            i_frame = i_frame + 1
            current_time = current_time + time_step

        if mode_step:
            not_found = True
            while not_found: # Discard everything except n, q, c
                key_pushed = cv2.waitKey(0) & 0xFF
                if key_pushed in [ord('n'), ord('q'), ord('c')]:
                    not_found = False
            if key_pushed == ord('q'):
                break
            if key_pushed == ord('c'):
                mode_step = False
        else:
            key_pushed = cv2.waitKey(1) & 0xFF
            if key_pushed == ord('q'):
                break
            if key_pushed == ord('s'):
                mode_step = True
       
#        time.sleep(1.0)
#        cv2.waitKey(0)

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
    