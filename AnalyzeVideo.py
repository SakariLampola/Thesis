# -*- coding: utf-8 -*-
"""
Analyze a video frame by frame, detecting objects and estimating states

USAGE
python AnalyzeVideo.py --video videos/CarsOnHighway001.mp4

Created on Wed Nov 29 13:44:02 2017
@author: Sakari Lampola
"""

from math import sqrt
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
    trace_file.write("x_min_c,x_max_c,y_min_c,y_max_c,")
    trace_file.write("vx_min_p,vx_max_p,vy_min_p,vy_max_p,")
    trace_file.write("vx_min_c,vx_max_c,vy_min_c,vy_max_c,distance,")
    trace_file.write("do_c1,do_c2,do_c3,do_c4,do_c5,do_c6,do_c7,do_c8,")
    trace_file.write("io_c1,io_c2,io_c3,io_c4,io_c5,io_c6,io_c7,io_c8,confidence\n")

    # Open the video and initialize follow-up variables
    video = cv2.VideoCapture(videofile)

    width = int(video.get(3))
    height = int(video.get(4))
    size_ratio = 900 / width
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

            if i_frame == 1: # First frame, display frame and wait for 's'
                frame_detected_objects_start = cv2.resize(frame_detected_objects, (0, 0), None, size_ratio, size_ratio)
                frame_image_objects_start = frame_detected_objects_start
                frame_current = np.concatenate((frame_detected_objects_start, frame_image_objects_start), axis=1)
                frame_previous = frame_current.copy()
                frame_final = np.concatenate((frame_previous, frame_current), axis=0)
                cv2.imshow(videofile, frame_final)
                cv2.moveWindow(videofile, 20, 20)
#                not_found = True
#                while not_found: # Discard everything except n, q, c
#                    key_pushed = cv2.waitKey(0) & 0xFF
#                    if key_pushed in [ord('s')]:
#                        not_found = False

            # Detect objects in the current frame
            detected_objects = od.detect_objects(frame_detected_objects, \
                                                 current_time, ic.CONFIDENFE_LEVEL_UPDATE)

            log_file.write("----------------------------------------------\n")
            log_file.write("Time {0:<.2f}, frame {1:d}\n".format(current_time, i_frame))
            
            # Display detected objects
            if len(detected_objects) > 0:
                log_file.write("Detected objects ({0:d}):\n".format(len(detected_objects)))

            for detected_object in detected_objects:

                log_file.write("---{0:d} {1:s} {2:6.2f} {3:4d} {4:4d} {5:4d} {6:4d} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11:.2f} {12:.2f} {13:.2f} {14:.2f}\n".format( \
                    detected_object.id, \
                    ic.CLASS_NAMES[detected_object.class_type], \
                    detected_object.confidence, \
                    detected_object.x_min, detected_object.x_max, \
                    detected_object.y_min, detected_object.y_max, \
                    float(detected_object.appearance[0]), float(detected_object.appearance[1]), \
                    float(detected_object.appearance[2]), float(detected_object.appearance[3]), \
                    float(detected_object.appearance[4]), float(detected_object.appearance[5]), \
                    float(detected_object.appearance[6]), float(detected_object.appearance[7])))

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
            world.update(current_time, detected_objects, log_file, trace_file, time_step)

            # display the world with updated image objects
            if len(world.image_objects) > 0:
                log_file.write("Image objects ({0:d}):\n".format(len(world.image_objects)))

            for image_object in world.image_objects:

                log_file.write("---{0:d} {1:s} {2:6.2f} {3:7.2f} {4:7.2f} {5:7.2f} {6:7.2f} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11:.2f} {12:.2f} {13:.2f} {14:.2f}\n".format(
                    image_object.id, image_object.name, image_object.confidence,
                    image_object.x_min, image_object.x_max, \
                    image_object.y_min, image_object.y_max, \
                    float(image_object.appearance[0]), float(image_object.appearance[1]), \
                    float(image_object.appearance[2]), float(image_object.appearance[3]), \
                    float(image_object.appearance[4]), float(image_object.appearance[5]), \
                    float(image_object.appearance[6]), float(image_object.appearance[7])))

                detected_label = ""
                if not image_object.detected:
                    detected_label = "Undetected"
                border_label = ""
                if image_object.border_left > 1:
                    border_label += " left " + str(image_object.border_left)
                if image_object.border_right > 1:
                    border_label += " right " + str(image_object.border_right)
                if image_object.border_top > 1:
                    border_label += " top " + str(image_object.border_top)
                if image_object.border_bottom > 1:
                    border_label += " bottom " + str(image_object.border_bottom)

                label = "{0:d} {1:s}: {2:.2f} {3:s} {4:s}".format(image_object.id, image_object.name, \
                         image_object.confidence, detected_label, border_label)
                cv2.rectangle(frame_image_objects, (int(image_object.x_min), \
                              int(image_object.y_min)), (int(image_object.x_max), \
                              int(image_object.y_max)), image_object.color, 2)
                
                if (image_object.is_center_reliable()):
                    x_center, y_center = image_object.center_point()
                    x_variance, y_variance = image_object.location_variance()
                    x_std2 = 2.0 * sqrt(x_variance)
                    y_std2 = 2.0 * sqrt(y_variance)
                    cv2.ellipse(frame_image_objects, (int(x_center), int(y_center)), (int(x_std2),int(y_std2)), \
                                0.0, 0, 360, image_object.color, 2)
    
                    x_center_velocity, y_center_velocity = image_object.center_point_velocity()
                    cv2.arrowedLine(frame_image_objects, (int(x_center), int(y_center)), (int(x_center+x_center_velocity),int(y_center+y_center_velocity)), \
                                image_object.color, 2)

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

TEST_VIDEOS = ['videos/AWomanStandsOnTheSeashore-10058.mp4', # 0
               'videos/BlueTit2975.mp4', # 1
               'videos/Boat-10876.mp4', # 2
               'videos/Calf-2679.mp4', # 3
               'videos/Cars133.mp4', # 4
               'videos/CarsOnHighway001.mp4', # 5
               'videos/Cat-3740.mp4', # 6
               'videos/Dog-4028.mp4', # 7
               'videos/Dunes-7238.mp4', # 8
               'videos/Hiker1010.mp4', # 9
               'videos/Horse-2980.mp4', # 10
               'videos/Railway-4106.mp4', # 11
               'videos/SailingBoat6415.mp4', # 12
               'videos/Sheep-12727.mp4', # 13
               'videos/Sofa-11294.mp4'] # 14

if __name__ == "__main__":
    # parse the video file
    AP = argparse.ArgumentParser()
    AP.add_argument("-v", "--video", required=True, help="path to input video")
    ARGS = vars(AP.parse_args())
    VIDEOFILE = ARGS["video"]
    if VIDEOFILE == 'testvideo':
        VIDEOFILE = TEST_VIDEOS[5]
    # spawn the analyzer
    analyze_video(VIDEOFILE)
    