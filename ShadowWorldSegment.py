# -*- coding: utf-8 -*-
"""
ShadowWorld version 2.1
Created on Wed Apr 04 013:30:48 2018

@author: Sakari Lampola
"""
# Imports----------------------------------------------------------------------

import sys
from pycocotools.coco import COCO
import numpy as np
import cv2
import time
import random as rnd
import pyttsx3
import winsound
from math import atan, cos, sqrt, exp, log, atan2, sin, pi
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
import tensorflow as tf
sys.path.append("..")
import matplotlib as mpl
from utils import label_map_util
from utils import visualization_utils as vis_util

# Hyperparameters--------------------------------------------------------------

# Other parameters-------------------------------------------------------------
DATADIR='D:\Thesis\Coco'
DATATYPE='val2017'
IMAGEDIR="D:\\Thesis\\Coco\\val2017\\"
STUFFDIR="D:\\Thesis\Coco\\annotations\\stuff_val2017_pixelmaps\\stuff_val2017_pixelmaps\\"
NUM_CLASSES = 90
PATH_TO_CKPT = r'D:\Thesis\Models\faster_rcnn_resnet101_coco_2018_01_28\frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
UI_X0 = 0
UI_X1 = 700
UI_X2 = 1400
UI_X3 = 2100
UI_X4 = 2450
UI_Y0 = 0
UI_Y1 = 700
UI_Y2 = 1300
# Code-------------------------------------------------------------------------
def check_keyboard_command(mode_in):
    """
    Wait for keyboard command and set the required mode
    """
    mode = mode_in
    if mode_in == "step":
        found = False
        while not found: # Discard everything except n, q, c
            key_pushed = cv2.waitKey(0) & 0xFF
            if key_pushed in [ord('s'), ord('q'), ord('c')]:
                found = True
        if key_pushed == ord('q'):
            mode = "quit"
        if key_pushed == ord('s'):
            mode = "step"
        if key_pushed == ord('c'):
            mode = "continuous"
    else:
        key_pushed = cv2.waitKey(1) & 0xFF
        if key_pushed == ord('q'):
            mode = "quit"
        if key_pushed == ord('s'):
            mode = "step"
    return mode

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


annFile='{}/annotations/instances_{}.json'.format(DATADIR,DATATYPE)
coco=COCO(annFile)

annFile = '{}/annotations/captions_{}.json'.format(DATADIR,DATATYPE)
coco_caps=COCO(annFile)

imgIds = coco.getImgIds()
np.random.seed(1)

frame = np.zeros((UI_Y2+1, UI_X4+1, 3), np.uint8)
line_color = (255,255,255)
# Horizontal lines
frame[UI_Y0, UI_X0:UI_X4, :] = line_color
frame[UI_Y1, UI_X0:UI_X3, :] = line_color
frame[UI_Y2, UI_X0:UI_X4, :] = line_color
# Vertical lines
frame[UI_Y0:UI_Y2, UI_X0, :] = line_color
frame[UI_Y0:UI_Y1, UI_X1, :] = line_color
frame[UI_Y0:UI_Y1, UI_X2, :] = line_color
frame[UI_Y0:UI_Y2, UI_X3, :] = line_color

cv2.imshow("ShadowWorld", frame)
cv2.moveWindow("ShadowWorld", 10, 10)

mode = "step"

for i in range(100):

    frame = np.zeros((UI_Y2+1, UI_X4+1, 3), np.uint8)
    line_color = (255,255,255)
    # Horizontal lines
    frame[UI_Y0, UI_X0:UI_X4, :] = line_color
    frame[UI_Y1, UI_X0:UI_X3, :] = line_color
    frame[UI_Y2, UI_X0:UI_X4, :] = line_color
    # Vertical lines
    frame[UI_Y0:UI_Y2, UI_X0, :] = line_color
    frame[UI_Y0:UI_Y1, UI_X1, :] = line_color
    frame[UI_Y0:UI_Y1, UI_X2, :] = line_color
    frame[UI_Y0:UI_Y2, UI_X3, :] = line_color

    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#    image = io.imread(img['coco_url'])
    file=IMAGEDIR + img['file_name']
    image = np.array(Image.open(file))
    
    dims=len(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    if dims == 2: # bw
        image_col = np.zeros((height,width,3), dtype=np.uint8)
        image_col[:,:,0] = image[:,:]
        image_col[:,:,1] = image[:,:]
        image_col[:,:,2] = image[:,:]
        image = image_col
    left_corner_height = int(UI_Y1/2.0-height/2.0)
    left_corner_width = int(UI_X1/2.0-width/2.0)
    frame[left_corner_height:left_corner_height+height, left_corner_width:left_corner_width+width, 2] = image[:,:,0]
    frame[left_corner_height:left_corner_height+height, left_corner_width:left_corner_width+width, 1] = image[:,:,1]
    frame[left_corner_height:left_corner_height+height, left_corner_width:left_corner_width+width, 0] = image[:,:,2]

    output_dict = run_inference_for_single_image(image, detection_graph)

    image_right = image.copy()
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_right,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=6)    
    frame[left_corner_height:left_corner_height+height, UI_X1+left_corner_width:UI_X1+left_corner_width+width, 2] = image_right[:,:,0]
    frame[left_corner_height:left_corner_height+height, UI_X1+left_corner_width:UI_X1+left_corner_width+width, 1] = image_right[:,:,1]
    frame[left_corner_height:left_corner_height+height, UI_X1+left_corner_width:UI_X1+left_corner_width+width, 0] = image_right[:,:,2]

    file = STUFFDIR + img['file_name']
    file = file.replace('.jpg','.png')
    image_stuff = np.array(Image.open(file))
    minValue = image_stuff.min()
    maxValue = image_stuff.max()
    disp = np.uint8(255.0 * (image_stuff - minValue) / (maxValue - minValue))
    disp_rgb = cv2.applyColorMap(disp, cv2.COLORMAP_RAINBOW)

    frame[left_corner_height:left_corner_height+height, UI_X2+left_corner_width:UI_X2+left_corner_width+width, 2] = disp_rgb[:,:,0]
    frame[left_corner_height:left_corner_height+height, UI_X2+left_corner_width:UI_X2+left_corner_width+width, 1] = disp_rgb[:,:,1]
    frame[left_corner_height:left_corner_height+height, UI_X2+left_corner_width:UI_X2+left_corner_width+width, 0] = disp_rgb[:,:,2]

    cv2.imshow("ShadowWorld", frame)
    mode = check_keyboard_command(mode)
    if mode == "quit":
        break

cv2.destroyWindow("ShadowWorld")

