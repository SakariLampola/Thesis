# -*- coding: utf-8 -*-
"""
ShadowWorld version 2.1
Created on Wed Apr 04 013:30:48 2018

@author: Sakari Lampola
"""
# Imports----------------------------------------------------------------------

import sys
import time
from pycocotools.coco import COCO
import numpy as np
import cv2
#import time
#import random as rnd
#import pyttsx3
#import winsound
#from math import atan, cos, sqrt, exp, log, atan2, sin, pi
from PIL import Image
#import skimage.io as io
#import matplotlib.pyplot as plt
#import pylab
import os
#import six.moves.urllib as urllib
#import sys
#import tarfile
import tensorflow as tf
#import zipfile
#from collections import defaultdict
#from io import StringIO
#from PIL import Image
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
sys.path.append("..")
#import matplotlib as mpl
from utils import label_map_util
from utils import visualization_utils as vis_util
import speech_recognition as sr
import pyttsx3

# Hyperparameters--------------------------------------------------------------

# Other parameters-------------------------------------------------------------
DATADIR='D:\Thesis\Coco'
DATATYPE='val2017'
IMAGEDIR="D:\\Thesis\\Coco\\val2017\\"
STUFFDIR="D:\\Thesis\Coco\\annotations\\stuff_val2017_pixelmaps\\stuff_val2017_pixelmaps\\"
NUM_CLASSES = 90
PATH_TO_CKPT = r'D:\Thesis\Models\faster_rcnn_resnet101_coco_2018_01_28\frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
CATEGORY_INDEX={
        0:{'name':'unlabeled'},
        1:{'name':'person'},
        2:{'name':'bicycle'},
        3:{'name':'car'},
        4:{'name':'motorcycle'},
        5:{'name':'airplane'},
        6:{'name':'bus'},
        7:{'name':'train'},
        8:{'name':'truck'},
        9:{'name':'boat'},
        10:{'name':'traffic light'},
        11:{'name':'fire hydrant'},
        12:{'name':'street sign'},
        13:{'name':'stop sign'},
        14:{'name':'parking meter'},
        15:{'name':'bench'},
        16:{'name':'bird'},
        17:{'name':'cat'},
        18:{'name':'dog'},
        19:{'name':'horse'},
        20:{'name':'sheep'},
        21:{'name':'cow'},
        22:{'name':'elephant'},
        23:{'name':'bear'},
        24:{'name':'zebra'},
        25:{'name':'giraffe'},
        26:{'name':'hat'},
        27:{'name':'backpack'},
        28:{'name':'umbrella'},
        29:{'name':'shoe'},
        30:{'name':'eye glasses'},
        31:{'name':'handbag'},
        32:{'name':'tie'},
        33:{'name':'suitcase'},
        34:{'name':'frisbee'},
        35:{'name':'skis'},
        36:{'name':'snowboard'},
        37:{'name':'sports ball'},
        38:{'name':'kite'},
        39:{'name':'baseball bat'},
        40:{'name':'baseball glove'},
        41:{'name':'skateboard'},
        42:{'name':'surfboard'},
        43:{'name':'tennis racket'},
        44:{'name':'bottle'},
        45:{'name':'plate'},
        46:{'name':'wine glass'},
        47:{'name':'cup'},
        48:{'name':'fork'},
        49:{'name':'knife'},
        50:{'name':'spoon'},
        51:{'name':'bowl'},
        52:{'name':'banana'},
        53:{'name':'apple'},
        54:{'name':'sandwich'},
        55:{'name':'orange'},
        56:{'name':'broccoli'},
        57:{'name':'carrot'},
        58:{'name':'hot dog'},
        59:{'name':'pizza'},
        60:{'name':'donut'},
        61:{'name':'cake'},
        62:{'name':'chair'},
        63:{'name':'couch'},
        64:{'name':'potted plant'},
        65:{'name':'bed'},
        66:{'name':'mirror'},
        67:{'name':'dining table'},
        68:{'name':'window'},
        69:{'name':'desk'},
        70:{'name':'toilet'},
        71:{'name':'door'},
        72:{'name':'tv'},
        73:{'name':'laptop'},
        74:{'name':'mouse'},
        75:{'name':'remote'},
        76:{'name':'keyboard'},
        77:{'name':'cell phone'},
        78:{'name':'microwave'},
        79:{'name':'oven'},
        80:{'name':'toaster'},
        81:{'name':'sink'},
        82:{'name':'refrigerator'},
        83:{'name':'blender'},
        84:{'name':'book'},
        85:{'name':'clock'},
        86:{'name':'vase'},
        87:{'name':'scissors'},
        88:{'name':'teddy bear'},
        89:{'name':'hair drier'},
        90:{'name':'toothbrush'},
        91:{'name':'hair brush'},
        92:{'name':'banner'},
        93:{'name':'blanket'},
        94:{'name':'branch'},
        95:{'name':'bridge'},
        96:{'name':'building-other'},
        97:{'name':'bush'},
        98:{'name':'cabinet'},
        99:{'name':'cage'},
        100:{'name':'cardboard'},
        101:{'name':'carpet'},
        102:{'name':'ceiling-other'},
        103:{'name':'ceiling-tile'},
        104:{'name':'cloth'},
        105:{'name':'clothes'},
        106:{'name':'clouds'},
        107:{'name':'counter'},
        108:{'name':'cupboard'},
        109:{'name':'curtain'},
        110:{'name':'desk-stuff'},
        111:{'name':'dirt'},
        112:{'name':'door-stuff'},
        113:{'name':'fence'},
        114:{'name':'floor-marble'},
        115:{'name':'floor-other'},
        116:{'name':'floor-stone'},
        117:{'name':'floor-tile'},
        118:{'name':'floor-wood'},
        119:{'name':'flower'},
        120:{'name':'fog'},
        121:{'name':'food-other'},
        122:{'name':'fruit'},
        123:{'name':'furniture-other'},
        124:{'name':'grass'},
        125:{'name':'gravel'},
        126:{'name':'ground-other'},
        127:{'name':'hill'},
        128:{'name':'house'},
        129:{'name':'leaves'},
        130:{'name':'light'},
        131:{'name':'mat'},
        132:{'name':'metal'},
        133:{'name':'mirror-stuff'},
        134:{'name':'moss'},
        135:{'name':'mountain'},
        136:{'name':'mud'},
        137:{'name':'napkin'},
        138:{'name':'net'},
        139:{'name':'paper'},
        140:{'name':'pavement'},
        141:{'name':'pillow'},
        142:{'name':'plant-other'},
        143:{'name':'plastic'},
        144:{'name':'platform'},
        145:{'name':'playingfield'},
        146:{'name':'railing'},
        147:{'name':'railroad'},
        148:{'name':'river'},
        149:{'name':'road'},
        150:{'name':'rock'},
        151:{'name':'roof'},
        152:{'name':'rug'},
        153:{'name':'salad'},
        154:{'name':'sand'},
        155:{'name':'sea'},
        156:{'name':'shelf'},
        157:{'name':'sky-other'},
        158:{'name':'skyscraper'},
        159:{'name':'snow'},
        160:{'name':'solid-other'},
        161:{'name':'stairs'},
        162:{'name':'stone'},
        163:{'name':'straw'},
        164:{'name':'structural-other'},
        165:{'name':'table'},
        166:{'name':'tent'},
        167:{'name':'textile-other'},
        168:{'name':'towel'},
        169:{'name':'tree'},
        170:{'name':'vegetable'},
        171:{'name':'wall-brick'},
        172:{'name':'wall-concrete'},
        173:{'name':'wall-other'},
        174:{'name':'wall-panel'},
        175:{'name':'wall-stone'},
        176:{'name':'wall-tile'},
        177:{'name':'wall-wood'},
        178:{'name':'water-other'},
        179:{'name':'waterdrops'},
        180:{'name':'window-blind'},
        181:{'name':'window-other'},
        182:{'name':'wood'},
        }
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


#recognizer = sr.Recognizer()

synthesizer = pyttsx3.init()

#voices = engine.getProperty('voices')
#for voice in voices:
#   engine.setProperty('voice', voice.id)
#   engine.say('The quick brown fox jumped over the lazy dog.')
#engine.runAndWait()


np.random.seed(1)

lut1 = np.array([np.random.randint(0,255) for i in np.arange(0, 256)]).astype("uint8")
lut2 = np.array([np.random.randint(0,255) for i in np.arange(0, 256)]).astype("uint8")
lut3 = np.array([np.random.randint(0,255) for i in np.arange(0, 256)]).astype("uint8")
lut = np.dstack((lut1, lut2, lut3))

np.random.seed(6)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

annFile='{}/annotations/instances_{}.json'.format(DATADIR,DATATYPE)
coco=COCO(annFile)

annFile = '{}/annotations/captions_{}.json'.format(DATADIR,DATATYPE)
coco_caps=COCO(annFile)

imgIds = coco.getImgIds()

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
      line_thickness=6,
      min_score_thresh=0.01)    
    frame[left_corner_height:left_corner_height+height, UI_X1+left_corner_width:UI_X1+left_corner_width+width, 2] = image_right[:,:,0]
    frame[left_corner_height:left_corner_height+height, UI_X1+left_corner_width:UI_X1+left_corner_width+width, 1] = image_right[:,:,1]
    frame[left_corner_height:left_corner_height+height, UI_X1+left_corner_width:UI_X1+left_corner_width+width, 0] = image_right[:,:,2]

    file = STUFFDIR + img['file_name']
    file = file.replace('.jpg','.png')
    image_stuff = np.array(Image.open(file))
    image_stuff_color = np.zeros((height, width,3), dtype='uint8')
    image_stuff_color[:,:,0] = image_stuff[:,:]
    image_stuff_color[:,:,1] = image_stuff[:,:]
    image_stuff_color[:,:,2] = image_stuff[:,:]
#    minValue = image_stuff.min()
#    maxValue = image_stuff.max()
#    disp = np.uint8(255.0 * (image_stuff - minValue) / (maxValue - minValue))
#    disp_rgb = cv2.applyColorMap(disp, cv2.COLORMAP_RAINBOW)
    disp_rgb = cv2.LUT(image_stuff_color, lut)

    frame[left_corner_height:left_corner_height+height, UI_X2+left_corner_width:UI_X2+left_corner_width+width, 2] = disp_rgb[:,:,0]
    frame[left_corner_height:left_corner_height+height, UI_X2+left_corner_width:UI_X2+left_corner_width+width, 1] = disp_rgb[:,:,1]
    frame[left_corner_height:left_corner_height+height, UI_X2+left_corner_width:UI_X2+left_corner_width+width, 0] = disp_rgb[:,:,2]

    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']

    x_rep = UI_X3 + 10
    y_rep = UI_Y0 + 30
    offset = 30
    font_size = 0.7

    label = "Things:"
    cv2.putText(frame, label, (x_rep,y_rep), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 1)
    y_rep += offset
    for i in range(len(classes)):
        if scores[i] > 0.0:
            label="  {0:s} {1:.3f}".format(CATEGORY_INDEX[classes[i]]['name'], scores[i])
            cv2.putText(frame, label, (x_rep,y_rep), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 1)
            y_rep += offset

    y_rep += offset
    label = "Stuff:"
    cv2.putText(frame, label, (x_rep,y_rep), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 1)
    y_rep += offset
    unique_stuff = np.unique(image_stuff)
    for i in range(len(unique_stuff)):
        if unique_stuff[i] > 91 and unique_stuff[i] < 183:
            label="  {0:s}".format(CATEGORY_INDEX[unique_stuff[i]]['name'])
            legend = unique_stuff[i]*np.ones((5,15), np.uint8)
            legend_color = np.zeros((5, 15,3), dtype='uint8')
            legend_color[:,:,0] = legend[:,:]
            legend_color[:,:,1] = legend[:,:]
            legend_color[:,:,2] = legend[:,:]
            
#            disp = np.uint8(255.0 * (frame_color - minValue) / (maxValue - minValue))
            legend_rgb = cv2.LUT(legend_color, lut)
            frame[y_rep-5:y_rep+5-5, x_rep+20:x_rep+15+20, 2] = legend_rgb[:,:,0]
            frame[y_rep-5:y_rep+5-5, x_rep+20:x_rep+15+20, 1] = legend_rgb[:,:,1]
            frame[y_rep-5:y_rep+5-5, x_rep+20:x_rep+15+20, 0] = legend_rgb[:,:,2]
            cv2.putText(frame, label, (x_rep+25,y_rep), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 1)
            y_rep += offset


    y_rep += offset
    label = "Note: Stuff is based on"
    cv2.putText(frame, label, (x_rep,y_rep), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 1)
    y_rep += offset
    label = "      ground truth"
    cv2.putText(frame, label, (x_rep,y_rep), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 1)
    y_rep += offset
    label = "      semantic segmentation."
    cv2.putText(frame, label, (x_rep,y_rep), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 1)
    y_rep += offset

    annIds = coco_caps.getAnnIds(imgIds=img['id']);
    anns = coco_caps.loadAnns(annIds)    

    x_rep = UI_X0 + 10
    y_rep = UI_Y1 + 30
    label = "Human generated captions:"
    cv2.putText(frame, label, (x_rep+5,y_rep), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 1)
    y_rep += offset
    for caption in anns:
        label="  {0:s}".format(caption['caption'])
        cv2.putText(frame, label, (x_rep+20,y_rep), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 1)
        y_rep += offset

    

    cv2.imshow("ShadowWorld", frame)
    mode = check_keyboard_command(mode)
#    cv2.waitKey(5)
   
#    command = 'None'
#    while command not in ['quit']:
#        # Listen to a sentence
#        synthesizer.say('Command, please!')
#        synthesizer.runAndWait()
#        time.sleep(2)
#        with sr.Microphone() as source:
##           print("Say something!")
#            audio = recognizer.listen(source)
#        # Recognize speech using Sphinx
#        try:
#            command = recognizer.recognize_sphinx(audio)
#            print("Sphinx thinks you said " + recognizer.recognize_sphinx(audio))
#        except sr.UnknownValueError:
##            synthesizer.say('I can not understand you')
##            synthesizer.runAndWait()
#            print("Sphinx could not understand audio")
#        except sr.RequestError as e:
##            synthesizer.say('Error')
##            synthesizer.runAndWait()
#            print("Sphinx error; {0}".format(e))
#        synthesizer.say(command)
#        synthesizer.runAndWait()
    
#        if command == "quit":
#            break
    if mode == "quit":
        break

    
cv2.destroyWindow("ShadowWorld")

