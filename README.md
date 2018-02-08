
Software requirements:

  Python 3.6.2
  OpenCV 3.3.0
  
Usage:

  python ShadowWorld

Object detection network:

  Single Shot Detector (SSD) + MobileNets 
  
  Both netwoks were developed by Google, papers can be found in [papers] folder.
  
  Off-the-shelf implementation is a Caffe version trained by chuanqi305 and
  located in two files: MobileNetSSD_deploy.caffemodel and 
  MobileNetSSD_deploy.prototxt.txt. 
  
Commands:
  s = step one frame
  c = continuos mode
  q = quit