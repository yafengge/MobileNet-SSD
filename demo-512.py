import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/adas/Netlab/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe
import time

from datetime import datetime


net_file= 'example512/MobileNetSSD_deploy.prototxt'
caffe_model='snapshot512/mobilenet_iter_16194.caffemodel'

EXAMPLES_BASE_DIR='/home/adas/video/detection/20180930/capture20180930/adas-2018-09-30_preview2.wmv'


if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def preprocess(src):
    img = cv2.resize(src, (512,512))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    #origimg = cv2.imread(imgfile)
    origimg=imgfile
    img = preprocess(imgfile)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("SSD", origimg)
    cv2.waitKey(1)
    #k = cv2.waitKey(1) & 0xff
        #Exit if ESC pressed
    #if k == 27 : return False
    return True
while (True):
   # for f in os.listdir(EXAMPLES_BASE_DIR):
    cap = cv2.VideoCapture(EXAMPLES_BASE_DIR)
    while (True):
        t = time.time()
        start = int(round(t * 1000))
        ret, infer_image = cap.read()

        if ret==False:
            break;
        detect(infer_image);

        t = time.time()
        end = int(round(t * 1000))

        c = end - start;

        print('in fertime:', c)
        #print EXAMPLES_BASE_DIR + "/" + "f"
    #  break
    # cap = cv2.VideoCapture(EXAMPLES_BASE_DIR + 'front.avi');
        #cap = cv2.VideoCapture(EXAMPLES_BASE_DIR + '13.3gp');
