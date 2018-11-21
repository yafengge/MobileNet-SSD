# coding: utf-8

# https://blog.csdn.net/Touch_Dream/article/details/80598901

import numpy as np
import sys, os
import cv2

caffe_root = '/home/nick2/NetLab/NetLab/framework/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import lmdb
from caffe.proto import caffe_pb2


CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

db_path = '/home/nick2/data/VOCdevkit/VOC0712/lmdb/'
train_lmdb = db_path + 'VOC0712_trainval_lmdb'
test_lmdb = db_path + 'VOC0712_test_lmdb'

lmdb_env = lmdb.open(train_lmdb, readonly=True)  # 打开数据文件

lmdb_txn = lmdb_env.begin()  # 生成处理句柄
lmdb_cursor = lmdb_txn.cursor()  # 生成迭代器指针
annotated_datum = caffe_pb2.AnnotatedDatum()  # AnnotatedDatum结构

for key, value in lmdb_cursor:
    print(key)
    annotated_datum.ParseFromString(value)
    datum = annotated_datum.datum  # Datum结构
    grps = annotated_datum.annotation_group  # AnnotationGroup结构
    type = annotated_datum.type

   # print("type:", type)

    for grp in grps:
        cls = grp.group_label
        clsname = CLASSES[int(cls)]
        if clsname == 'person':
            print("cls:", grp.group_label)  # object的name标签
            print("clsname:", clsname)  # object的name标签
            for ans in grp.annotation:
                xmin = ans.bbox.xmin * datum.width  # Annotation结构
                ymin = ans.bbox.ymin * datum.height
                xmax = ans.bbox.xmax * datum.width
                ymax = ans.bbox.ymax * datum.height
                print("bbox:", xmin, ymin, xmax, ymax)  # object的bbox标签

                label = datum.label  # Datum结构label以及三个维度
                channels = datum.channels
                height = datum.height
                width = datum.width
                # print("label:", label)
                print("channels:", channels)
                print("height:", height)
                print("width:", width)

                image_x = np.fromstring(datum.data, dtype=np.uint8)  # 字符串转换为矩阵
                image = cv2.imdecode(image_x, -1)  # decode
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.imshow("image", image)  # 显示图片
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break