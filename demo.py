#!/usr/bin/env python
# -*- coding:utf-8 -*-
 
from __future__ import division
import cv2
import time
import sys
import torch 
import numpy as np 

def detectFaceOpenCVDnn(net, frame, age_gender_model):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    '''检测'''
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
 
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            #roi and detect landmark 
            roi = frameOpencvDnn[np.int32(y1):np.int32(y2), np.int32(x1):np.int32(x2), :]
            rw = x2-x1  
            rh = y2-y1 
            img = cv2.resize(roi, (64, 64))    #预处理，与训练时一样
            img = (np.float32(img)/255.0 - 0.5) / 0.5   #[-1, 1]之间
            #h,w, c to c, h, w
            img = img.transpose((2, 0, 1))
            x_input = torch.from_numpy(img).view(1, 3, 64, 64)
            age, gender = age_gender_model(x_input)
            predict_gender = torch.max(gender, 1)[1].cpu().detach().numpy()[0]
            gender = "Male"
            if (predict_gender == 1):
                gender = "Female"
            predict_age = age.cpu().detach().numpy()*116.0     #等价于age.item()
            #probs = landmark_model(x_input.cuda())
            # probs = landmark_model(x_input)
            # landmark_pts = probs.view(5, 2).cpu().detach().numpy()
            # for x, y in landmark_pts:
            #     point_x1 = x*rw 
            #     point_y1 = y*rh
            #     cv2.circle(roi, (np.int32(point_x1), np.int32(point_y1)), 2, (0, 0, 255), 2, 8, 0)
            # cv2.imshow("roi", roi)
            cv2.putText(frameOpencvDnn, ("gender:%s, age:%d"%(gender, int(predict_age[0][0]))), (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes


if __name__ == "__main__":
    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
    '''模型加载'''
    DNN = "TF"
    if DNN == "CAFFE":
        modelFile = "./model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "./model/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "./model/opencv_face_detector_uint8.pb"
        configFile = "./model/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    #load landmark model  
    #landmark_model = torch.load("./checkpoint_bak_lr=0.01/model_landmarks.pth")   #no landmark
    #landmark_model = torch.load("./checkpoint_bak_lr=0.001/model_landmarks.pth")  #no landmark too 
    #landmark_model = torch.load("./checkpoint_bak_lr=0.01_tensor/Epoch2990-train_loss-0.0050.pth")  #the effect is good 
    age_gender_model = torch.load("./checkpoint/Epoch150-train_loss-0.0057.pth")

    conf_threshold = 0.7

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    #cap = cv2.VideoCapture(source)
    cap = cv2.VideoCapture('FistTest.wmv')
    #hasFrame, frame = cap.read()

    # vid_writer = cv2.VideoWriter('output-dnn-{}.avi'.format(str(source).split(".")[0]),
    #                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame.shape[1], frame.shape[0]))
 
    frame_count = 0
    tt_opencvDnn = 0
    while (1):
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        frame_count += 1
 
        t = time.time()
        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net, frame, age_gender_model)
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn
        label = "OpenCV DNN ; FPS : {:.2f}".format(fpsOpencvDnn)
        cv2.putText(outOpencvDnn, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
 
        cv2.imshow("Face Detection Comparison", outOpencvDnn)
 
        #vid_writer.write(outOpencvDnn)
        if frame_count == 1:
            tt_opencvDnn = 0
 
        k = cv2.waitKey(10)
        if k == 27:
            break
    cv2.destroyAllWindows()
    #vid_writer.release()