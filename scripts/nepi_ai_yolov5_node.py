#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus <https://www.numurus.com>.
#
# This file is part of nepi applications (nepi_apps) repo
# (see https://https://github.com/nepi-engine/nepi_apps)
#
# License: nepi applications are licensed under the "Numurus Software License", 
# which can be found at: <https://numurus.com/wp-content/uploads/Numurus-Software-License-Terms.pdf>
#
# Redistributions in source code must retain this top-level comment block.
# Plagiarizing this software to sidestep the license obligations is illegal.
#
# Contact Information:
# ====================
# - mailto:nepi@numurus.com
#

import os
import copy
import sys
import rospy
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
np.bool = np.bool_
import pandas

from nepi_sdk import nepi_ros
from nepi_sdk import nepi_msg
from nepi_sdk import nepi_ais

from nepi_sdk.ai_node_if import AiNodeIF

# Define your PyTorch model and load the weights
# model = ...

TEST_DETECTION_DICT_ENTRY = {
    'name': 'TEST_DATA', # Class String Name
    'id': 1, # Class Index from Classes List
    'uid': '', # Reserved for unique tracking by downstream applications
    'prob': .3, # Probability of detection
    'xmin': 10,
    'ymin': 10,
    'xmax': 100,
    'ymax': 100
}



class Yolov5Detector():

    #######################
    ### Node Initialization
    DEFAULT_NODE_NAME = "ai_yolov5" # Can be overwitten by luanch command
    def __init__(self):
        #### APP NODE INIT SETUP ####
        nepi_ros.init_node(name= self.DEFAULT_NODE_NAME)
        self.node_name = nepi_ros.get_node_name()
        self.base_namespace = nepi_ros.get_base_namespace()
        self.node_namespace = self.base_namespace + self.node_name
        nepi_msg.createMsgPublishers(self)
        nepi_msg.publishMsgInfo(self,"Starting Initialization Processes")
        ##############################
        # Initialize parameters and fields.
        #node_params = nepi_ros.get_param(self,"~")
        #nepi_msg.publishMsgInfo(self,"Starting node params: " + str(node_params))
        self.model_name = nepi_ros.get_param(self,"~model_name","")
        self.pub_sub_namespace = nepi_ros.get_param(self,"~pub_sub_namespace",self.node_namespace)
        self.yolov5_path = nepi_ros.get_param(self,"~yolov5_path","")
        self.weights_path = nepi_ros.get_param(self,"~weights_path","")
        self.source_img_topic = nepi_ros.get_param(self,"~source_img_topic","")
        threshold_str = nepi_ros.get_param(self,"~detector_threshold","0.5")
        try:
            self.threshold = float(threshold_str)
        except Exception as e:
            self.threshold = 0.5
            nepi_msg.publishMsgWarn(self,"Failed to convert threshold str " + threshold_str + " to float")
        max_rate_str = nepi_ros.get_param(self,"~max_rate_hz","5.0")
        try:
            self.max_rate = float(max_rate_str)
            nepi_msg.publishMsgWarn(self,"Starting with max rate " + str(self.max_rate))
        except Exception as e:
            self.max_rate = 1
            nepi_msg.publishMsgWarn(self,"Failed to convert max rate str " + max_rate_str + " to float")

        if self.model_name == "":
            nepi_msg.publishMsgWarn(self,"Failed to get required node info from param server: ")
            rospy.signal_shutdown("Failed to get valid model info from param")
        else:
            model_info = nepi_ros.get_param(self,"~ai_model","")
            if model_info == "":
                nepi_msg.publishMsgWarn(self,"Failed to get required model info from params: ")
                rospy.signal_shutdown("Failed to get valid model file paths")
            else:

                # Add paths to python
                #nepi_msg.publishMsgWarn(self,"Got model info from param server: " + str(model_info))
                #self.appendProjectFolderPaths(self.yolov5_path)
                self.weight_file_path = os.path.join(self.weights_path, model_info['weight_file']['name'])
                self.classes = model_info['detection_classes']['names']


                # Load the model
                # Model
                #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                self.raw_yolov5_path = r"{}".format(self.yolov5_path)
                self.model = torch.hub.load(self.raw_yolov5_path,'custom', path=self.weight_file_path,source='local')

                '''
                #yolo_py_path = os.path.join(self.yolov5_path,'models')
                #sys.path.append(yolo_py_path)
                # Load the model
                #YOLO = nepi_ais.importAIClass('yolo.py',yolo_py_path,'yolo','Model')
                #self.model = YOLO(self.weight_file_path)
                #self.load_state_dict(torch.load(self.weight_file_path))
                '''
                self.ai_if = AiNodeIF(node_name = self.node_name, 
                                    source_img_topic = self.source_img_topic,
                                    pub_sub_namespace = self.pub_sub_namespace,
                                    classes_list = self.classes,
                                    setThresholdFunction = self.setThreshold,
                                    setMaxRateFunction = self.setMaxRate,
                                    getMaxRateFunction = self.getMaxRate,
                                    processDetectionFunction = self.processDetection)

                #########################################################
                ## Initiation Complete
                nepi_msg.publishMsgInfo(self,"Initialization Complete")
                # Spin forever (until object is detected)
                nepi_ros.spin()
                #########################################################        
              

    def appendProjectFolderPaths(self,project_path):
        for entry in os.scandir(path):
            if entry.is_dir():
                rospy.logwarn(entry)

    def setThreshold(self,threshold):
        self.threshold = threshold

    def setMaxRate(self,rate):
        self.max_rate = rate

    def getMaxRate(self):
        return self.max_rate
             

    def processDetection(self,cv2_img):
        detect_dict_list = [TEST_DETECTION_DICT_ENTRY]
        # Example image
        #img = 'https://ultralytics.com/images/zidane.jpg'
        # Convert BW image to RGB
        if cv2_img.shape[2] != 3:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)
        # Convert BGR image to RGB image
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        cv2_img_shape = cv2_img.shape
        cv2_img_area = cv2_img_shape[0] * cv2_img_shape[1]
        #nepi_msg.publishMsgInfo(self,"Original image size: " + str(orig_size))

        # Update model settings
        self.model.conf = self.threshold  # Confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
        self.model.max_det = 20  # Maximum number of detections per image
        #self.model.eval()
        # Run the detection model on tensor

        try:
            # Inference
            results = self.model(cv2_img)
            #nepi_msg.publishMsgInfo(self,"Got Yolo detection result: " + str(results))
            #cv2_out_img = results.image
            #cv2.imwrite('/mnt/nepi_storage/data/yolov5test.jpg',cv2_out_img)
            rp = results.pandas().xyxy[0]  # img1 predictions (pandas)
            #nepi_msg.publishMsgInfo(self,"Got pandas formated results: " + str(rp))
            #      xmin    ymin    xmax   ymax  confidence  class    name
            # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
            # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
            # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
            # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
        except Exception as e:
            nepi_msg.publishMsgInfo(self,"Failed to process img with exception: " + str(e))
        detect_dict_list = []
        for i, name in enumerate(rp['name']):
            detect_box_area = ( int(rp['xmax'][i]) - int(rp['xmin'][i]) ) * ( int(rp['ymax'][i]) - int(rp['ymin'][i]) )
            detect_box_ratio = detect_box_area / cv2_img_area
            detect_dict = {
                'name': rp['name'][i], # Class String Name
                'id': rp['class'][i], # Class Index from Classes List
                'uid': '', # Reserved for unique tracking by downstream applications
                'prob': rp['confidence'][i], # Probability of detection
                'xmin': int(rp['xmin'][i]),
                'ymin': int(rp['ymin'][i]),
                'xmax': int(rp['xmax'][i]),
                'ymax': int(rp['ymax'][i]),
                'area_pixels': detect_box_area,
                'area_ratio': detect_box_ratio,
            }
            detect_dict_list.append(detect_dict)
            #nepi_msg.publishMsgInfo(self,"Got detect dict entry: " + str(detect_dict))

        return detect_dict_list



if __name__ == '__main__':
    Yolov5Detector()
