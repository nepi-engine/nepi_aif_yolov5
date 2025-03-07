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
import time
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

from nepi_sdk.ai_detector_if import AiDetectorIF

# Define your PyTorch model and load the weights
# model = ...


TEST_DETECTION_DICT_ENTRY = {
    'name': 'TEST_DATA', # Class String Name
    'id': 1, # Class Index from Classes List
    'uid': '', # Reserved for unique tracking by downstream applications
    'prob': .3, # Probability of detection
    'xmin': 10,
    'ymin': 10,
    'xmax': 50,
    'ymax': 50,
    'width_pixels': 40,
    'height_pixels': 40,
    'area_pixels': 16000,
    'area_ratio': 0.22857
}



class Yolov5Detector():
    defualt_config_dict = {'threshold': 0.3,'max_rate': 5}

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
        node_params = nepi_ros.get_param(self,"~")
        nepi_msg.publishMsgInfo(self,"Starting node params: " + str(node_params))
        self.all_namespace = nepi_ros.get_param(self,"~all_namespace","")
        if self.all_namespace == "":
            self.all_namespace = self.node_namespace
        self.weight_file_path = nepi_ros.get_param(self,"~weight_file_path","")
        self.yolov5_path = nepi_ros.get_param(self,"~yolov5_path","")
        if self.weight_file_path == "" or self.yolov5_path == "":
            nepi_msg.publishMsgWarn(self,"Failed to get required node info from param server: ")
            rospy.signal_shutdown("Failed to get valid model info from param")
        else:
            # The ai_models param is created by the launch files load network_param_file line
            model_info = nepi_ros.get_param(self,"~ai_model","")
            if model_info == "":
                nepi_msg.publishMsgWarn(self,"Failed to get required model info from params: ")
                rospy.signal_shutdown("Failed to get valid model file paths")
            else:
                try: 
                    model_framework = model_info['framework']['name']
                    model_type = model_info['type']['name']
                    model_description = model_info['description']['name']
                    self.classes = model_info['classes']['names']
                    self.img_width = model_info['image_size']['image_width']['value']
                    self.img_height = model_info['image_size']['image_height']['value']
                except Exception as e:
                    nepi_msg.publishMsgWarn(self,"Failed to get required model info from params: " + str(e))
                    rospy.signal_shutdown("Failed to get valid model file paths")

                if model_framework != 'yolov5':
                    nepi_msg.publishMsgWarn(self,"Model not a yolov5 model: " + model_framework)
                    rospy.signal_shutdown("Model not a valid framework")


                if model_type != 'detection':
                    nepi_msg.publishMsgWarn(self,"Model not a valid type: " + model_type)
                    rospy.signal_shutdown("Model not a valid type")

                self.classes = model_info['classes']['names']

                raw_yolov5_path = r"{}".format(self.yolov5_path)
                self.model = torch.hub.load(raw_yolov5_path,'custom', path=self.weight_file_path,source='local')

                nepi_msg.publishMsgInfo(self,"Starting ai_if with defualt_config_dict: " + str(self.defualt_config_dict))
                self.ai_if = AiDetectorIF(model_name = self.node_name,
                                    framework = model_framework,
                                    description = model_description,
                                    img_height = self.img_height,
                                    img_width = self.img_width,
                                    classes_list = self.classes,
                                    defualt_config_dict = self.defualt_config_dict,
                                    all_namespace = self.all_namespace,
                                    processDetectionFunction = self.processDetection)

                #########################################################
                ## Initiation Complete
                nepi_msg.publishMsgInfo(self,"Initialization Complete")
                # Spin forever (until object is detected)
                nepi_ros.spin()
                #########################################################        
              



    def processDetection(self,cv2_img, threshold):
        start_time = time.time()
        detect_dict_list = [TEST_DETECTION_DICT_ENTRY]
        # Example image
        #img = 'https://ultralytics.com/images/zidane.jpg'
        # Convert BW image to RGB
        if cv2_img.shape[2] != 3:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)
        # Convert BGR image to RGB image
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        cv2_img_shape = cv2_img.shape
        cv2_img_width = cv2_img_shape[1]
        cv2_img_height = cv2_img_shape[0]
        cv2_img_area = cv2_img_shape[0] * cv2_img_shape[1]
        #nepi_msg.publishMsgInfo(self,"Original image size: " + str(orig_size))

        # Update model settings
        self.model.conf = threshold  # Confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
        self.model.max_det = 20  # Maximum number of detections per image
        #self.model.eval()
        # Run the detection model on tensor

        try:
            # Inference
            results = self.model(cv2_img)
            #nepi_msg.publishMsgInfo(self,"Got Yolo detection result: " + str(results))
            rp = results.pandas().xyxy[0]  # img1 predictions (pandas)
            #nepi_msg.publishMsgInfo(self,"Got pandas formated results: " + str(rp))
        except Exception as e:
            nepi_msg.publishMsgInfo(self,"Failed to process img with exception: " + str(e))
        detect_time = round( (time.time() - start_time) , 3)
        #nepi_msg.publishMsgInfo(self,"Detect Time: {:.2f}".format(detect_time))
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
                'width_pixels': cv2_img_width,
                'height_pixels': cv2_img_height,
                'area_pixels': detect_box_area,
                'area_ratio': detect_box_ratio
            }
            detect_dict_list.append(detect_dict)
            #nepi_msg.publishMsgInfo(self,"Got detect dict entry: " + str(detect_dict))
        detect_time = round( (time.time() - start_time) , 3)
        #nepi_msg.publishMsgInfo(self,"Detect Time: {:.2f}".format(detect_time))
        return detect_dict_list, detect_time



if __name__ == '__main__':
    Yolov5Detector()
