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
import torch
torch.set_num_threads(3)
import cv2
import torchvision.transforms as transforms
import numpy as np
np.bool = np.bool_
import pandas

from nepi_sdk import nepi_sdk
from nepi_sdk import nepi_utils
from nepi_sdk import nepi_img
from nepi_sdk import nepi_ais

from nepi_api.ai_if_detector import AiDetectorIF
from nepi_api.messages_if import MsgIF

# Define your PyTorch model and load the weights
# model = ...



class Yolov5Detector():
    default_config_dict = {'threshold': 0.3,'max_rate': 5}
    model = None
    #######################
    ### Node Initialization
    DEFAULT_NODE_NAME = "ai_yolov5" # Can be overwitten by luanch command
    def __init__(self):
        ####  NODE Initialization ####
        nepi_sdk.init_node(name= self.DEFAULT_NODE_NAME)
        self.class_name = type(self).__name__
        self.base_namespace = nepi_sdk.get_base_namespace()
        self.node_name = nepi_sdk.get_node_name()
        self.node_namespace = nepi_sdk.get_node_namespace()

        ##############################  
        # Create Msg Class
        self.msg_if = MsgIF(log_name = self.class_name)
        self.msg_if.pub_info("Starting Node Initialization Processes")

        ##############################  
        # Initialize Class Variables
        # Create Msg Class
        self.msg_if = MsgIF(log_name = self.class_name)
        self.msg_if.pub_info("Starting Node Initialization Processes")

        ##############################  
        # Initialize Class Variables

        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.msg_if.pub_warn('GPU is available. Using device: ' + str(device))
        else:
            device = torch.device('cpu')
            self.msg_if.pub_warn('GPU not available, using CPU instead.')


        # Initialize parameters and fields.
        node_params = nepi_sdk.get_param("~")
        self.msg_if.pub_info("Starting node params: " + str(node_params))
        self.all_namespace = nepi_sdk.get_param("~all_namespace","")
        if self.all_namespace == "":
            self.all_namespace = self.node_namespace
        self.weight_file_path = nepi_sdk.get_param("~weight_file_path","")
        self.yolov5_path = nepi_sdk.get_param("~yolov5_path","")
        if self.weight_file_path == "" or self.yolov5_path == "":
            self.msg_if.pub_warn("Failed to get required node info from param server: ")
            nepi_sdk.signal_shutdown("Failed to get valid model info from param")
        else:
            # The ai_models param is created by the launch files load network_param_file line
            model_info = nepi_sdk.get_param("~ai_model","")
            if model_info == "":
                self.msg_if.pub_warn("Failed to get required model info from params: ")
                nepi_sdk.signal_shutdown("Failed to get valid model file paths")
            else:
                try: 
                    model_framework = model_info['framework']['name']
                    model_type = model_info['type']['name']
                    model_description = model_info['description']['name']
                    self.classes = model_info['classes']['names']
                    self.proc_img_width = model_info['image_size']['image_width']['value']
                    self.proc_img_height = model_info['image_size']['image_height']['value']
                except Exception as e:
                    self.msg_if.pub_warn("Failed to get required model info from params: " + str(e))
                    nepi_sdk.signal_shutdown("Failed to get valid model file paths")

                if model_framework != 'yolov5':
                    self.msg_if.pub_warn("Model not a yolov5 model: " + model_framework)
                    nepi_sdk.signal_shutdown("Model not a valid framework")


                if model_type != 'detection':
                    self.msg_if.pub_warn("Model not a valid type: " + model_type)
                    nepi_sdk.signal_shutdown("Model not a valid type")

                self.classes = model_info['classes']['names']

                
                raw_yolov5_path = r"{}".format(self.yolov5_path)
                self.model = torch.hub.load(raw_yolov5_path,'custom', path=self.weight_file_path,source='local')
                
                # Initialize Detector with Blank Img
                self.msg_if.pub_info("Initializing detector with blank img")
                init_cv2_img=nepi_img.create_cv2_blank_img()
                img_dict=self.preprocessImage(init_cv2_img)
                det_dict=self.processDetection(img_dict)


                self.msg_if.pub_info("Starting ai_if with default_config_dict: " + str(self.default_config_dict))
                self.ai_if = AiDetectorIF(
                                    namespace = self.node_namespace,
                                    model_name = self.node_name,
                                    framework = model_framework,
                                    description = model_description,
                                    proc_img_height = self.proc_img_height,
                                    proc_img_width = self.proc_img_width,
                                    classes_list = self.classes,
                                    default_config_dict = self.default_config_dict,
                                    all_namespace = self.all_namespace,
                                    preprocessImageFunction = self.preprocessImage,
                                    processDetectionFunction = self.processDetection,
                                    has_img_tiling = False)

                #########################################################
                ## Initiation Complete
                self.msg_if.pub_info("Initialization Complete")
                # Spin forever (until object is detected)
                nepi_sdk.spin()
                #########################################################        
              


    def preprocessImage(self,cv2_img,options_dict=dict()):
        height, width = cv2_img.shape[:2]
        
        # For Future
        '''
        tile = False
        if 'tile'  in options_dict.keys():
            tile = options_dict['tile']
        '''
        [cv2_img,ratio,new_width,new_height] = nepi_img.resize_proportionally(cv2_img, self.proc_img_width,self.proc_img_height,interp = cv2.INTER_NEAREST)
        
        # Convert BW image to RGB
        if nepi_img.is_gray(cv2_img):
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)

        #self.msg_if.pub_info(":yolov5: Preprocessed image with image size: " + str(cv2_img.shape))
        # Create image dict with new image
        img_dict = dict()
        img_dict['cv2_img'] = cv2_img
        img_shape = cv2_img.shape
        img_dict['image_width'] = width 
        img_dict['image_height'] = height 
        img_dict['prc_width'] = new_width 
        img_dict['prc_height'] = new_height 
        img_dict['ratio'] = ratio 
        img_dict['tiling'] = False

        return img_dict



    def processDetection(self,img_dict, threshold = 0.3):
        detect_dict_list = []

        tile = False
        # For Future
        #if 'tile'  in options_dict.keys():
        #    tile = options_dict['tile']
        if img_dict is not None and self.model is not None:
            if 'cv2_img' in img_dict.keys():
                cv2_img = img_dict['cv2_img']
                if cv2_img is not None:

                    # Convert BGR image RGB
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                    cv2_img_shape = cv2_img.shape
                    cv2_img_width = cv2_img_shape[1]
                    cv2_img_height = cv2_img_shape[0]
                    cv2_img_area = cv2_img_shape[0] * cv2_img_shape[1]
                    #self.msg_if.pub_info(":yolov5: Processing detection with image size: " + str(cv2_img_shape))

                    # Update model settings
                    self.model.conf = threshold  # Confidence threshold (0-1)
                    self.model.iou = 0.45  # NMS IoU threshold (0-1)
                    self.model.max_det = 20  # Maximum number of detections per image
                    #self.model.eval()
                    # Run the detection model on tensor

                    try:
                        # Inference
                        results = self.model(cv2_img)
                        #self.msg_if.pub_info("Got Yolo detection result: " + str(results))
                        rp = results.pandas().xyxy[0]  # img1 predictions (pandas)
                        #self.msg_if.pub_info("Got pandas formated results: " + str(rp))
                    except Exception as e:
                        self.msg_if.pub_info("Failed to process img with exception: " + str(e))
                    detect_dict_list = []
                    rescale_ratio = float(1) / img_dict['ratio']
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
                            'area_ratio': detect_box_ratio
                        }
                        # Rescale to orig image size
                        detect_dict['xmin'] = int(detect_dict['xmin'] * rescale_ratio)
                        detect_dict['ymin'] = int(detect_dict['ymin'] * rescale_ratio)
                        detect_dict['xmax'] = int(detect_dict['xmax'] * rescale_ratio)
                        detect_dict['ymax'] = int(detect_dict['ymax'] * rescale_ratio)
                        detect_dict_list.append(detect_dict)
                        #self.msg_if.pub_info("Got detect dict entry: " + str(detect_dict))
        else:
            time.sleep(1)
        return detect_dict_list



if __name__ == '__main__':
    Yolov5Detector()
