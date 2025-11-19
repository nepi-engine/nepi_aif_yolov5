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
    MODEL_FRAMEWORK="yolov5"

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

        ############  Get ALL_NAMESPACE if provided
        param_namespace = nepi_sdk.create_namespace(self.node_namespace,'all_namespace')
        self.all_namespace = nepi_sdk.get_param(param_namespace,"")
        if self.all_namespace == "":
            self.all_namespace = self.node_namespace


        ############  Get WEIGHT_FILE Path
        param_namespace = nepi_sdk.create_namespace(self.node_namespace,'weight_file_path')
        self.weight_file_path = str(nepi_sdk.get_param(param_namespace,""))
        self.msg_if.pub_warn("Got weight file path: " + self.weight_file_path)
        if self.weight_file_path == "" or os.path.exists(self.weight_file_path) == False:
            self.msg_if.pub_warn("Failed to get required node info from param server at: " + str(param_namespace))
            nepi_sdk.signal_shutdown("Failed to get valid weight path, got: " + self.weight_file_path)
            return

        ############  Get PARAMS_FILE Path
        param_namespace = nepi_sdk.create_namespace(self.node_namespace,'param_file_path')
        self.param_file_path = str(nepi_sdk.get_param(param_namespace,""))
        self.msg_if.pub_warn("Got param file path: " + self.param_file_path)
        if self.param_file_path == "" or os.path.exists(self.param_file_path) == False:
            self.msg_if.pub_warn("Failed to get required node info from param server at: " + str(param_namespace))
            nepi_sdk.signal_shutdown("Failed to get valid param path, got: " + self.param_file_path)
            return

        ############### Load Model Params
        yaml_dict = nepi_utils.read_dict_from_file(self.param_file_path)
        
        self.msg_if.pub_warn("Got model info: " + str(yaml_dict))

        if yaml_dict is None:
            self.msg_if.pub_warn("Failed load model info dict from: " + str(self.param_file_path))
            nepi_sdk.signal_shutdown("Failed to get valid model info from param: " + str(self.param_file_path))
            return
        else:
            try: 
                model_info_dict = yaml_dict['ai_model']
                model_framework = model_info_dict['framework']['name']
                model_type = model_info_dict['type']['name']
                model_description = model_info_dict['description']['name']
                self.classes = model_info_dict['classes']['names']
                self.proc_img_width = model_info_dict['image_size']['image_width']['value']
                self.proc_img_height = model_info_dict['image_size']['image_height']['value']
            except Exception as e:
                self.msg_if.pub_warn("Failed to get required model info from params: " + str(e))
                nepi_sdk.signal_shutdown("Failed to get valid model file paths")
                return
            if model_framework != self.MODEL_FRAMEWORK:
                self.msg_if.pub_warn("Model not a " + self.MODEL_FRAMEWORK  + " model: " + model_framework)
                nepi_sdk.signal_shutdown("Model not a valid framework")
                return

            if model_type != 'detection':
                self.msg_if.pub_warn("Model not a valid type: " + model_type)
                nepi_sdk.signal_shutdown("Model not a valid type")
                return
            
            self.device = 'cpu'
            has_cuda = torch.cuda.is_available()
            self.msg_if.pub_warn("CUDA available: " + str(has_cuda))
            if has_cuda == True:
                cuda_count = torch.cuda.device_count()
                self.msg_if.pub_warn("CUDA GPU Count: " + str(cuda_count))
                if cuda_count > 0:
                    self.device = 'cuda'


            ##############################  
            # Load Model

            raw_yolov5_path = r"{}".format(self.yolov5_path)
            self.model = torch.hub.load(raw_yolov5_path,'custom', path=self.weight_file_path,source='local')

            ##############################  


            # Initialize Detector with Blank Img
            self.msg_if.pub_warn("Initializing detector with blank img")
            init_cv2_img=nepi_img.create_cv2_blank_img()
            det_dict=self.processDetection(init_cv2_img)

            # Run Tests
            NUM_TESTS=10
            self.msg_if.pub_warn("Running Detection Speed Test on " + str(NUM_TESTS) + " Images")
            start_time = time.time()
            for i in range(1, NUM_TESTS):
                det_dict=self.processDetection(init_cv2_img)
            elapsed_time = round( ( time.time() - start_time ) , 3)  # Slower for real images
            detect_rate = round( float(1.0)/elapsed_time * NUM_TESTS , 2)
            self.msg_if.pub_warn("Average Detection Time: " + str(elapsed_time) + " sec")
            self.msg_if.pub_warn("Average Detection Rate: " + str(detect_rate) + " hz")

            # Create API IF Class
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
                                processDetectionFunction = self.processDetection,
                                has_img_tiling = False)

            #########################################################
            ## Initiation Complete
            
   

            # Spin forever (until object is detected)
            nepi_sdk.spin()
            #########################################################              
              
              



    def processDetection(self, cv2_img, img_dict=dict(), threshold = 0.3, resize = False, verbose = False):

        img_dict['image_width'] = 1
        img_dict['image_height'] = 1 
        img_dict['prc_width'] = 1
        img_dict['prc_height'] = 1 
        img_dict['ratio'] = 1
        img_dict['tiling'] = False

        detect_dict_list = []
        if cv2_img is not None:

                cv2_img_shape = cv2_img.shape
                cv2_img_width = cv2_img_shape[1]
                cv2_img_height = cv2_img_shape[0]
                cv2_img_area = cv2_img_shape[0] * cv2_img_shape[1]

                if resize == True:
                    [cv2_img,rescale_ratio,prc_width,prc_height] = nepi_img.resize_proportionally(cv2_img, self.proc_img_width,self.proc_img_height,interp = cv2.INTER_NEAREST)
                else:
                    rescale_ratio = 1
                    prc_width = cv2_img_width
                    prc_height = cv2_img_height

                # Convert to RGB
                if nepi_img.is_gray(cv2_img):
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)
                else:
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

                #self.msg_if.pub_info(":yolov5: Preprocessed image with image size: " + str(cv2_img.shape))
                # Create image dict with new image
                img_dict['image_width'] = cv2_img_width 
                img_dict['image_height'] = cv2_img_height 
                img_dict['prc_width'] = prc_width 
                img_dict['prc_height'] = prc_height 
                img_dict['ratio'] = rescale_ratio 
                img_dict['tiling'] = False


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
                    
        return [detect_dict_list, img_dict]



if __name__ == '__main__':
    Yolov5Detector()
