#!/usr/bin/env python

import sys
import os
import os.path
# ROS namespace setup for stand alone testing. Comment out for deployed version
#NEPI_BASE_NAMESPACE = '/nepi/s2x/'
#os.environ["ROS_NAMESPACE"] = NEPI_BASE_NAMESPACE[0:-1] # remove to run as automation script
import rospy
import glob
import subprocess
import yaml
import time
import numpy as np


from nepi_edge_sdk_base import nepi_ros
from nepi_edge_sdk_base import nepi_msg


from std_msgs.msg import Empty, Float32
from nepi_ros_interfaces.msg import ObjectCount
from nepi_ros_interfaces.srv import ImageClassifierStatusQuery, ImageClassifierStatusQueryResponse


from nepi_edge_sdk_base.save_cfg_if import SaveCfgIF


AI_NAME = 'Yolov5' # Use in display menus
FILE_TYPE = 'AIF_IF'


TEST_AI_DICT = {'description': 'Yolov5 ai framework support', 
'pkg_name': 'nepi_ai_yolov5', 
'if_file_name': 'ai_yolov5_if.py', 
'if_path_name': '/opt/nepi/ros/share/nepi_aifs', 
'if_module_name': 'ai_yolov5_if', 
'if_class_name': 'Yolov5AIF', 
'models_folder_name': 'yolov5_ros', 
'model_prefix': 'yolov5_', 
'launch_file_name': 'yolov5_ros.launch', 
'node_file_name': 'nepi_ai_yolov5_node.py',  
'node_name': 'ai_yolov5',

'active': True
}

TEST_PUB_SUB_NAMESPACE = "/nepi/s2x/ai_detector_mgr"

TEST_MODELS_LIB_PATH = "/mnt/nepi_storage/ai_models/"

TEST_CLASSIFIER = "darknet_common_object_detection_small"

TEST_IMAGE_TOPIC = "color_2d_image"

TEST_THRESHOLD = "0.3"

TEST_RATE = "1.0"


class Yolov5AIF(object):
    TYPICAL_LOAD_TIME_PER_MB = 5

    yolov5_path = '/opt/nepi/ros/share/yolov5'

    def __init__(self, ai_dict,pub_sub_namespace,models_lib_path, run_test = False):
      if pub_sub_namespace[-1] == "/":
        pub_sub_namespace = pub_sub_namespace[:-1]
      self.pub_sub_namespace = pub_sub_namespace
      self.models_lib_path = models_lib_path
      self.pkg_name = ai_dict['pkg_name']
      self.launch_node_name = ai_dict['node_name']
      self.launch_pkg = ai_dict['launch_pkg_name']
      self.launch_file = ai_dict['launch_file_name']
      self.model_prefix = ai_dict['model_prefix']
      self.models_folder = ai_dict['models_folder_name']
      self.models_folder_path =  os.path.join(self.models_lib_path, self.models_folder)
      nepi_msg.printMsgInfo("Yolov5 models path: " + self.models_folder_path)
      if run_test == True:
        self.startClassifier(TEST_CLASSIFIER, TEST_IMAGE_TOPIC, TEST_THRESHOLD, TEST_RATE)
    
    #################
    # Yolov5 Model Functions

    def getModelsDict(self):
        models_dict = dict()
        classifier_name_list = []
        classifier_size_list = []
        classifier_classes_list = []
        # Try to obtain the path to Yolov5 models from the system_mgr
        configs_path_config_folder = os.path.join(self.models_folder_path, 'configs')
        nepi_msg.printMsgInfo("ai_yolov5_if: Looking for models config files in folder: " + configs_path_config_folder)
        # Grab the list of all existing yolov5 cfg files
        if os.path.exists(configs_path_config_folder) == False:
            nepi_msg.printMsgInfo("ai_yolov5_if: Failed to find models config files in folder: " + configs_path_config_folder)
            return models_dict
        else:
            self.cfg_files = glob.glob(os.path.join(configs_path_config_folder,'*.yaml'))
            #nepi_msg.printMsgInfo("ai_yolov5_if: Found network config files: " + str(self.cfg_files))
            # Remove the ros.yaml file -- that one doesn't represent a selectable trained neural net
            for f in self.cfg_files:
                cfg_dict = dict()
                success = False
                try:
                    yaml_stream = open(f, 'r')
                    success = True
                except Exception as e:
                    nepi_msg.printMsgWarn("ai_yolov5_if: Failed to open yaml file: " + str(e))
                if success:
                    try:
                        # Validate that it is a proper config file and gather weights file size info for load-time estimates
                        cfg_dict = yaml.load(yaml_stream)  
                        classifier_keys = list(cfg_dict.keys())
                        classifier_key = classifier_keys[0]
                    except Exception as e:
                        nepi_msg.printMsgWarn("ai_yolov5_if: Failed load yaml data: " + str(e)) 
                        success = False 
                try: 
                    yaml_stream.close()
                except Exception as e:
                    nepi_msg.printMsgWarn("ai_yolov5_if: Failed close yaml file: " + str(e))
                
                if success == False:
                    nepi_msg.printMsgWarn("ai_yolov5_if: File does not appear to be a valid A/I model config file: " + f + "... not adding this classifier")
                    continue
                #nepi_msg.printMsgWarn("ai_yolov5_if: Import success: " + str(success) + " with cfg_dict " + str(cfg_dict))
                cfg_dict_keys = cfg_dict[classifier_key].keys()
                if ("weight_file" not in cfg_dict_keys):
                    nepi_msg.printMsgWarn("ai_yolov5_if: File does not appear to be a valid A/I model config file: " + f + "... not adding this classifier")
                    continue


                classifier_name = os.path.splitext(os.path.basename(f))[0]
                weight_file = os.path.join(self.models_folder_path, "weights",cfg_dict[classifier_key]["weight_file"]["name"])
                if not os.path.exists(weight_file):
                    nepi_msg.printMsgWarn("ai_yolov5_if: Classifier " + classifier_name + " specifies non-existent weights file " + weight_file + "... not adding this classifier")
                    continue
                classifier_classes_list.append(cfg_dict[classifier_key]['detection_classes']['names'])
                #nepi_msg.printMsgWarn("ai_yolov5_if: Classes: " + str(classifier_classes_list))
                classifier_name_list.append(classifier_name)
                classifier_size_list.append(os.path.getsize(weight_file))
            for i,name in enumerate(classifier_name_list):
                model_name = self.model_prefix + name
                model_dict = dict()
                model_dict['name'] = name
                model_dict['size'] = classifier_size_list[i]
                model_dict['load_time'] = self.TYPICAL_LOAD_TIME_PER_MB * classifier_size_list[i] / 1000000
                model_dict['classes'] = classifier_classes_list[i]
                models_dict[model_name] = model_dict
            #nepi_msg.printMsgWarn("Classifier returning models dict" + str(models_dict))
        return models_dict


    def startClassifier(self, classifier, source_img_topic, threshold, max_rate):
        source_img_topic = nepi_ros.find_topic(source_img_topic)
        if source_img_topic == "":
            nepi_msg.printMsgWarn("ai_yolov5_if: Failed to find image topic with str: " + source_img_topic)
            return

        # Check for files
        if os.path.exists(self.yolov5_path) == False:
            nepi_msg.printMsgWarn("ai_yolov5_if: Failed to find yolov5 path: " + self.yolov5_path)
            return

        # Check for files
        weights_path = os.path.join(self.models_folder_path, "weights")
        if os.path.exists(weights_path) == False:
            nepi_msg.printMsgWarn("ai_yolov5_if: Failed to find weights path: " + weights_path)
            return

        configs_path = os.path.join(self.models_folder_path, "configs")
        if os.path.exists(configs_path) == False:
            nepi_msg.printMsgWarn("ai_yolov5_if: Failed to find configs path: " + configs_path)
            return

        network_param_file = (classifier + ".yaml")
        network_param_file_path = os.path.join(configs_path, network_param_file)
        if os.path.exists(network_param_file_path) == False:
            nepi_msg.printMsgWarn("ai_yolov5_if: Failed to find network params file: " + network_param_file_path)
            return

        
        # Build Yolov5 new classifier launch command
        launch_cmd_line = [
            "roslaunch", self.launch_pkg, self.launch_file,
            "pkg_name:=" + self.launch_pkg,
            "model_name:=" + classifier,
            "pub_sub_namespace:=" + self.pub_sub_namespace, 
            "node_name:=" + self.launch_node_name,
            "file_name:=" + self.launch_file,
            "yolov5_path:=" + self.yolov5_path,
            "weights_path:=" + weights_path,
            "configs_path:=" + configs_path,
            "network_param_file:=" + network_param_file,
            "source_img_topic:=" + source_img_topic,
            "detector_threshold:=" + str(threshold),
            "max_rate_hz:=" + str(max_rate)
        ]
        nepi_msg.printMsgInfo("ai_yolov5_if: Launching Yolov5 ROS Process: " + str(launch_cmd_line))
        self.ros_process = subprocess.Popen(launch_cmd_line)
        

        # Setup Classifier Setup Tracking Progress




    def stopClassifier(self):
        nepi_msg.printMsgInfo("ai_yolov5_if: Stopping classifier")
        if not (None == self.ros_process):
            self.ros_process.terminate()
            self.ros_process = None
        self.current_classifier = "None"
        self.current_img_topic = "None"
        
        #self.current_threshold = 0.3

    

if __name__ == '__main__':
    node_name = TEST_AI_DICT['node_name']
    while nepi_ros.check_for_node(node_name):
        nepi_msg.printMsgInfo("ai_yolov5_if: Trying to kill running node: " + node_name)
        nepi_ros.kill_node(node_name)
        nepi_ros.sleep(2,10)
    Yolov5AIF(TEST_AI_DICT,TEST_PUB_SUB_NAMESPACE,TEST_MODELS_LIB_PATH, run_test = True)
