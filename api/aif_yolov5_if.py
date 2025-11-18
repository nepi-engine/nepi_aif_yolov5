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

import sys
import os
import os.path
import glob
import subprocess
import yaml
import time
import numpy as np


from nepi_sdk import nepi_sdk



from std_msgs.msg import Empty, Float32, Int32, String, Bool


TEST_AI_DICT = {
'description': 'Yolov5 ai framework support', 
'pkg_name': 'nepi_aif_yolov5', 
'if_file_name': 'aif_yolov5_if.py', 
'if_path_name': '/opt/nepi/nepi_engine/share/nepi_aifs', 
'if_module_name': 'aif_yolov5_if', 
'if_class_name': 'Yolov5AIF', 
'models_folder_name': 'yolov5', 
'launch_pkg_name': 'nepi_aif_yolov5',
'launch_file_name': 'yolov5_ros.launch', 
'node_file_name': 'nepi_ai_yolov5_detection_node.py',  
'active': True
}

TEST_LAUNCH_NAMESPACE = "/nepi/yolov5_test"
TEST_MGR_NAMESPACE = "/nepi/ai_detector_mgr"
TEST_MODELS_LIB_PATH = "/mnt/nepi_storage/ai_models/"



class Yolov5AIF(object):
    TYPICAL_LOAD_TIME_PER_MB = 3.5
    MODEL_FRAMEWORK="yolov5"

    launch_node_process=None
    ai_node_dict = dict()

    yolov5_path = '/opt/nepi/nepi_engine/share/yolov5'

    def __init__(self, aif_dict,launch_namespace, mgr_namespace, models_lib_path):
      if launch_namespace[-1] == "/":
        launch_namespace = launch_namespace[:-1]
      self.launch_namespace = launch_namespace  
      #nepi_sdk.log_msg_warn(self.log_name + "Launch Namespace: " + self.launch_namespace)
      if mgr_namespace[-1] == "/":
        mgr_namespace = mgr_namespace[:-1]
      self.mgr_namespace = mgr_namespace
      self.models_lib_path = models_lib_path
      self.pkg_name = aif_dict['pkg_name']
      self.log_name = self.pkg_name
      self.node_file_dict = aif_dict['node_file_dict']
      self.launch_pkg = aif_dict['launch_pkg_name']
      self.launch_file = aif_dict['launch_file_name']
      self.models_folder = aif_dict['models_folder_name']
      self.models_folder_path =  os.path.join(self.models_lib_path, self.models_folder)
      nepi_sdk.log_msg_info("Yolov5 models path: " + self.models_folder_path)

    
    #################
    # Yolov5 Model Functions

    def getModelsDict(self):
        models_dict = dict()
        # Try to obtain the path to Yolov8 models from the system_mgr
        nepi_sdk.log_msg_debug(self.log_name + ": Looking for model files in folder: " + self.models_folder_path)
        # Grab the list of all existing cfg files
        if os.path.exists(self.models_folder_path) == False:
            nepi_sdk.log_msg_debug(self.log_name + ": Failed to find models folder: " + self.models_folder_path)
            return models_dict
        else:
            self.cfg_files = glob.glob(os.path.join(self.models_folder_path,'*.yaml'))

            # Remove the ros.yaml file -- that one doesn't represent a selectable trained neural net
            for f in self.cfg_files:
                cfg_dict = dict()
                success = False
                try:
                    nepi_sdk.log_msg_warn(self.log_name + ": Opening yaml file: " + f) 
                    yaml_stream = open(f, 'r')
                    success = True
                    nepi_sdk.log_msg_warn(self.log_name + ": Opened yaml file: " + f) 
                except Exception as e:
                    nepi_sdk.log_msg_warn(self.log_name + ": Failed to open yaml file: " + str(e))
                if success:
                    try:
                        # Validate that it is a proper config file and gather weights file size info for load-time estimates
                        nepi_sdk.log_msg_warn(self.log_name + ": Loading yaml data from file: " + f) 
                        cfg_dict = yaml.load(yaml_stream, Loader=yaml.FullLoader)
                        model_keys = list(cfg_dict.keys())
                        model_key = model_keys[0]
                        nepi_sdk.log_msg_warn(self.log_name + ": Loaded yaml data from file: " + f) 
                    except Exception as e:
                        nepi_sdk.log_msg_warn(self.log_name + ": Failed load yaml data: " + str(e)) 
                        success = False 
                try: 
                    nepi_sdk.log_msg_warn(self.log_name + ": Closing yaml data stream for file: " + f) 
                    yaml_stream.close()
                except Exception as e:
                    nepi_sdk.log_msg_warn(self.log_name + ": Failed close yaml file: " + str(e))
                
                if success == False:
                    nepi_sdk.log_msg_warn(self.log_name + ": File does not appear to be a valid A/I model config file: " + f + "... not adding this model")
                    continue
                nepi_sdk.log_msg_warn(self.log_name + ": Import success: " + str(success) + " with cfg_dict " + str(cfg_dict))
                cfg_dict_keys = cfg_dict[model_key].keys()
                nepi_sdk.log_msg_warn(self.log_name + ": Imported model key names: " + str(cfg_dict_keys))
                nepi_sdk.log_msg_warn(self.log_name + ": Imported model key names: " + str(cfg_dict_keys))
                if ("framework" not in cfg_dict_keys):
                    nepi_sdk.log_msg_warn(self.log_name + ": Framework does not specified in model yaml file: " + f + "... not adding this model")
                    continue
                if ("weight_file" not in cfg_dict_keys):
                    nepi_sdk.log_msg_warn(self.log_name + ": File does not appear to be a valid A/I model config file: " + f + "... not adding this model")
                    continue
                nepi_sdk.log_msg_warn(self.log_name + ": Imported model key names: " + str(cfg_dict_keys))
                if ("image_size" not in cfg_dict_keys):
                    nepi_sdk.log_msg_warn(self.log_name + ": File does not specify a image size: " + f + "... not adding this model")
                    continue
                nepi_sdk.log_msg_warn(self.log_name + ": Imported model key names: " + str(cfg_dict_keys))
                if ("classes" not in cfg_dict_keys):
                    nepi_sdk.log_msg_warn(self.log_name + ": File does not specify a classes: " + f + "... not adding this model")
                    continue

                param_file = os.path.basename(f)
                framework = cfg_dict[model_key]["framework"]["name"]
                model_name = os.path.splitext(param_file)[0]

                if framework != self.MODEL_FRAMEWORK:
                    nepi_sdk.log_msg_warn(self.log_name + ": Model " + model_name + " not a yolov3 model" + framework + "... not adding this model")
                    continue


                weight_file = cfg_dict[model_key]["weight_file"]["name"]
                weight_file_path = os.path.join(self.models_folder_path,weight_file)
                nepi_sdk.log_msg_warn(self.log_name + ": Checking that model weights file exists: " + weight_file_path + " for model name " + model_name)
                if not os.path.exists(weight_file_path):
                    nepi_sdk.log_msg_warn(self.log_name + ": Model " + model_name + " specifies non-existent weights file " + weight_file_path + "... not adding this model")
                    continue
                model_type = cfg_dict[model_key]['type']['name']
                if model_type not in self.node_file_dict.keys():
                    nepi_sdk.log_msg_warn(self.log_name + ": Model " + model_name + " specifies non-supported model type " + model_type + "... not adding this model")
                    continue
                else:
                    node_file_name = self.node_file_dict[model_type]
                model_size_mb = float(os.path.getsize(weight_file_path) / 1000000)
                model_dict = dict()

                model_dict['param_file'] = param_file
                model_dict['framework'] = framework
                model_dict['model_name'] = model_name
                model_dict['model_path'] = self.models_folder_path
                model_dict['type'] = model_type
                model_dict['description'] = cfg_dict[model_key]['description']['name']
                model_dict['pkg_name'] = self.pkg_name
                model_dict['img_height'] = cfg_dict[model_key]['image_size']['image_height']['value']
                model_dict['img_width'] = cfg_dict[model_key]['image_size']['image_width']['value']
                model_dict['classes'] = cfg_dict[model_key]['classes']['names']
                model_dict['weight_file']= weight_file
                model_dict['node_file_name'] = node_file_name
                model_dict['size'] = model_size_mb
                model_dict['load_time'] = self.TYPICAL_LOAD_TIME_PER_MB * model_size_mb
                nepi_sdk.log_msg_info(self.log_name + ": Model dict create for model : " + model_name)
                nepi_sdk.log_msg_info(self.log_name + ": Model has size MB: " + str(model_size_mb) + " and load time per MB: " + str(self.TYPICAL_LOAD_TIME_PER_MB)) 
                nepi_sdk.log_msg_info(self.log_name + ": Model has an estimated load time of: " + str(model_dict['load_time']) + " seconds" ) 
                models_dict[model_name] = model_dict

                
        nepi_sdk.log_msg_warn(self.log_name + " Model returning models dict" + str(models_dict))
        return models_dict

    def launchModel(self, model_dict):
        nepi_sdk.log_msg_warn(self.log_name + " Launching Model Node with model dict" + str(model_dict))
        success = False

        model_name = model_dict['model_name']
        node_name = model_name
        node_namespace = self.launch_namespace
        pkg_name = model_dict['pkg_name']
        node_file_folder = os.path.join("/opt/nepi/nepi_engine/lib",pkg_name)
        node_file_name = model_dict['node_file_name']
        
        param_file_path = os.path.join(model_dict['model_path'],model_dict['param_file']),
        weight_file_path = os.path.join(model_dict['model_path'],model_dict['weight_file'])

        nepi_sdk.log_msg_warn(self.log_name + " Launching Model Node with with inputs " + str([pkg_name, node_file_name, node_name]))
        ###############################
        # Launch Node
        node_file_path = os.path.join(node_file_folder,node_file_name)
        if model_name in self.ai_node_dict.keys():
            nepi_sdk.log_msg_info(self.log_name + ": Node Already Launched: " + node_name)
        elif os.path.exists(node_file_path) == False:
            nepi_sdk.log_msg_info(self.log_name + ": Could not find Node File at: " + node_file_path)
        else: 

            # Pre Set Node Params
            param_ns = nepi_sdk.create_namespace(node_namespace,'param_file_path')
            nepi_sdk.set_param(param_ns,param_file_path)

            param_ns = nepi_sdk.create_namespace(node_namespace,'weight_file_path')
            nepi_sdk.set_param(param_ns,weight_file_path)
                   
            #Try and launch node
            
            [success, msg, node_process] = nepi_sdk.launch_node(pkg_name, node_file_name, node_name)
            if success == True:
                self.ai_node_dict[model_name] = {'namesapce':node_namespace, 'process':node_process}
            nepi_sdk.log_msg_info(self.log_name + ": Node launch return msg: " + str(msg))

        return success, node_namespace


    def killModel(self,model_name):
        if model_name in self.ai_node_dict.keys():
            node_process = self.ai_node_dict[model_name]['process']
            nepi_sdk.log_msg_info(self.log_name + ": Killing Yolov5 AI node: " + model_name)
            if not (None == node_process):
                node_process.terminate()
            del self.ai_node_dict[model_name]


 
   

if __name__ == '__main__':
    node_name = "ai_yolov5_test"
    while nepi_sdk.check_for_node(node_name):
        nepi_sdk.log_msg_info(self.log_name + ": Trying to kill running node: " + node_name)
        nepi_sdk.kill_node(node_name)
        nepi_sdk.sleep(2,10)
    Yolov5AIF(TEST_AI_DICT,TEST_LAUNCH_NAMESPACE,TEST_MGR_NAMESPACE,TEST_MODELS_LIB_PATH)
