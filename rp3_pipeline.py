#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 18:38:05 2018

@author: Cuong Ngo
"""
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='Run in debug mode',
                    action='store_true')
args = parser.parse_args()

mode = args.mode
##############
###Blinking###
##############
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)

def blink(section_id):
	"""Blinking LED light when human is detected
	   Print out the Section with the most human

	Args: 
		section_id: 1,2,3. As the entire field is divided into three main Section
	"""
    for i in range(section_id+1):
        GPIO.output(4, True)
        time.sleep(0.15)
        GPIO.output(4, False)
    print('Sending Drone to Section {}...'.format(section_id))


##############
###PiCamera###
##############
from picamera import PiCamera
from picamera.array import PiRGBArray

camera = PiCamera()
camera.resolution = camera.MAX_RESOLUTION
camera.sharpness = 100

rawCapture = PiRGBArray(camera)
########################
###Detection Pipeline###
########################
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import label_map_util
from utils import visualization as vis_util

###Preprocessing###
def split_into_images(full_img, num_img, overlap_percentage):
	""" Splitting paranomic images into 6 square images

	Args:
		full_img: the entire original panoramic image
		num_img: number of small images to split into (6)
		overlap_percentage: two adjacent image will overlap each other at this percentage of the height (0.1 or 10%)
	Returns:
		img_list: list of smaller images
	"""
    h, w, _ = full_img.shape
    overlap = overlap_percentage*h #calculated based on height
    small_w = (w+(num_img-1)*overlap)//num_img
    img_list = [ full_img[:,int(i*(small_w-overlap)):int(i*(small_w-overlap)+small_w),:] for i in range(num_img)]
    return img_list

###Loading Frozen graph###
MODEL_NAME = 'trained_model'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'person_label_map.pbtxt')
NUM_CLASSES = 1

###Loading label map### code from Tensorflow Object Detection API
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

###Load the Tensorflow model into memory###
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

###Getting the essential tensor###
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

###Lightning flag### Using for testing
def lightning():
    if time.time() - lightning_start < 60:
        return True
    else:
        return False

###Performing detection on the splitted images###
last_full_img = [{} for i in range(6)] 
#list of previous image list used to compare with current image list, so that we can eliminate:
#+ stationary human-like objects (which shouldn't be human as who would stand still for 6 seconds)
#+ or stationary false positive like metal poles (which may be mistaken to be human)

while lightning(): #This is to stimulate lightning alert signal.
    start_time = time.time()
    instance_folder = ''
    
    if args.debug: #save detection results in debug mode so that security guards can check.
        instance_folder = os.path.join('debug',datetime.now().strftime('%d_%b_%Y'),datetime.now().strftime('%H-%M-%S_%d_%b_%Y'))
        if not os.path.exists(instance_folder):
            os.makedirs(instance_folder)   

    num_person_list = []
    
    camera.capture(rawCapture, 'rgb') #capture with picamera
    rawCapture.truncate(0)
    full_img = rawCapture.array[550:850] #crop out to get wide angle panorama
    
    img_list = split_into_images(full_img, 6, 0.1) #split image into 6 smaller square image with 10% overlap
    for count, image in enumerate(img_list): #run human detection per image
        start = time.time()
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: np.expand_dims(image, 0)})

        num_person_item, split_img = vis_util.visualize_boxes_and_labels_on_image_array(
                                            args.debug,
                                            instance_folder,
                                            last_full_img[count], #last detected list to compare with
                                            image,
                                            np.squeeze(boxes),
                                            np.squeeze(classes).astype(np.int32),
                                            np.squeeze(scores),
                                            category_index,
                                            use_normalized_coordinates=True,
                                            line_thickness=2,
                                            min_score_thresh=0.40)
        #Refractored visualization code from Tensorflow Object Detection API
        #to incorporate the trick to eliminate stationary objects.

        num_person_list.append(num_person_item)
        last_full_img[count] = split_img
    num_person_section = [num_person_list[i]+num_person_list[i+1] for i in (0,2,4)] 
    #since each Section consist of 2 images (6 images over 3 Sections)
    #Adding up number of human in groups of 2 images to get number of people per section
    print('Total time per image {}s'.format(time.time()-start_time))
    if max(num_person_section):
        print('There are {} people on the field'.format(max(num_person_section)))
        blink(num_person_section.index(max(num_person_section)))
    else:
        print('Aint no one is here')

GPIO.cleanup() #free up GPIO port