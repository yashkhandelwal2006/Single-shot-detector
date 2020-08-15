import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import json

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'trained-inference-graphs/output_inference_graph_v1.pb/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'annotations/label_map.pbtxt'

# Number of classes to detect
NUM_CLASSES = 1


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

test_images = os.listdir("images/test")


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image_count={}
        for image in test_images:
            print(image)
            file = open("evaluation/detections/"+image.replace('.JPG','.txt').replace('.jpg','.txt'), "w") 
            image_np = cv2.imread('images/test/'+image)
            im_height, im_width, _ = image_np.shape
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            final_score = np.squeeze(scores)
            final_boxes = np.squeeze(boxes)
            text=''
            count = 0
            for i in range(100):
                if scores is None or final_score[i] > 0.5:
                    ymin, xmin, ymax, xmax = list(final_boxes[i])
                    det_box = [xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height]
                    text+='product 0 '+ ' '.join([str(int(i)) for i in det_box])+'\n'
                    count = count + 1
            file.write(text)
            file.close()
            image_count[image] = count
        json_object = json.dumps(image_count)
        with open("image2products.json", "w") as outfile: 
            outfile.write(json_object) 
            
