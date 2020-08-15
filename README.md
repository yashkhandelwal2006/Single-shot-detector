**Train SSD with only “one” anchor box per feature-map cell**

**STEP-1:  Data preparation**

First download the dataset and annotations.csv file from [https://github.com/gulvarol/grocerydataset](https://github.com/gulvarol/grocerydataset). As it goes in the csv all the filenames with bounding boxes and class information is present for all training and testing images. Split them in two csv’s i.e. separate data for training and testing. Then convert these csv files into record files which will be the input to our tensorflow model. Also the final .record files have been removed from the from the directory as it was a large file, but just to clear things out for whoso ever sees the code it was in the below mentioned folder(which is empty right now.)
Path to script:   ./scripts/preprocessing/
Path to train test csv and .record files:     ./workspace/training_demo/annotations/

**STEP-2:  Detection Network Used**

So for training a SSD usually it’s better to use pretrained weights(transfer learning) rather than starting from scratch as it may take a lot of time and you might need a lot of data to do so. The model used for this problem is  the ssd_inception_v2_coco model, since it provides a relatively good trade-off between performance and speed. Also took the config file for the same model in order to train in on the custom dataset. Although the  pre-trained weights are not included as it adds up to the total size but they can be downloaded from the model zoo provided by tensorflow and it has to be put at the following path in order to start the training.
Path to pretrained weights:    ./workspace/training_demo/pre-trained-model/
Path to config file used:    ./workspace/training_demo/training/*.config

**STEP-3:  Training and Hyperparameters / Anchor Boxes Tuning **

For training the model, tensorflow’s official models installation is important which contains all the necessary scripts specifically for object detection including the model specific scripts i.e. in this case for “ssd_inception_v2_coco mode”. Once installed all that, one needs to run the “model_main.py” script for starting the training process. Okay, so before starting the training process some parameters should be taken care off. 
**_Anchors_:**   It is one of the most important parameters which needs to be tuned according to the requirement as well as incoming image data(training data) in order to get good results. This further depends on aspect ratio, scale and number of anchors used per feature map cell. For getting a better idea of what should be the values of these parameters most important is visualizing through your training data. Go through all the images and find what are the most common(ranges) aspect ratios and the scales relative to the whole image. Script for this purpose can be found at the below mentioned path. After careful investigation the final values for scale were set as 0.03 to 0.058(the values were tuned multiple times to eventually get the final values). Here 0.03 is for the default anchor layer(with aspect ratio being 1) and 0.58 is for the only layer used with aspect ratio as o.76. The idea behind this is that in the initial layers the image size(feature-map) is large enough to detect small objects, as the no. of layers increases small features are no longer prominent and that’s why it is preferred for larger objects.      
**_Feature-maps_:**   Feature-maps are nothing but the output of a layer which is usually obtained by convolving over the images, It contains the most prominent features which an image of size as of the feature-map can take. As we move forward into deeper layers, it’s size keeps on decreasing and that is the reason we can no longer search for small objects in them. For this problem as we are dealing with highly populated images with a lot of small objects, only the initial layers have been used and the rest have been removed.
**_Learning Rate_:    **Learning rate is another important parameter which can drastically change the model’s performance. If taken very high it may skip the minimum i.e. as in gradient descent we gradually move towards the minimum, if it is kept high it may skip it and similarly for the case where it is very low it may never reach the minimum or it may take a lot of time. So for this case the learning rate is set to 0.004 with a decay factor of 0.95 if it reaches the decay_steps, but for this case it never reaches there as the training was completed before that.
All these changes can be found in the config file for which the path is given above.
Path to estimate optimal anchor values: ./scripts/preprocessing/
Path to Tensorflow models: ./models

**STEP-4:  Evaluation Scripts **

For calculating the mAP, precision, recall values the scripts can be found at the path:   ./workspace/training_demo/evaluation

**STEP-5:  Final Results **

The final results (mAP, precision, recall) can be found in metrics.json.

**The Final “.pb” file can be found in ./workspace/training_demo/trained-inference-graphs.**
