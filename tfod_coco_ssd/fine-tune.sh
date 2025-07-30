#!/bin/bash

mkdir -p pretrained_models
wget -O- http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz | tar xzf - -C pretrained_models
