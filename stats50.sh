#!/bin/sh

mkdir -p resnet50_stats

#ResNet50
python3 timer.py --d 50 > resnet50_stats/resnet50_cityscapes.txt
python3 timer.py --i "pascal" --d 50 > resnet50_stats/resnet50_pascal.txt