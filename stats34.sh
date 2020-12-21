#!/bin/sh

mkdir -p resnet34_stats

#ResNet34
python3 timer.py --d 34 > resnet34_stats/resnet34_cityscapes.txt
python3 timer.py --i "pascal" --d 34 > resnet34_stats/resnet34_pascal.txt