#!/bin/sh

mkdir -p resnet34_stats
mkdir -p resnet34_stats/layers

#ResNet34
python3 timer.py --d 34 > resnet34_stats/resnet34_cityscapes.txt
python3 timer.py --i "pascal" --d 34 > resnet34_stats/resnet34_pascal.txt

#ResNet34
python3 timer.py --d 34 --w False > resnet34_stats/layers/resnet34_cityscapes.txt
python3 timer.py --i "pascal" --d 34 --w False > resnet34_stats/layers/resnet34_pascal.txt