#!/bin/sh

mkdir -p resnet101_stats
mkdir -p resnet101_stats/layers

#ResNet101
python3 timer.py --d 101 > resnet101_stats/resnet101_cityscapes.txt
python3 timer.py --i "pascal" --d 101 > resnet101_stats/resnet101_pascal.txt

#ResNet101
python3 timer.py --d 101 --w False > resnet101_stats/layers/resnet101_cityscapes.txt
python3 timer.py --i "pascal" --d 101 --w False > resnet101_stats/layers/resnet101_pascal.txt