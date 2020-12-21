#!/bin/sh

mkdir -p resnet101_stats

#ResNet101
python3 timer.py --d 101 --s 1 > resnet101_stats/resnet101_cityscapes.txt
python3 timer.py --i "pascal" --d 101 --s 1 > resnet101_stats/resnet101_pascal.txt