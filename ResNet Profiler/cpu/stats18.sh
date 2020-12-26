#!/bin/sh

mkdir -p resnet18_stats
mkdir -p resnet18_stats/layers

#ResNet18
python3 timer.py --d 18 > resnet18_stats/resnet18_cityscapes.txt
python3 timer.py --i "pascal" --d 18 > resnet18_stats/resnet18_pascal.txt

#ResNet18
python3 timer.py --d 18 --w False > resnet18_stats/layers/resnet18_cityscapes.txt
python3 timer.py --i "pascal" --d 18 --w False > resnet18_stats/layers/resnet18_pascal.txt