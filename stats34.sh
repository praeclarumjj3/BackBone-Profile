#!/bin/sh

#ResNet34
python3 timer.py --d 34 > resnet34_stats/resnet34_cityscapes_basic.txt
python3 timer.py --i "pascal" --d 34 > resnet34_stats/resnet34_pascal_basic.txt
python3 timer.py --b "bt" --d 34 > resnet34_stats/resnet34_cityscapes_bottle.txt
python3 timer.py --i "pascal" --b "bt" --d 34 > resnet34_stats/resnet34_pascal_bottle.txt