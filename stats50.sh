#!/bin/sh

#ResNet50
python3 timer.py --d 50 > resnet50_stats/resnet50_cityscapes_basic.txt
python3 timer.py --i "pascal" --d 50 > resnet50_stats/resnet50_pascal_basic.txt
python3 timer.py --b "bt" --d 50 > resnet50_stats/resnet50_cityscapes_bottle.txt
python3 timer.py --i "pascal" --b "bt" --d 50 > resnet50_stats/resnet50_pascal_bottle.txt