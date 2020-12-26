#!/bin/sh

mkdir -p memory_reports

# Cityscapes
python3 mem.py --d 18 > memory_reports/resnet18_cityscapes.txt
python3 mem.py --d 34 > memory_reports/resnet34_cityscapes.txt
python3 mem.py --d 50 > memory_reports/resnet50_cityscapes.txt
python3 mem.py --d 101 > memory_reports/resnet101_cityscapes.txt

#Pascal-VOC
python3 mem.py --i "pascal" --d 18 > memory_reports/resnet18_pascal.txt
python3 mem.py --i "pascal" --d 34 > memory_reports/resnet34_pascal.txt
python3 mem.py --i "pascal" --d 50 > memory_reports/resnet50_pascal.txt
python3 mem.py --i "pascal" --d 101 > memory_reports/resnet101_pascal.txt