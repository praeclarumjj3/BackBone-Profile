#!/bin/sh

mkdir -p memory_reports

# Cityscapes
python3 mem.py > memory_reports/xception_cityscapes.txt

#Pascal-VOC
python3 mem.py --i "pascal" > memory_reports/xception_pascal.txt
