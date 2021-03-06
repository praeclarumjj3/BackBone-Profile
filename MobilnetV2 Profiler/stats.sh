#!/bin/sh

mkdir -p mobilenetv2_stats
mkdir -p mobilenetv2_stats/layers

python3 timer.py > mobilenetv2_stats/mobilenetv2_cityscapes.txt
python3 timer.py --i "pascal" > mobilenetv2_stats/mobilenetv2_pascal.txt

# Layerwise
python3 timer.py --w False > mobilenetv2_stats/layers/mobilenetv2_cityscapes.txt
python3 timer.py --i "pascal" --w False > mobilenetv2_stats/layers/mobilenetv2_pascal.txt