#!/bin/sh

mkdir -p xception_stats
mkdir -p xception_stats/layers

python3 timer.py > xception_stats/xception_cityscapes.txt
python3 timer.py --i "pascal" > xception_stats/xception_pascal.txt

python3 timer.py --w False > xception_stats/layers/xception_cityscapes.txt
python3 timer.py --i "pascal" --w False > xception_stats/layers/xception_pascal.txt