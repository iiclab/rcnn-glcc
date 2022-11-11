# [Mackerel Classification using RCNN and Global and Local Features](https://doi.org/10.1109/ACCESS.2019.2917554)

This is reimplementation of [glcc-frcn.pytorch](https://github.com/iiclab/glcc-frcn.pytorch).

# Quick Start

1. Install the [detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)
2. Clone this repository.
3. Put [dataset (1.8GB)](http:www.iic.ecei.tohoku.ac.jp/~tomo/saba_glcc.tar.gz) to ```path/to/the/repo/data```.  
4. Run the codes. ```$ python demo.py / train.py / test.py```

# Results on Dataset 2022

Average Precision @ IoU=0.5: 0.76  
Accuracy: 0.976 

# Notes

- Parameters are written in ```conf/*.yaml```
- GLCC source code is ```rcnn_glcc.py```
