## Introduction

This is the code implementation of the RoI attention method. Our code is built on the basis of MMDetection.

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the OpenMMLab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

The master branch works with **PyTorch 1.6** and [**MMDetection v2.6.0**](https://github.com/open-mmlab/mmdetection).


## Key code

The implementation of the global attention module is in [single_level_roi_extractor_with_attention.py](mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor_with_attention.py).
The implementation of the self attention module is in [roi_attention_bbox_head.py](mmdet/models/roi_heads/bbox_heads/roi_attention_bbox_head.py).

## Conifgs

We set up 4 config files to realize Global RoI Attention Module, three mannners of Slef RoI Attention Module, and Cascade RoI Attention. Refer to [configs/tct](configs/tct) for details.

## Main Results
Model | mAP  | mAP@50 | mAP@75
--- |:---:|:---:|:---:
Faster R-CNN with FPN (baseline) | 28.4 | 50.0 | 29.0
Global RoI Attention | 29.7 | 52.7 | 30.3 
Self RoI Attention | 29.5 | 52.2 | 30.2 
Cascade RoI Attention | 30.0 | 52.9 | 31.2

## Contributing to the project

Any pull requests or issues are welcome.


## Contact

This repo is currently maintained by Shuo Feng ([@FengsShuo96](http://github.com/FengsShuo96)). 
