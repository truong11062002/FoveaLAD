
# FoveaLAD

> [FoveaBox with Consistent Label Assignment Distillation for Page Object Detection]()

<!-- [ALGORITHM] -->

## Abstract

With the increasing prevalence of digital documents replacing traditional paper documents, there is an urgent need to develop techniques to extract information from them. In particular, image-based documents pose a significant challenge in object detection, where the identification of objects such as figures, tables, and captions can be difficult. The crux of object detection lies in the accuracy of label assignment, and we all know that the human visual system is highly capable and proficient in recognizing objects. In light of this, we propose a novel approach, FoveaLAD, which integrates FoveaBox drawed inspiration from the human eye, and Label Assignment Distillation to enhance the label assignment process. We evaluate FoveaLAD on the UIT-DODV-Ext image document dataset, which comprises three classes: Table, Figure, and Caption. Our findings demonstrate that FoveaLAD effectively selects different scales for objects of varying sizes and improves the accuracy of label assignment in object detection.

![FoveaLAD](https://github.com/truong11062002/FoveaLAD/blob/main/images/FoveaLAD%20(1).png)




## Comparison with state-of-the-art methods. The top results are highlighted in red font

| **Method**          	| **Table** 	| **Figure** 	| **Caption** 	| **AP@50** 	| **AP@75** 	| **mAP** 	| **Config**                                                                                 	| **Download** 	|
|---------------------	|-----------	|------------	|-------------	|-----------	|-----------	|---------	|--------------------------------------------------------------------------------------------	|--------------	|
| **RepPoints**       	| 41.2      	| 46.4       	| 42.3        	| 74.0      	| 46.3      	| 43.3    	| [config]() 	| [log]()      	|
| **FSAF**            	| **51.1**      	| 85.1       	| 60.1        	| **86.0**      	| 73.3      	| 65.4    	|                                                                                            	|              	|
| **FCOS**            	| 40.1      	| **89.2**       	| 69.2        	| 84.2      	| 70.2      	| 66.2    	|                                                                                            	|              	|
| **FoveaBox**            	| 46.4       	| 88.4       	| 69.5        	| 84.7      	| **74.1**      	| 66.9    	|                                                                                            	|              	|
| **Ours (FoveaLAD)** 	| 44.9      	| 88.0       	| **69.6**        	| 85.1      	| 71.8      	| **67.5**    	|                                                                                            	|              	|

## Experiment on Object Detection Knowledge Distillation. The top results are highlighted in red font

| **Method**          	| **Table** 	| **Figure** 	| **Caption** 	| **AP@50** 	| **AP@75** 	| **mAP** 	| **Config**                                                                                 	| **Download** 	|
|---------------------	|-----------	|------------	|-------------	|-----------	|-----------	|---------	|--------------------------------------------------------------------------------------------	|--------------	|
| **FGD**       	| 32.6      	| 77.3       	| 67.0        	| 81.2      	| 62.6      	| 59.0    	| [config]() 	| [log]()      	|
| **LAD**            	| 42.6      	| 84.2       	| 51.6        	| 81.1      	| 65.3      	| 59.5    	|                                                                                            	|              	|
| **LD**            	| 33.4      	| 83.6      	| **72.3**        	| **85.6**      	| 65.6     	| 63.1   	|                                                                                            	|              	|
| **Ours (FoveaLAD)** 	| **44.9**      	| **88.0**       	| 69.6        	| 85.1      	| **71.8**      	| **67.5**    	|                                                                                            	|              	|

![image](https://github.com/truong11062002/FoveaLAD/blob/main/images/result.png)

## Installation
- Please setup [UIT-DODV-Ext dataset](https://github.com/nguyenvd-uit/uit-together-dataset/blob/main/UIT-DODV-Ext.md) for MMDetection.

- This repository builds upon [MMDetection](https://github.com/open-mmlab/mmdetection). 
See [The MMDetection documentation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation instructions.

- You can see FGD setting at [Focal and Global Knowledge Distillation for Detectors](https://github.com/yzd-v/FGD)

## Citation

We provide config files to reproduce the document object detection performance in the MAPR 2023 paper for [FoveaBox with Consistent Label Assignment Distillation for Page Object Detection]().

```latex



