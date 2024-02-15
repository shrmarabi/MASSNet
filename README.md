# MASSNet:Multiscale Attention for Single-Stage Ship Instance Segmentation

![image](images/architecture.jpg)

## Performances
![image](images/performance.jpg)

![Table](images/github-mariboats.jpg) 

![Table](images/github-shipinsseg.jpg) 

![Table](images/github-shipsg.jpg) 

## Installation
Our MASSNet is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please check [INSTALL.md](https://github.com/shrmarabi/MASSNet/blob/main/install.md) for installation instructions.

## A quick Demo
a. Download the train weights from google drive (link attached) in the reviewer response file.

 

## After downloading the train weights from the link above
b.To visualize and test the demo for an image
1. Demo script to test a single image for MariboatS:
   
    ```
    python demo/image_demo.py demo/demo/MariboatS/1.jpg work_dirs/MASSNet_Journal_Experiments/MariboatS/MASSNet/massnet_r101_fpn_1x_coco.py work_dirs/MASSNet_Journal_Experiments/MariboatS/MASSNet/latest.pth
    ```
2. Demo script to test a single image for ShipInsSeg:

    ```
    python demo/image_demo.py demo/demo/ShipInsSeg/1.jpg work_dirs/MASSNet_Journal_Experiments/ShipInsSeg/MASSNet/massnet_r101_fpn_1x_coco.py work_dirs/MASSNet_Journal_Experiments/ShipInsSeg/MASSNet/latest.pth
    ```
3. Demo script to test a single image for ShipSG:

    ```
    python demo/image_demo.py demo/demo/ShipSG/1.jpg work_dirs/MASSNet_Journal_Experiments/ShipSG/MASSNet/massnet_r101_fpn_1x_coco.py work_dirs/MASSNet_Journal_Experiments/ShipSG/MASSNet/latest.pth
    ```
   



## Note
### Download the train weights and paste in "work_dirs/MASSNet_Journal_Experiments/MariboatS/MASSNet/latest.pth" path for MariboatS Dataset.(Same for other two datasets)
