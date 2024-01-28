# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import time
import torch
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

img = "./demo/demo/MariboatS/1.jpg"
config = './work_dirs/MASSNet_Journal_Experiments/MariboatS/MariboatS-Yolact-R50/yolact_r50_1x8_coco.py'
checkpoint = './work_dirs/MASSNet_Journal_Experiments/MariboatS/MariboatS-Yolact-R50/epoch_12.pth'
model = init_detector(config, checkpoint, device='cuda:0')
time_start = time.time()
for i in range(100):
    result = inference_detector(model, img)
time_end = time.time()
averge_fps = 1/((time_end - time_start)/100)
print("time is:", averge_fps)
 




