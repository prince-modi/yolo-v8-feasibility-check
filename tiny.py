import cv2
import os,datetime
import time
import glob
import darknet
import custom_darknet as detect
import numpy as np
import pandas as pd

cfg = 'cfg/yolov4-tiny.cfg'
data = 'cfg/coco.data'
weights = 'weights/yolov4-tiny.weights'
model,class_names,class_colors = darknet.load_network(cfg, data, weights, batch_size=1)

images=glob.iglob("resized/*.jpg")

print("image,read,run,write")
for i,image in enumerate(images):
    img=image[8:-4]
    i1=time.time()
    image=cv2.imread(image)
    i2=time.time()
    final_img, run, write = detect.main(model, class_names, class_colors, image, "output", i) # out is output dir, i is itr
    print(f"{img},{i2-i1:0,.5f},{run:0,.5f},{write:0,.5f}")



