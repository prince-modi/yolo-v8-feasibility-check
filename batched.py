from ultralytics import YOLO
import argparse
import glob
import cv2
import time

parser=argparse.ArgumentParser()
parser.add_argument("-n","--Name",help="Model name")
parser.add_argument("-b","--Batch",help="Batch size")
args=parser.parse_args()
model = args.Name

# Batch size
batch=int(args.Batch)

# Load a model
m1=time.time()
model = YOLO(model)  # load an official model
m2=time.time()

images=glob.iglob("resized/*.jpg")
g1=time.time()

print("img,class,conf,class-name,read,run,write")

for j in range(int(1000/batch)):
    fetch_start=time.time()
    jpeg=[]
    name=[]
    for i in range(batch):
        name.append(next(images))
        jpeg.append(cv2.imread(name[i]))
    fetch_done=time.time()
    run_start=time.time()
    results = model.predict(jpeg, conf=0.5)  # predict on an image
    run_done=time.time()
    # print(f'Class:{cls}, Conf:{conf}')
    for k in range(batch):
        post_start=time.time()
        boxes = results[k].boxes
        cls=boxes.cls.to('cpu').numpy()
        conf=boxes.conf.to('cpu').numpy()
        names = [model.names[int(i)] for i in cls]
        names = ';'.join(i for i in names)
        cls = ';'.join(str(i)[:-2] for i in cls)
        conf = ';'.join(str(i)[:4] for i in conf)
        img=results[k].plot(conf=True, labels=True, boxes=True, masks=True, probs=True)
        jpg=name[k][8:]
        cv2.imwrite(f"results/{jpg}",img)
        post_done=time.time()
        print(f"{jpg},{cls},{conf},{names},{fetch_done-fetch_start:0,.5f},{run_done-run_start:0,.5f},{post_done-post_start:0,.5f}")
