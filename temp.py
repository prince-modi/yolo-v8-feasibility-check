from ultralytics import YOLO
import argparse
import glob
import cv2
import time

parser=argparse.ArgumentParser()
parser.add_argument("-n","--Name",help="Model name")
args=parser.parse_args()
model = args.Name

# Load a model
m1=time.time()
model = YOLO(model)  # load an official model
m2=time.time()

images=sorted(glob.iglob("resized/*.jpg"))
g1=time.time()

print("img,class,conf,class-name,read,run,write")

for i,image in enumerate(images):
    img=image[8:-4]
    i1=time.time()
    jpeg=cv2.imread(image)
    i2=time.time()
    results = model.predict(jpeg, conf=0.5)  # predict on an image
    r1=time.time()
    boxes = results[0].boxes
    cls=boxes.cls.to('cpu').numpy()
    conf=boxes.conf.to('cpu').numpy()
    names = [model.names[int(i)] for i in cls]
    names = ';'.join(i for i in names)
    cls = ';'.join(str(i)[:-2] for i in cls)
    conf = ';'.join(str(i)[:4] for i in conf)
    # print(f'Class:{cls}, Conf:{conf}')
    jpeg=results[-1].plot(conf=True, labels=True, boxes=True, masks=True, probs=True)
    cv2.imwrite(f"results/{i}.jpg",jpeg)
    w1=time.time()
    print(f"{img},{cls},{conf},{names},{i2-i1:0,.3f},{r1-i2:0,.3f},{w1-r1:0,.3f}")
