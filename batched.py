from ultralytics import YOLO
import argparse
import glob
import cv2
import time

parser=argparse.ArgumentParser()
parser.add_argument("-n","--Name",help="Model name")
args=parser.parse_args()
model = args.Name

# Batch size
batch=2

# Load a model
m1=time.time()
model = YOLO(model)  # load an official model
m2=time.time()

images=glob.iglob("resized/*.jpg")
print(images)
g1=time.time()

print("img,class,conf,class-name,read,run,write")

for j in range(int(1000/batch)):
#    img=image[8:-4]
    i1=time.time()
    jpeg=[]
    name=[]
    for i in range(batch):
        name.append(next(images))
        jpeg.append(cv2.imread(next(images)))
    i2=time.time()
    results = model.predict(jpeg, conf=0.5)  # predict on an image
    r1=time.time()
    print(len(results))
    # print(f'Class:{cls}, Conf:{conf}')
    for k in range(batch):
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
        w1=time.time()
        print(f"{jpg},{cls},{conf},{names},{i2-i1:0,.5f},{r1-i2:0,.5f},{w1-r1:0,.5f}")
