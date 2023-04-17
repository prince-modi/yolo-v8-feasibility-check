from ultralytics import YOLO
import glob
import cv2


# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
images=glob.iglob("/home/dream-nano6/prince/dataset/frames/*.jpg")
itr=1
for image in images:
    jpeg=cv2.imread(image)
    jpeg=cv2.resize(jpeg,(1280,720),
            interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"resized/{itr}.jpg",jpeg)
    itr+=1
    print(image)
# Predict with the model
#results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

