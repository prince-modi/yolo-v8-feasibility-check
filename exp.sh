for i in "yolov8n.pt" "yolov8s.pt" "yolov8m.pt" "yolov8l.pt" "yolov8x.pt" "yolov8n-seg.pt" "yolov8s-seg.pt" "yolov8m-seg.pt" "yolov8l-seg.pt" "yolov8x-seg.pt"
do
	python temp.py -n $i > $i.csv
        cp -r results $i-results
done
