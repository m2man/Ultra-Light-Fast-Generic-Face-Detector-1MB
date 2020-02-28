"""
This code uses the pytorch model to detect faces from live video or camera.
"""
import argparse
import sys
import cv2

from vision.ssd.config.fd_config import define_img_size

input_img_size = 320 # 128/160/320/480/640/1280
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer


net_type = "RFB" # slim will faster but less accurate
candidate_size = 1000
threshold = 0.7
test_device = 'cpu' #cuda:0 or cpu
video_path = ''

label_path = "./models/voc-model-labels.txt" # 2 class --> BACKGROUND or face
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    #model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    #model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

timer = Timer()
sum = 0

if video_path != '':
    cap = cv2.VideoCapture(video_path)  # capture from video 
else:
    cap = cv2.VideoCapture(0)  # capture from camera


portion = 1
resize = 1/portion

while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("end")
        break
    small_frame = cv2.resize(orig_image, (0, 0), fx=portion, fy=portion)

    #image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(small_frame, candidate_size / 2, threshold)
    interval = timer.end()
    print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = [x*resize for x in boxes[i, :]]
        label = f" {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (0, 0, 255),
                    2)  # line type

    sum += boxes.size(0)
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("all face num:{}".format(sum))
