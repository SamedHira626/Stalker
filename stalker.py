"""
This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""
import Jetson.GPIO as GPIO
import jetson.inference
import jetson.utils
import os
import time
import argparse
import numpy as np
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

ENA = 11
IN1 = 13
IN2 = 15
IN3 = 19
IN4 = 21
ENB = 23

GPIO.setmode(GPIO.BOARD)

GPIO.setup(ENA, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ENB, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)

GPIO.output(ENA, GPIO.HIGH)
GPIO.output(ENB, GPIO.HIGH)

WINDOW_NAME = 'Open Zeka'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        if len(boxes) > 0:
            maxacc = 0.0
            index = 0
            l = 0
            while l < len(boxes):
                if confs[l] > maxacc:
                    maxacc = confs[l]
                    index = l
                l += 1
        
            xmin = boxes[index][0]
            xmax = boxes[index][2]
            print("xmax:",xmax)
            print("xmin:",xmin)
            if (xmax+xmin)/2 < 300:
                print("left")
				GPIO.output(IN1, GPIO.HIGH)
				GPIO.output(IN2, GPIO.LOW)
				GPIO.output(IN3, GPIO.LOW)
				GPIO.output(IN4, GPIO.HIGH)
            elif (xmax+xmin)/2 >= 500 :
                print("right")
				GPIO.output(IN1, GPIO.LOW)
				GPIO.output(IN2, GPIO.HIGH)
				GPIO.output(IN3, GPIO.HIGH)
				GPIO.output(IN4, GPIO.LOW)
            else:
                print("forward")
				GPIO.output(IN1, GPIO.HIGH)
				GPIO.output(IN2, GPIO.LOW)
				GPIO.output(IN3, GPIO.HIGH)
				GPIO.output(IN4, GPIO.LOW)


        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'OPENZEKA',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)

    cam.release()
    cv2.destroyAllWindows()
	GPIO.output(ENA, GPIO.LOW)
	GPIO.output(ENB, GPIO.LOW)
	GPIO.output(IN1, GPIO.LOW)
	GPIO.output(IN2, GPIO.LOW)
	GPIO.output(IN3, GPIO.LOW)
	GPIO.output(IN4, GPIO.LOW)
	GPIO.cleanup()


if __name__ == '__main__':
    main()
