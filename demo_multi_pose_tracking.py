import os
import cv2
import numpy as np
import argparse
#CenterNet
import sys
CENTERNET_PATH = '/home/kaikai/anaconda3/envs/CenterNet_DeepSort/centerNet-deep-sort/CenterNet/src/lib'
sys.path.insert(0, CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from opts import opts

from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time

from utils.debugger import Debugger


def bbox_to_xywh_cls_conf_kps(bbox):
    person_id = 1
    #confidence = 0.5
    # only person
    bbox = bbox[person_id]

    bbox = np.array(bbox)

    if any(bbox[:, 4] > opt.vis_thresh):

        bbox = bbox[bbox[:, 4] > opt.vis_thresh, :]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #

        return bbox[:, :4], bbox[:, 4], bbox[:, 5:39]

    else:

        return None, None, None


class Detector(object):
    def __init__(self, opt):
        self.vdo = cv2.VideoCapture()


        #centerNet detector
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")

        self.write_video = opt.write_video
        # self.write_video = False

    def open(self):

        if opt.input_type == 'webcam':
            self.vdo.open(opt.webcam_ind)

        elif opt.input_type == 'ipcam':
            # load cam key, secret
            with open("cam_secret.txt") as f:
                lines = f.readlines()
                key = lines[0].strip()
                secret = lines[1].strip()

            self.vdo.open(opt.ipcam_url.format(key, secret, opt.ipcam_no))

        # video
        else :
            assert os.path.isfile(opt.vid_path), "Error: path error"
            self.vdo.open(opt.vid_path)

        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("zkk_results/{}/demo_{}.avi".format(opt.task, opt.vid_path.split('/')[-1].split('.')[0]), fourcc, 20, (self.im_width, self.im_height))
        #return self.vdo.isOpened()


    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        avg_fps = 0.0
        while self.vdo.grab():

            frame_no +=1
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = ori_im[ymin:ymax, xmin:xmax]
            #im = ori_im[ymin:ymax, xmin:xmax, :]

            #start_center =  time.time()

            results = self.detector.run(im)['results']
            bbox_xywh, cls_conf, kps = bbox_to_xywh_cls_conf_kps(results)

            if bbox_xywh is not None:
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                end = time.time()
                # print("deep time: {}s, fps: {}".format(end - start_deep_sort, 1 / (end - start_deep_sort)))

                fps = 1 / (end - start)

                avg_fps += fps
                print("centernet time: {}s, fps: {}, avg fps : {}".format(end - start, fps, avg_fps / frame_no))


                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin, ymin))
                    #draw keypoints and show
                    debugger = Debugger(num_classes=1, dataset='coco_hp')
                    debugger.add_img(ori_im, img_id='default')
                    debugger.zkk_add_coco_hps(kps)#画kps
                    ori_im = debugger.imgs['default']#得到画了kps的图
                    debugger.zkk_show_img(pause=True)
            else:
                end = time.time()
                # print("deep time: {}s, fps: {}".format(end - start_deep_sort, 1 / (end - start_deep_sort)))
                fps = 1 / (end - start)
                avg_fps += fps
                #no detection , show original img
                debugger = Debugger(num_classes=1, dataset='coco_hp')
                debugger.add_img(ori_im, img_id='default')
                debugger.zkk_show_img(pause=True)

                print("centernet time: {}s, fps: {}, avg fps : {}".format(end - start, fps, avg_fps / frame_no))

            # cv2.imshow("test", ori_im)
            # cv2.waitKey(1)

            if self.write_video:
                self.output.write(ori_im)



if __name__ == "__main__":
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("test", 800, 600)
#------------------------------------------
    args = argparse.ArgumentParser()
    args.add_argument('--task', default='multi_pose',
                             help='ctdet | ddd | multi_pose | exdet')
    args.add_argument('--model', default='dla_34',
                             help='det backbone model: dla_34| resdcn_18 | resdcn_101')
    args.add_argument('--input_type', default='video',
                             help='video | webcam')
    args.add_argument('--video_path', default='zkk_videos/ETH-Crossing.mp4',
                             help='input video path')
    args.add_argument('--webcam_ind', default=0)

    arch_modelpath = {'dla_34': './CenterNet/models/ctdet_coco_dla_2x.pth',
                      'resdcn_18': './CenterNet/models/ctdet_coco_resdcn18.pth',
                      'resdcn_101': './CenterNet/models/ctdet_coco_resdcn101.pth'}

    input_parse = args.parse_args()
    TASK = input_parse.task
    ARCH = input_parse.model
    MODEL_PATH = arch_modelpath[ARCH]
    if TASK=='multi_pose':
        MODEL_PATH = './CenterNet/models/multi_pose_dla_3x.pth'

    opt = opts().init('{} --load_model {} --arch {}'.format(TASK, MODEL_PATH, ARCH).split(' '))
    opt.input_type = input_parse.input_type
    if opt.input_type == 'video':
        opt.vid_path = input_parse.video_path
    else:
        opt.webcam_ind = input_parse.webcam_ind
        opt.vid_path = '/' + time.strftime("%m-%d_%H:%M:%S", time.localtime()) + '.'

    # vis_thresh
    opt.vis_thresh = 0.5
    opt.arch = ARCH
    #save result video
    opt.write_video = True
#------------------------------------------
    det = Detector(opt)
    det.open()
    det.detect()
