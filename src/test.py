"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.08
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import torch.nn as nn
import numpy as np

import onnx
import onnxruntime as rt

sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model, create_model_yolo_only
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format

def fuse_model(m):
    prev_previous_type = nn.Identity()
    prev_previous_name = ''
    previous_type = nn.Identity()
    previous_name = ''
    for name, module in m.named_modules():
        if isinstance(module, torch.nn.Module)==False:
           print("\n\n\n !!! not Module = ", name)
           print(module)
           print(type(module))

        if prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d and type(module) == nn.ReLU:
            print("FUSED ", prev_previous_name, previous_name, name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name, name], inplace=True)
        elif prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d:
            print("FUSED ", prev_previous_name, previous_name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name], inplace=True)
        elif previous_type == nn.Conv2d and type(module) == nn.ReLU:
            print("FUSED ", previous_name, name)
            #torch.quantization.fuse_modules(m, [previous_name, name], inplace=True)

        prev_previous_type = previous_type
        prev_previous_name = previous_name
        previous_type = type(module)
        previous_name = name

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing

    model = create_model(configs)
    model.print_network()

    model_yolo = create_model_yolo_only(configs)
    model_yolo.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location="cpu"))
    
    print("configs.img_size = ", configs.img_size)
    rand_example = torch.rand(1, 3, configs.img_size, configs.img_size)
    #print("backbone...")
    fuse_model(model)
    model.eval()
    with torch.no_grad():
        tmp = model(rand_example)
    torch.onnx.export(model, rand_example, 'model.onnx', opset_version=9)  
    
    print("loading model...")
    model = rt.InferenceSession('model.onnx')
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    print("output_name = ", output_name)
    print("model loaded! \n")
    

    

    #model = torch.jit.trace(model, rand_example)
    #traced_script_module = torch.jit.script(model)

    print("model = ", model)
    
    
    #model = model.to(memory_format=torch.channels_last)  
    #model = model.half()

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    #model = model.to(device=configs.device)
    model_yolo = model_yolo.to(device=configs.device)

    out_cap = None

    #model.eval()
    model_yolo.eval()

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, (img_paths, imgs_bev) in enumerate(test_dataloader):
            #input_imgs = imgs_bev.to(device=configs.device).float()
            input_imgs = imgs_bev.float()

            #input_imgs = input_imgs.to(memory_format=torch.channels_last)  
            #input_imgs = input_imgs.half()

            num_array = input_imgs.numpy()
                        
            t1 = time_synchronized()
            outputs = model.run(['1180', '1157', '1134'], {input_name: num_array})
            #outputs = torch.tensor(outputs)
            outputs = [outputs[0], outputs[1], outputs[2]]
            
            for index, item in enumerate(outputs):
                outputs[index] = torch.tensor(item).to(device=configs.device).half()
                     
            #outputs = model(input_imgs)
            outputs = model_yolo(outputs, input_imgs)

            t2 = time_synchronized()

            #print("outputs.shape = ", outputs.shape)
            #print("outputs.type = ", outputs.type)
            #print(outputs)
            
            detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

            img_detections = []  # Stores detections for each image index
            img_detections.extend(detections)

            img_bev = imgs_bev.squeeze() * 255
            img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
            for detections in img_detections:
                if detections is None:
                    continue
                # Rescale boxes to original image
                detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
                for x, y, w, l, im, re, *_, cls_pred in detections:
                    yaw = np.arctan2(im, re)
                    # Draw rotated box
                    kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

            img_rgb = cv2.imread(img_paths[0])
            calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, configs.img_size)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

            img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

            out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=608)

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))
            print("configs.save_test_output = ", configs.save_test_output)
            if configs.save_test_output:
                print("configs.output_format = ", configs.output_format)
                if configs.output_format == 'image':
                    print("try...")
                    print("img_paths[0] = ", img_paths[0])
                    print("os.path.basename(img_paths[0]) = ", os.path.basename(img_paths[0]))
                    img_fn = os.path.basename(img_paths[0])[:-4]
                    print("img_fn = ", img_fn)
                    cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
                    print("img savedsaved_fn")
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError

            if configs.show_image:
                cv2.imshow('test-img', out_img)
                print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                if cv2.waitKey(0) & 0xFF == 27:
                    break
            
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()


