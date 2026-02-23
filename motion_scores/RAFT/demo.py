import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from tqdm import tqdm
import matplotlib.pyplot as plt

import torchvision
torchvision.set_video_backend("video_reader")
from torchvision.io import VideoReader
import time
DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# def load_video(video_path):
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = torch.from_numpy(frame).permute(2, 0, 1).float()
#         frames.append(frame[None].to(DEVICE))
    
#     return frames

def load_video(video_path, args):

    try:
        video, audio, info = torchvision.io.read_video(video_path, pts_unit="sec") # [250, 720, 1280, 3])
        T, H, W, C = video.shape
    except Exception as e:
        print(f"torchvision failed: {e}")
        print("Trying OpenCV instead...")
        # 使用 OpenCV 作为备选方案
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV 读取的是 BGR，转换为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame)
        cap.release()
        
        if len(frames_list) == 0:
            raise ValueError(f"No frames could be read from: {video_path}")
        
        video = torch.from_numpy(np.array(frames_list))
        T, H, W, C = video.shape
        audio = None
        info = {}
        print(f"Successfully loaded {T} frames using OpenCV")
        

    if H > args.H:
        # print(f"Resizing video from {H} to {args.H}")
        ratio =  args.H / H
        video = torch.nn.functional.interpolate(video.permute(0, 3, 1, 2).float(), (360, int(W*ratio)))
        video = video.permute(0, 2, 3, 1)
    print(video.shape)
    frames = []
    for i in range(video.shape[0]):
        frame = video[i]
        frame = frame.permute(2, 0, 1).float()
        frames.append(frame[None])

    # print(frames[0].shape)
    # return frames, audio.transpose(0, 1).to(DEVICE)
    return frames, None



    # frames = []
    # cap = cv2.VideoCapture(video_path)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    #     frames.append(frame[None].to(DEVICE))
    
    # return frames


def viz(img, flo, output_name=None):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    if output_name is not None:
        cv2.imwrite(output_name, img_flo[:, :, [2,1,0]])
    # cv2.imwrite('output.png', img_flo[:, :, [2,1,0]])
    return img_flo[:, :, [2,1,0]]


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)

        result_video = []
        flow_curve = []
        flow_curve_low = []
        flow_curve_up = []
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            
            flow_score = torch.mean(flow_up[0][0].abs() + flow_up[0][1].abs())
            flow_curve.append(flow_score.cpu().numpy())


            output_name = os.path.join(args.path, os.path.basename(imfile1)[:-4] + '_flow.png')
            output_name = output_name.replace('_frames', '_flows')


            output_name_video = os.path.dirname(output_name)+'.mp4'
            os.makedirs(os.path.dirname(output_name_video), exist_ok=True)

            # os.makedirs(os.path.dirname(output_name), exist_ok=True)
            
            result_video.append(viz(image1, flow_up, output_name=None))
        # save to video
        # import ipdb; ipdb.set_trace()
        
        # result_video = np.stack(result_video)
        h, w, _ = result_video[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_name_video, fourcc, 24, (w, h))
        for frame in result_video:
            out.write(frame.astype(np.uint8))
        out.release()

        # save flow curve
        flow_curve = np.array(flow_curve)
        
        plt.figure(figsize=(10, 3))
        # plt.axis('off')
        # x grid size (all values)
        plt.xticks(np.arange(0, len(flow_curve), 1))

        plt.title("")
        plt.tight_layout(pad=0)  # Set pad=0 for tightest layout
        # set xlim
        plt.xlim(0, len(flow_curve))
        plt.plot(flow_curve)
        plt.savefig(os.path.join(".", 'flow_curve.png'))
        plt.close()
    
def demo_dataset(args, save_fig=False):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for cate in tqdm(os.listdir(args.dataset_root)):
            if cate.startswith('metadata'):
                continue
             
            for instance in tqdm(os.listdir(os.path.join(args.dataset_root, cate)), leave=False):
                path = os.path.join(args.dataset_root, cate, instance)

                # images = glob.glob(os.path.join(path, '*.png')) + \
                #         glob.glob(os.path.join(path, '*.jpg'))
                
                # images = sorted(images)

                if not path.endswith('.mp4'):
                    continue

                images, audio = load_video(path, args)

                result_video = []
                flow_curve = []
                flow_curve_low = []
                flow_curve_up = []
                for image1, image2 in zip(images[:-1], images[1:]):
                    
                    # image1 = load_image(imfile1)
                    # image2 = load_image(imfile2)

                    image1 = image1.to(DEVICE)
                    image2 = image2.to(DEVICE)

                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                    
                    
                    flow_score = torch.mean(flow_up[0][0].abs() + flow_up[0][1].abs())
                    flow_curve.append(flow_score.cpu().numpy())


                    # output_name = os.path.join(os.path.join(args.dataset_root, cate, instance) + '_flow.png')
                    # output_name = output_name.replace('_frames', '_flows')


                    # output_name_video = os.path.dirname(output_name)+'.mp4'
                    # os.makedirs(os.path.dirname(output_name_video), exist_ok=True)
                    # if save_fig and not os.path.exists(f"./data/AVSync15/test_flows/{cate}"):
                    if save_fig :
                        # os.makedirs(os.path.dirname(output_name), exist_ok=True)
                    
                        result_video.append(viz(image1, flow_up, output_name='debug.png'))
                
                # save to video
                # h, w, _ = result_video[0].shape
                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                # out = cv2.VideoWriter("debug.mp4", fourcc, 6, (w, h))
                # for frame in result_video:
                #     out.write(frame.astype(np.uint8))
                # out.release()
                # import ipdb; ipdb.set_trace()
                

                # save flow curve
                flow_curve = np.array(flow_curve)
                
                # plot flow curve
                # '''
                # plt.figure(figsize=(10, 2))
                # # plt.axis('off')
                # # x grid size (all values)
                # plt.xticks(np.arange(0, len(flow_curve), 1))

                # plt.title("")
                # plt.tight_layout(pad=0)  # Set pad=0 for tightest layout
                # # set xlim
                # plt.xlim(0, len(flow_curve))
                # plt.plot(flow_curve)

                # curve_save_path = path+'low_curve.png'
                # curve_save_path = curve_save_path.replace('_frames', '_curves')
                # os.makedirs(os.path.dirname(curve_save_path), exist_ok=True)
                # plt.savefig(curve_save_path)
                # '''

                # save flow curve
                flow_curve = np.array(flow_curve)
                save_path = path+'.npy'
                # save_path = save_path.replace('_frames', '_curves_npy')
                # save_path = save_path.replace('samples', 'curves_npy')
                save_path = save_path.replace('video_clips', 'curves_npy')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, flow_curve)
    
def demo_vggsound(args):
    
    os.makedirs(os.path.join(args.output_path, 'audios') , exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'label')     , exist_ok=True)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    batch_size = args.batch_size

    # 只处理一个特定的视频文件
    single_video_path = "./data/example_video.mp4"
    single_video_name = os.path.basename(single_video_path) 
    
    print(f"Processing single video: {single_video_name}")
    print(f"Full path: {single_video_path}")
    
    # 检查文件是否存在
    if not os.path.exists(single_video_path):
        print(f"Error: Video file not found at {single_video_path}")
        return
    
    # 创建只包含这一个视频的列表
    all_sorted_instances = [single_video_name]
    print(f"Total instances: {len(all_sorted_instances)}")

    with torch.no_grad():
        for instance in tqdm(all_sorted_instances, desc=f"Processing single video"):
            
            # 对于单个视频，直接使用完整路径
            video_path = single_video_path

            images, audio = load_video(video_path, args)

            result_video = []
            flow_curve = []
            flow_curve_low = []
            flow_curve_up = []

            L = len(images)
            
            batch_1 = []
            batch_2 = []

            for i in tqdm(range(0, L-1, batch_size), leave=False):

                batch_2 = images[i+1:i+batch_size+1]
                n_samples = len(batch_2)
                batch_1 = images[i:i+n_samples]

                padder = InputPadder(batch_1[0].shape)
                
                combinded_batch = padder.pad(*batch_1, *batch_2)
                
                batch_1 = combinded_batch[:n_samples]
                batch_1 = torch.cat(batch_1, dim=0).to(DEVICE)
                batch_2 = combinded_batch[n_samples:]
                batch_2 = torch.cat(batch_2, dim=0).to(DEVICE)

                # flow_low, flow_up = model.forward_batch(batch_1, batch_2, iters=20, test_mode=True)
                flow_up = model.forward_batch(batch_1, batch_2, iters=20, test_mode=True)
    
                for j in range(len(flow_up)):
                    flow_score = torch.mean(flow_up[j][0].abs() + flow_up[j][1].abs())
                    flow_curve.append(flow_score.cpu().numpy())

            flow_curve = np.array(flow_curve)
            
            # save flow curve

            
            save_path = os.path.join(args.output_path, 'label', instance.replace('mp4', 'npy'))
            
            # import ipdb; ipdb.set_trace()

            # # plot audio and curver
            # figure, ax = plt.subplots(2, 1, figsize=(10, 4))
            # # plt.axis('off')
            # # x grid size (all values)
            # ax[0].plot(flow_curve)
            # ax[1].plot(audio.cpu().numpy().mean(0))
            # plt.title(instance)
            # plt.savefig("demo.png")
            # plt.close()
            # import ipdb; ipdb.set_trace()
            
            np.save(save_path, flow_curve)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--dataset_root', help="dataset for evaluation")
    parser.add_argument('--output_path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--rank', type=int, help='dist rank')
    parser.add_argument('--node', type=int, help='dist rank')
    parser.add_argument('--N', default=8, type=int, help='dist rank')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--H', default=480, type=int, help='batch size')
    parser.add_argument('--all_instances', default=None, help='instance list')

    args = parser.parse_args()

    # demo(args)
    # demo_dataset(args, save_fig=False)
    demo_vggsound(args)
    # only run one video instance

    # with open("./data/TheGreatestHits/train.txt") as f:
    #     all_listed = f.readlines()
    #     all_listed = [instance.strip() for instance in all_listed]
    # for instance in all_listed:
    #     if os.path.exists(os.path.join('./data/TheGreatestHits/train_video_motion', instance.replace('mp4', 'npy'))):
    #         print(instance)
