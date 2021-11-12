import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"


import argparse
import torch
import clip
from PIL import Image
import json
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm
import math


def encode_video(video_file, preprocess, model):
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []
    count = 0
    ret = True
    
    while (count < frameCount  and ret):
        ret, frame = cap.read()
        if not ret: # if file is empty break loop
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
        count += 1
        # print('{}/{}'.format(count, frameCount))
    
    if len(images) == 0:
        return None

    image_input = torch.tensor(np.stack(images)).cuda()
    image_features_list = []
    bs = 256
    with torch.no_grad():
        n_inter = math.ceil(len(image_input)/bs)
        for i in tqdm(range(n_inter), desc='encoding vid: {}'.format(video_file)):
            image_features = model.encode_image(image_input[i*bs: (i+1)*bs]).float()
            image_features_list.append(image_features)
    image_features = torch.cat(image_features_list, dim=0)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    cap.release()

    return image_features


def extract_dataset_videos_embeddings(preprocess, model, opt):
    save_dir_path = os.path.join(opt.save_path, 'clip_vid_feats')
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    all_videos_path = glob.glob(opt.videos_path + '/*.mp4')
    for vid_path in tqdm(all_videos_path):
        vid = vid_path.split('/')[-1][:-4]
        save_vid_path = os.path.join(save_dir_path, vid+'.pt')
        if os.path.exists(save_vid_path):
            print('vid:{} done'.format(vid))
            continue
        frames_feature = encode_video(vid_path, preprocess, model)

        if frames_feature == None:
            continue
        else:
            frames_feature = frames_feature.cpu().data
        
        torch.save(frames_feature, save_vid_path)
        # print(vid)



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--videos_path', type=str, default='')
    parse.add_argument('--save_path', type=str, default='', 
        help='the path to save reformat files')
    parse.add_argument('--backbone', type=str, default='RN50')
    opt = parse.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # backbone = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/16']
    if 'ViT-B/16' == opt.backbone:
        opt.save_path = os.path.join(opt.save_path, 'ViT-B-16')
    elif 'ViT-B/32' == opt.backbone:
        opt.save_path = os.path.join(opt.save_path, 'ViT-B-32')
    else:
        opt.save_path = os.path.join(opt.save_path, opt.backbone)
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    model, preprocess = clip.load(opt.backbone, device=device)
    
    extract_dataset_videos_embeddings(preprocess, model, opt)
