import os
import argparse
import pickle
import numpy as np
import json
import glob
import torch
import math
from tqdm import tqdm
from emscore import EMScorer
from emscore.utils import get_idf_dict, compute_correlation_uniquehuman
import clip

def get_feats_dict(feat_dir_path):
    print('loding cache feats ........')
    file_path_list = glob.glob(feat_dir_path+'/*.pt')
    feats_dict = {}
    for file_path in tqdm(file_path_list):
        vid = file_path.split('/')[-1][:-3]
        data = torch.load(file_path)
        feats_dict[vid] = data
    return feats_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', default='', type=str, help='The path you storage VATEX-EVAL dataset')
    parser.add_argument('--vid_base_path', default='', type=str, help='The path you storage VATEX-EVAL videos (optinal, if you use prepared video feats, You do not need to consider this)')
    parser.add_argument('--use_n_refs', default=1, type=int, help='How many references do you want to use for evaluation (1~9)')
    parser.add_argument('--use_feat_cache', default=True, action='store_true', help='Whether to use pre-prepared video features')
    parser.add_argument('--use_idf', action='store_true', default=True)

    opt = parser.parse_args()
    
    """
    Dataset prepare
    """
    samples_list = pickle.load(open(os.path.join(opt.storage_path, 'candidates_list.pkl'), 'rb'))
    gts_list = pickle.load(open(os.path.join(opt.storage_path, 'gts_list.pkl'), 'rb'))
    all_human_scores = pickle.load(open(os.path.join(opt.storage_path, 'human_scores.pkl'), 'rb'))
    all_human_scores = np.transpose(all_human_scores.reshape(3, -1), (1, 0))
    video_ids = pickle.load(open(os.path.join(opt.storage_path, 'video_ids.pkl'), 'rb'))
    vid_base_path = 'your path to save vatex val videos'  # optional
    cands = samples_list.tolist()
    refs = gts_list.tolist()
    

    """
    Video feats prepare
    """
    use_uniform_sample = 10

    if not opt.use_feat_cache:
        vids = [vid_base_path+vid+'.mp4' for vid in video_ids]
        metric = EMScorer(vid_feat_cache=[])
    else:
        vid_clip_feats_dir = os.path.join(opt.storage_path, 'VATEX-EVAL_video_feats')
        video_clip_feats_dict = get_feats_dict(vid_clip_feats_dir)
        if use_uniform_sample:
            for vid in video_clip_feats_dict:
                data = video_clip_feats_dict[vid]
                select_index = np.linspace(0, len(data)-1, use_uniform_sample)
                select_index = [int(index) for index in select_index]
                video_clip_feats_dict[vid] = data[select_index]

        vids = video_ids.tolist()
        metric = EMScorer(vid_feat_cache=video_clip_feats_dict)
    

    """
    Prepare IDF
    """
    if opt.use_idf:
        vatex_train_corpus_path = os.path.join(opt.storage_path, 'vatex_train_en_annotations.json')
        vatex_train_corpus = json.load(open(vatex_train_corpus_path))
        vatex_train_corpus_list = []
        for vid in vatex_train_corpus:
            vatex_train_corpus_list.extend(vatex_train_corpus[vid])

        emscore_idf_dict = get_idf_dict(vatex_train_corpus_list, clip.tokenize, nthreads=4)
        # max token_id are eos token id
        # set idf of eos token are mean idf value
        emscore_idf_dict[max(list(emscore_idf_dict.keys()))] = sum(list(emscore_idf_dict.values()))/len(list(emscore_idf_dict.values()))
    else:
        emscore_idf_dict = False
    

    """
    Metric calculate
    """
    refs = np.array(refs)[:, :opt.use_n_refs].tolist()
    # results = metric.score(cands, refs, vids=vids)
    results = metric.score(cands, refs=refs, vids=vids, idf=emscore_idf_dict)
    
    
    if 'EMScore(X,V)' in results:
        print('EMScore(X,V) correlation --------------------------------------')
        # vid_figr_res_P = results['EMScore(X,V)']['figr_P']
        # vid_figr_res_R = results['EMScore(X,V)']['figr_R']
        # vid_figr_res_F = results['EMScore(X,V)']['figr_F']
        # vid_cogr_res = results['EMScore(X,V)']['cogr']
        # vid_full_res_P = results['EMScore(X,V)']['full_P']
        # vid_full_res_R = results['EMScore(X,V)']['full_R']
        vid_full_res_F = results['EMScore(X,V)']['full_F']
        # compute_correlation_uniquehuman(vid_figr_res_P.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(vid_figr_res_R.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(vid_figr_res_F.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(vid_cogr_res.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(vid_full_res_P.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(vid_full_res_R.numpy(), all_human_scores)
        compute_correlation_uniquehuman(vid_full_res_F.numpy(), all_human_scores)


    if 'EMScore(X,X*)' in results:
        print('EMScore(X,X*) correlation --------------------------------------')

        # refs_figr_res_P = results['EMScore(X,X*)']['figr_P']
        # refs_figr_res_R = results['EMScore(X,X*)']['figr_R']
        # refs_figr_res_F = results['EMScore(X,X*)']['figr_F']
        # refs_cogr_res = results['EMScore(X,X*)']['cogr']
        # refs_full_res_P = results['EMScore(X,X*)']['full_P']
        # refs_full_res_R = results['EMScore(X,X*)']['full_R']
        refs_full_res_F = results['EMScore(X,X*)']['full_F']
        # compute_correlation_uniquehuman(refs_figr_res_P.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(refs_figr_res_R.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(refs_figr_res_F.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(refs_cogr_res.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(refs_full_res_P.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(refs_full_res_R.numpy(), all_human_scores)
        compute_correlation_uniquehuman(refs_full_res_F.numpy(), all_human_scores)


    if 'EMScore(X,V,X*)' in results:
        print('EMScore(X,V,X*) correlation --------------------------------------')
        # vid_refs_figr_res_P = results['EMScore(X,V,X*)']['figr_P']
        # vid_refs_figr_res_R = results['EMScore(X,V,X*)']['figr_R']
        # vid_refs_figr_res_F = results['EMScore(X,V,X*)']['figr_F']
        # vid_refs_cogr_res = results['EMScore(X,V,X*)']['cogr']
        # vid_refs_full_res_P = results['EMScore(X,V,X*)']['full_P']
        # vid_refs_full_res_R = results['EMScore(X,V,X*)']['full_R']
        vid_refs_full_res_F = results['EMScore(X,V,X*)']['full_F']
        # compute_correlation_uniquehuman(vid_refs_figr_res_P.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(vid_refs_figr_res_R.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(vid_refs_figr_res_F.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(vid_refs_cogr_res.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(vid_refs_full_res_P.numpy(), all_human_scores)
        # compute_correlation_uniquehuman(vid_refs_full_res_R.numpy(), all_human_scores)
        compute_correlation_uniquehuman(vid_refs_full_res_F.numpy(), all_human_scores)
