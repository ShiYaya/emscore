import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import clip
import torch
import glob
from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from itertools import chain
from math import log
import argparse
from emscore import EMScorer


def process(a, tokenizer=None):
    if tokenizer is not None:
        a = tokenizer(a)[0].tolist()
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict


class EMScore_ANET_FOIL(object):

    def __init__(self, args, prediction_filename=None,
                 idf=True, verbose=False,):
        # For clip
        self.storage_path = args.storage_path
        self.verbose = verbose
        self.args = args

        self.vid_duration_dict = self.import_vid_duration()
        self.clip_prediction = self.import_emscore_prediction(prediction_filename)
        if idf:
            self.emscore_idf_dict = self.compute_emscore_idf()
        else:
            self.emscore_idf_dict = False
        self.vid_clip_feats = self.import_clip_vid_feats()
        self.cands_timestamp = self.get_cands_timestamp()
        self.refs_for_eval = self.import_refs_for_eval()


    def get_gt_vid_ids(self):
        pred_vid_ids = set(list(self.clip_prediction.keys()))
        exist_videos = [item.split('.')[0] for item in os.listdir(self.args.anet_vid_clip_feats_path)]
        gt_exist_videos = set(pred_vid_ids).intersection(set(exist_videos))
        return list(gt_exist_videos)

    def import_clip_vid_feats(self):
        def get_feats_dict(feat_dir_path, gt_vid_ids):
            file_path_list = glob.glob(feat_dir_path+'/*.pt')
            feats_dict = {}
            for file_path in file_path_list:
                vid = file_path.split('/')[-1][:-3]
                if vid not in gt_vid_ids:
                    continue
                data = torch.load(file_path)
                feats_dict[vid] = data
            return feats_dict
        gt_vid_ids = self.get_gt_vid_ids()
        vid_feat_dict = get_feats_dict(self.args.anet_vid_clip_feats_path, gt_vid_ids)
        assert len(vid_feat_dict.keys()) == len(gt_vid_ids)
        return vid_feat_dict

    def import_vid_duration(self):
        # the duration for each video 
        filenames = (os.path.join(self.storage_path, 'anet_entities_test_1.json'),
                     os.path.join(self.storage_path, 'anet_entities_test_2.json'))
        vid_duration_dict = {}
        for filename in filenames:
            gt = json.load(open(filename))
            for vid in gt:
                vid_duration_dict[vid] = gt[vid]['duration']

        return vid_duration_dict
    
    def get_cands_timestamp(self):
        ref_filename = os.path.join(self.storage_path, 'anet_entities_test_1.json')
        clip_refs = json.load(open(ref_filename))
        return clip_refs

    def import_refs_for_eval(self):
        ref_filename = os.path.join(self.storage_path, 'anet_entities_test_2.json')
        refs_for_eval = json.load(open(ref_filename))
        return refs_for_eval
    
    def compute_emscore_idf(self):
        print('compute emscore idf ..................')
        data = json.load(open(self.args.idf_corpus))
        train_corpus = []
        for vid in data:
            sents = data[vid]['sentences']
            new_sents = []
            for sent in sents:
                if len(sent.split(' ')) > 66: # Filter out too long sentences
                    continue
                else:
                    new_sents.append(sent)
            train_corpus.extend(new_sents)
        
        idf_dict = get_idf_dict(train_corpus, clip.tokenize, nthreads=4)
        idf_dict[max(list(idf_dict.keys()))] = sum(list(idf_dict.values()))/len(list(idf_dict.values()))
        return idf_dict

    def import_emscore_prediction(self, prediction_filename):
        pred = json.load(open(prediction_filename))
        return pred['results']

    def use_ref_timestamps(self, timestamp, ref_timestamps):
        use_ref_idxs = []
        # 如果timestamp 在 ref_timestamp 中的交集占 ref_timestamp 的50%以上，则使用该 ref_timestamp
        a = int(timestamp[0])  # a
        b = int(timestamp[1])  # b
        for ref_i, ref_time in enumerate(ref_timestamps):
            c = int(ref_time[0])  # c
            d = int(ref_time[1])  # d
            result = list(set(range(a, b+1)) &
                            set(range(c, d+1)))
            if len(result)/len(range(c, d+1)) > 0.4:
                use_ref_idxs.append(ref_i)
        return use_ref_idxs
                    
    
    def evaluate(self):
        gt_vid_ids = self.get_gt_vid_ids()
        self.filter_clip_prediction = {
            vid: self.clip_prediction[vid] for vid in gt_vid_ids}

        Use_Ref = self.args.use_references
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        vid_emscore_with_idf = {}
        count = 0
        vid_group_boundaries = []
        cands_list = []
        refs_list = []
        seg_feat_dict = {}
        seg_list = []
        for vid in tqdm(gt_vid_ids, desc='Computing EMScore'):
            vid_pred = self.filter_clip_prediction[vid]

            for sent_i, sent_segment in enumerate(vid_pred):
                cand_sent = sent_segment['sentence']  # 预测的caption
                cands_list.append(cand_sent)
                # timestamp = sent_segment['timestamp']
                ref_idx = sent_i
                timestamp = self.cands_timestamp[vid]['timestamps'][ref_idx]
                if Use_Ref:
                    ref = self.refs_for_eval[vid]['sentences']
                    ref_timestamps = self.refs_for_eval[vid]['timestamps']
                    use_ref_idx = self.use_ref_timestamps(timestamp, ref_timestamps)
                    if not use_ref_idx:
                        use_ref_idx = list(
                            range(len(ref_timestamps)))
                    refs_sent = [ref[idx] for idx in use_ref_idx]
                    refs_list.append(refs_sent)

                """
                使用 video作为 references
                """
                vid_feats = self.vid_clip_feats[vid]
                vid_frames_len = len(vid_feats)
                duration = self.vid_duration_dict[vid]
                start = timestamp[0]*vid_frames_len//duration
                end = timestamp[1]*vid_frames_len//duration
                fg_vid_segment_feat = vid_feats[int(start):int(end)]
                
                seg_feat_dict['{}_seg_{}'.format(vid, sent_i)] = fg_vid_segment_feat
                seg_list.append('{}_seg_{}'.format(vid, sent_i))
                
            vid_group_boundaries.append((count, count + len(vid_pred)))
            count += len(vid_pred)
            
        emscore_metric = EMScorer(vid_feat_cache=seg_feat_dict)
        vid_emscore_with_idf = emscore_metric.score(cands=cands_list, refs=refs_list, vids=seg_list, idf=self.emscore_idf_dict)
        for key in vid_emscore_with_idf:
            for item in vid_emscore_with_idf[key]:
                scores = []
                for beg, end in vid_group_boundaries:
                    scores.append(float(torch.mean(vid_emscore_with_idf[key][item][beg: end])))
                vid_emscore_with_idf[key][item] = scores
        
        final_vid_emscore = {}
        for key in vid_emscore_with_idf:
            for item in vid_emscore_with_idf[key]:
                final_vid_emscore[key + '_' + item] = vid_emscore_with_idf[key][item]
        return final_vid_emscore


def main(args):

    evaluator = EMScore_ANET_FOIL(args, prediction_filename=args.submission_right, idf=args.use_idf, verbose=args.verbose)
    right_vid_emscores = evaluator.evaluate()

    evaluator = EMScore_ANET_FOIL(args, prediction_filename=args.submission_foil, idf=args.use_idf, verbose=args.verbose)
    foil_vid_emscores = evaluator.evaluate()

    for key in right_vid_emscores:
        res = np.array(right_vid_emscores[key]) > np.array(foil_vid_emscores[key])
        res_sum = np.sum(res)
        print(key, res_sum, '{:.2f}'.format(100*res_sum/len(res)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', type=str,  default='',
                        help='the path you storage ActivityNet-FOIL dataset.')
    parser.add_argument('--verbose', default=True,
                        help='Print intermediate steps.')
    parser.add_argument('--use_references', action='store_true', default=True)
    parser.add_argument('--use_idf', action='store_true', default=True)

    args = parser.parse_args()

    args.submission_right = os.path.join(args.storage_path, 'final_right_video_sentences.json')    
    args.submission_foil = os.path.join(args.storage_path, 'final_foil_video_sentences.json') 
    args.idf_corpus = os.path.join(args.storage_path, 'train.json') 
    args.anet_vid_clip_feats_path = os.path.join(args.storage_path, 'ActivityNet-FOIL_video_feats') 

    main(args)