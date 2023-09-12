import copy
from lib.trainer import Trainer
from collections import defaultdict
from tqdm import tqdm
from mbes_data.lib.evaluations import update_results, save_results_to_file
from mbes_data.common.misc import prepare_logger
from lib.benchmark_utils import ransac_pose_estimation, to_array
import torch
import numpy as np
import os

class MBESTester(Trainer):
    """
    Multibeam dataset tester for OverlapPredator
    """
    def __init__(self, args):
        Trainer.__init__(self, args)
        self.raw_config = args.raw_config


    def test(self):
        print('Start to evaluate on test dataset...')
        _logger, _log_path = prepare_logger(self.config, log_path=os.path.join(self.snapshot_dir,'results'))

        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        self.model.eval()

        total_rotation = []
        results = defaultdict(dict)
        self.config.exp_dir = self.config.snapshot_dir
        outdir = os.path.join(self.config.exp_dir, self.config.pretrain)
        os.makedirs(outdir, exist_ok=True)

        with torch.no_grad():
            for i, inputs in tqdm(enumerate(self.loader['test']), total=num_iter):
                if inputs is None:
                    print(f'Idx {i} is None! Skipping...')
                    continue
                for key in inputs.keys():
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].squeeze(0)
                try:
                    ##################################
                    # load inputs to device.
                    for k, v in inputs.items():  
                        if type(v) == list:
                            inputs[k] = [item.to(self.device) for item in v]
                        elif type(v) == dict:
                            pass
                        else:
                            inputs[k] = v.to(self.device)

                    rot_trace = inputs['sample']['transform_gt'][0, 0] + inputs['sample']['transform_gt'][1, 1] + \
                            inputs['sample']['transform_gt'][2, 2]
                    rot_trace = torch.tensor(rot_trace).to(self.device)
                    rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
                    total_rotation.append(np.abs(to_array(rotdeg)))

                    ###################################
                    # forward pass
                    feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
                    scores_overlap = scores_overlap.detach().cpu()
                    scores_saliency = scores_saliency.detach().cpu()

                    len_src = inputs['stack_lengths'][0][0]
                    src_feats, tgt_feats = feats[:len_src], feats[len_src:]
                    src_pcd , tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
                    src_overlap, tgt_overlap = scores_overlap[:len_src], scores_overlap[len_src:]
                    src_saliency, tgt_saliency = scores_saliency[:len_src], scores_saliency[len_src:]

                    
                    ########################################
                    # run probabilistic sampling
                    n_points = 5000
                    src_scores = src_overlap * src_saliency
                    tgt_scores = tgt_overlap * tgt_saliency

                    if(src_pcd.size(0) > n_points):
                        idx = np.arange(src_pcd.size(0))
                        probs = (src_scores / src_scores.sum()).numpy().flatten()
                        idx = np.random.choice(idx, size= n_points, replace=False, p=probs)
                        src_pcd, src_feats = src_pcd[idx], src_feats[idx]
                    if(tgt_pcd.size(0) > n_points):
                        idx = np.arange(tgt_pcd.size(0))
                        probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
                        idx = np.random.choice(idx, size= n_points, replace=False, p=probs)
                        tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]


                    ########################################
                    # run ransac 
                    distance_threshold = self.config.voxel_size * 1.5
                    ts_est = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False, distance_threshold=distance_threshold, 
                                                    ransac_n=self.config.ransac_n)
                    success = True
                except: # sometimes we left over with too few points in the bottleneck and our k-nn graph breaks
                    print(f'Input {i}: RANSAC failed, using identity matrix')
                    ts_est = np.eye(4)
                    success = False

                transform_pred = copy.deepcopy(ts_est)
                for k in inputs['sample'].keys():
                    if isinstance(inputs['sample'][k], np.ndarray):
                        inputs['sample'][k] = torch.from_numpy(inputs['sample'][k])
                data = inputs['sample']
                # record features into data dict
                data['feat_src_points'] = src_pcd
                data['feat_ref_points'] = tgt_pcd
                data['feat_src'] = src_feats
                data['feat_ref'] = tgt_feats
                # record registration success
                data['success'] = success
                results = update_results(results, data, transform_pred,
                                         self.raw_config, outdir, _logger)

        total_rotation = np.array(total_rotation)
        _logger.info(('Rotation range in data: {}(avg), {}(max)'.format(np.mean(total_rotation), np.max(total_rotation))))
        save_results_to_file(_logger, results, self.raw_config, outdir)
