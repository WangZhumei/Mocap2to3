##############
# Most of them are from https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/mld/models/modeltype/mld.py
##############
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities import rank_zero_only
from einops import einsum, rearrange, repeat
from hmr4d.utils.metric_utils import ListAggregator
from hmr4d.utils.pylogger import Log
from hmr4d.utils.check_utils import check_equal_get_one
from hmr4d.model.mas.utils.motion3d_endecoder import Hmlvec263OriginalEnDecoder
from hydra.utils import instantiate
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.hml3d.metric import (
    euclidean_distance_matrix,
    calculate_top_k,
    calculate_diversity_np,
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_multimodality_np,
)
import hmr4d.network.evaluator.t2m_motionenc as t2m_motionenc
import hmr4d.network.evaluator.t2m_textenc as t2m_textenc
from hmr4d.network.evaluator.word_vectorizer import POS_enumerator
from hmr4d.utils.eval.eval_utils import compute_pampjpe_metrics,compute_pampjpe_t_metrics,compute_wampjpe_metrics

def is_target_task(batch):
    task = check_equal_get_one(batch["task"], "task")
    return task == "3D"


class MetricPAMPJPEGlobal(pl.Callback):
    def __init__(
        self,
        endecoder_opt,
        top_k=3,
        R_size=32,
        diversity_times=300,
        is_export=False,
        exp_name=None,
        data_name=None,
        postfix="",
    ):
        super().__init__()
        self.endecoder_opt = endecoder_opt
        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times
        self.is_export = is_export
        self.exp_name = exp_name
        self.data_name = data_name
        if postfix != "":
            postfix = "_" + postfix
        self.postfix = postfix

        self.text_embeddings = ListAggregator()
        self.gen_motion_embeddings = ListAggregator()
        self.gt_motion_embeddings = ListAggregator()
        self.gt_motion = ListAggregator()
        self.pred_motion = ListAggregator()
        self.pampjpe = ListAggregator()
        self.mpjpe = ListAggregator()
        self.abs_mpjpe = ListAggregator()
        self.w_mpjpe = ListAggregator()
        self.wa_mpjpe = ListAggregator()
        # self._get_t2m_evaluator()
        self.res = {}

        self.data_endecoder: Hmlvec263OriginalEnDecoder = instantiate(endecoder_opt, _recursive_=False)
        self.encoder_motion3d = self.data_endecoder.encode

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Send models to GPU
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    def _get_t2m_evaluator(self):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        ######
        # OPT is from https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/networks/evaluator_wrapper.py
        ######
        opt = {
            "dim_word": 300,
            "max_motion_length": 196,
            "dim_pos_ohot": len(POS_enumerator),
            "dim_motion_hidden": 1024,
            "max_text_len": 20,
            "dim_text_hidden": 512,
            "dim_coemb_hidden": 512,
            "dim_pose": 263,
            "dim_movement_enc_hidden": 512,
            "dim_movement_latent": 512,
            "checkpoints_dir": "./inputs/checkpoints/t2m",
            "unit_length": 4,
        }
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=opt["dim_word"],
            pos_size=opt["dim_pos_ohot"],
            hidden_size=opt["dim_text_hidden"],
            output_size=opt["dim_coemb_hidden"],
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=opt["dim_pose"] - 4,
            hidden_size=opt["dim_movement_enc_hidden"],
            output_size=opt["dim_movement_latent"],
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=opt["dim_movement_latent"],
            hidden_size=opt["dim_motion_hidden"],
            output_size=opt["dim_coemb_hidden"],
        )
        # load pretrianed
        # t2m_checkpoint = torch.load(
        #     os.path.join(opt["checkpoints_dir"], "text_mot_match/model/finest.tar"),
        #     map_location="cpu",
        # )
        # self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        # self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        # self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])
        self.res = {}
        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        The behaviour is the same for val/test/predict
        """
        if not is_target_task(batch):
            return

        B = batch["length"].shape[0]
        length = batch["length"]
        device = batch["length"].device

        pred_motion = outputs["pred_global_motion"] 
        B, L, J, _ = pred_motion.shape
        pred_ayfz_motion = pred_motion  

        
        for i, l in enumerate(length):
            pred_ayfz_motion[i, l:] = 0.0

        gt_ayfz_motion = batch["gt_motion"] 
        # word_embs = batch["word_embs"]
        # pos_onehot = batch["pos_onehot"]
        # text_length = batch["text_len"]
        seq_name = batch["data_info"]
        start_frame = batch["start_frame"]
        end_frame = batch["end_frame"]
        
        # Check if pred_motion has nan, if so, set it to zero and print a warning
        if torch.isnan(pred_ayfz_motion).any():
            nan_mask = torch.isnan(pred_ayfz_motion)
            num_nan_item = (nan_mask.view(B, -1).any(dim=-1)).sum()
            Log.warn(f"{num_nan_item} pred_motion has nan")
            pred_ayfz_motion[nan_mask] = 0

        
        self.gt_motion.update(gt_ayfz_motion)
        self.pred_motion.update(pred_ayfz_motion)
        
        
        pa_mpjpe_b_m = []
        mpjpe_b_m = []
        abs_mpjpe_b_m = []
        w_mpjpe_b_m = []
        wa_mpjpe_b_m = []
        for b in range(B):
            l = length[b]
            if seq_name[b] not in self.res:
                self.res[seq_name[b]] = {}
            self.res[seq_name[b]][start_frame[b].cpu()] = {
                'gt_ayfz_motion' : gt_ayfz_motion[b,:l],
                'pred_ayfz_motion' : pred_ayfz_motion[b,:l],
                'start_frame' : start_frame[b].cpu(),
                'end_frame' : end_frame[b].cpu(),
            }
            pa_mpjpe_b,mpjpe_b,abs_mpjpe_b = compute_pampjpe_t_metrics(gt_ayfz_motion[b,:l], pred_ayfz_motion[b,:l])
            w_mpjpe_b,wa_mpjpe_b = compute_wampjpe_metrics(gt_ayfz_motion[b,:l], pred_ayfz_motion[b,:l])
            pa_mpjpe_b_m.append(pa_mpjpe_b.mean())
            mpjpe_b_m.append(mpjpe_b.mean())
            abs_mpjpe_b_m.append(abs_mpjpe_b.mean())
            w_mpjpe_b_m.append(w_mpjpe_b.mean())
            wa_mpjpe_b_m.append(wa_mpjpe_b.mean())
        self.pampjpe.update(torch.from_numpy(np.array([np.array(pa_mpjpe_b_m).mean()])))
        self.mpjpe.update(torch.from_numpy(np.array([np.array(mpjpe_b_m).mean()])))
        self.abs_mpjpe.update(torch.from_numpy(np.array([np.array(abs_mpjpe_b_m).mean()])))
        self.w_mpjpe.update(torch.from_numpy(np.array([np.array(w_mpjpe_b_m).mean()])))
        self.wa_mpjpe.update(torch.from_numpy(np.array([np.array(wa_mpjpe_b_m).mean()])))

        

    def on_predict_epoch_start(self, trainer, pl_module):
        self.gt_motion.reset()
        self.pred_motion.reset()
        self.pampjpe.reset()
        self.mpjpe.reset()
        self.abs_mpjpe.reset()
        self.w_mpjpe.reset()
        self.wa_mpjpe.reset()

    # ================== Epoch Summary  ================== #
    @rank_zero_only
    def on_predict_epoch_end(self, trainer, pl_module):
        metrics = {}
        pa_mpjpe =  self.pampjpe.get_tensor()
        mpjpe =  self.mpjpe.get_tensor()
        abs_mpjpe =  self.abs_mpjpe.get_tensor()
        w_mpjpe =  self.w_mpjpe.get_tensor()
        wa_mpjpe =  self.wa_mpjpe.get_tensor()

        
        pa_mpjpe = pa_mpjpe.detach().cpu().numpy()
        mpjpe = mpjpe.detach().cpu().numpy()
        abs_mpjpe = abs_mpjpe.detach().cpu().numpy()
        w_mpjpe = w_mpjpe.detach().cpu().numpy()
        wa_mpjpe = wa_mpjpe.detach().cpu().numpy()
        metrics["pa-mpjpe"] = pa_mpjpe.mean()
        metrics["mpjpe"] = mpjpe.mean()
        metrics["abs_mpjpe"] = abs_mpjpe.mean()
        metrics["w_mpjpe"] = w_mpjpe.mean()
        metrics["wa_mpjpe"] = wa_mpjpe.mean()


        # log to stdout
        for k, v in metrics.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = v.item()
            Log.info(f"{k}: {v:.3f}")

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                pl_module.logger.log_metrics({f"val_metric/{k}": v}, step=cur_epoch)

        self.gt_motion.reset()
        self.pred_motion.reset()
        self.pampjpe.reset()
        self.mpjpe.reset()
        self.abs_mpjpe.reset()
        self.w_mpjpe.reset()
        self.wa_mpjpe.reset()
       
        os.makedirs("./res", exist_ok=True)
        torch.save(self.res,'./res/rich_smpl.pth')
        
