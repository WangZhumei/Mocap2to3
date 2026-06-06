import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
import hmr4d.utils.matrix as matrix
from hmr4d.utils.pylogger import Log

from diffusers.schedulers import DDPMScheduler
from hmr4d.utils.diffusion.pipeline_helper import PipelineHelper
from hmr4d.model.mas.utils.motion2d_endecoder import EnDecoderBase

from hmr4d.utils.geo.triangulation import triangulate_ortho, triangulate_persp
from hmr4d.dataset.motionx.utils import normalize_keypoints_to_patch
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    project_p2d,
    cvt_to_bi01_p2d,
    cvt_from_bi01_p2d,
    cvt_p2d_from_i_to_c,
)
from hmr4d.utils.diffusion.utils import randlike_shape
from hmr4d.utils.check_utils import check_equal_get_one


class MAS_Offset_Pipeline(nn.Module, PipelineHelper):
    def __init__(self, args, args_clip, args_denoiser2d, **kwargs):
        """
        Args:
            args: pipeline
            args_clip: clip
            args_denoiser2d: denoiser2d network
        """
        super().__init__()
        self.args = args
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        self.record_interval = self.num_inference_steps // args.num_visualize if args.num_visualize > 0 else torch.inf

        # ----- Scheduler ----- #
        self.tr_scheduler = DDPMScheduler(**args.scheduler_opt_train)
        self.te_scheduler = instantiate(args.scheduler_opt_sample)
        self.te3d_scheduler = instantiate(args.scheduler_opt_sample3d)

        # ----- Networks ----- #
        self.clip = instantiate(args_clip, _recursive_=False)
        self.denoiser2d = instantiate(args_denoiser2d, _recursive_=False)
        Log.info(self.denoiser2d)

        # Functions for En/Decoding from motion to x, including normalization
        self.data_endecoder: EnDecoderBase = instantiate(args.endecoder_opt, _recursive_=False)
        self.encoder_motion2d = self.data_endecoder.encode
        self.decoder_motion2d = self.data_endecoder.decode

        # ----- Freeze ----- #
        self.freeze_clip()

    def freeze_clip(self):
        Log.info("Freezing CLIP")
        self.clip.eval()
        self.clip.requires_grad_(False)

    # ========== Training ========== #
    @staticmethod
    def build_model_kwargs(x, timesteps, length, f_condition, enable_cfg=False):
        if enable_cfg:
            length = torch.cat([length, length])
            for k in f_condition.keys():
                if k == "f_text":
                    pass
                else:
                    f_condition[k] = torch.cat([torch.zeros_like(f_condition[k]), f_condition[k]])
        return dict(x=x, timesteps=timesteps, length=length, **f_condition)

    def forward_train(self, inputs):
        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample
        motion = inputs["gt_motion2d"]  # (B, L, J, 3) #64, 300, 22, 2
        scheduler = self.tr_scheduler
        B, L, J, _ = motion.shape

        # *. Encoding
        x = self.encoder_motion2d(motion)  # (B, C, L)

        # *. Add noise
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)

        # Conditions
        assert self.training
        f_condition = {"f_text": inputs["text"]}

        if self.args.is_camemb:
            f_condition["f_RT"] = inputs["RT_emb"]  # (B, C)
        if self.args.is_Kemb:
            f_condition["f_K"] = inputs["K_emb"]  # (B, C)
        if self.args.is_pointmap:
            f_condition["f_pm"] = inputs["pointmap"]  # (B,V, C)
            #f_condition["pm_mask"] = inputs["pointmap_mask"]  # (B,V, C)
            

        f_condition = randomly_set_null_condition(f_condition)
        clip_text = self.clip.encode_text(f_condition["f_text"], enable_cfg=False, with_projection=True)
        f_condition["f_text"] = clip_text.f_text  # (B, D)

        # Forward
        model_kwargs = self.build_model_kwargs(x=noisy_x, timesteps=t, length=length, f_condition=f_condition)
        model_output = self.denoiser2d(**model_kwargs)
        model_pred = model_output.sample  # (B, C, L)
        mask = model_output.mask  # (B, 1, L)

        prediction_type = scheduler.config.prediction_type
        if prediction_type == "sample":
            target = x
        else:
            assert prediction_type == "epsilon"
            target = noise
        if mask is not None:
            model_pred = model_pred * mask
            target = target * mask

           
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        outputs["simple_loss"] = loss
        outputs["loss"] = loss
        return outputs

    def cfg_denoise_func(self, denoiser, model_kwargs, scheduler, enable_cfg):
        x = model_kwargs.pop("x")
        t = model_kwargs.pop("timesteps")

        # expand the x if we are doing classifier free guidance
        x_model_input = torch.cat([x] * 2) if enable_cfg else x
        x_model_input = scheduler.scale_model_input(x_model_input, t)

        # predict
        denoiser_out = denoiser(x_model_input, t, **model_kwargs)
        noise_pred = denoiser_out.sample

        # classifier-free guidance
        if enable_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # a special case since our motion prior is predicting x0 # TODO: extra check may be needed
        x0_ = noise_pred
        return x0_, denoiser_out

    

def randomly_set_null_condition(f_condition, uncond_prob=0.1):
    """
    To support classifier-free guidance, randomly set-to-unconditioned,
    Conditions are in shape (B, L, C)
    f_text: List of str
    """
    B = len(f_condition["f_text"])

    keys = list(f_condition.keys())
    for k in keys:
        if k == "f_text":
            # text
            text = f_condition[k]
            text_mask = torch.rand(B) < uncond_prob
            text_ = ["" if m else t for m, t in zip(text_mask, text)]
            f_condition[k] = text_
        else:
            f_condition[k] = f_condition[k].clone()
            mask = torch.rand(B, f_condition[k].shape[1]) < uncond_prob
            f_condition[k][mask] = 0.0

    return f_condition


def localmotion_withnr_to_globalmotion(pred_motion):
    # convert local motion with next root prediction to global motion
    # pred_motion: (..., L, J+1, 3)
    J = pred_motion.shape[-2]
    assert J == 23, "Only support SMPL now!"

    # Assume last joints is next frame root
    accum_root = torch.cumsum(pred_motion[..., -1:, :], dim=-3)  # (..., L, 1, 3)
    # Convert accumlated world pose to true world pose
    accum_pred_motion = pred_motion.clone()
    # Add accumulated root to each joints instead of virtual next frame root
    accum_pred_motion[..., 1:, :-1, :] += accum_root[..., :-1, :, :]
    # Assign virtual next frame root
    accum_pred_motion[..., -1:, :] = accum_root
    # Remove next root
    global_pred_motion = accum_pred_motion[..., :-1, :]
    return global_pred_motion
