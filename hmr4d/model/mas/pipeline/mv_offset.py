import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log

from diffusers.schedulers import DDPMScheduler
from diffusers.schedulers import DDIMScheduler
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
from hmr4d.model.mas.pipeline.mas import localmotion_withnr_to_globalmotion
from hmr4d.utils.diffusion.utils import randlike_shape
from hmr4d.utils.check_utils import check_equal_get_one


class MV_Offset_Pipeline(nn.Module, PipelineHelper):
    def __init__(self, args, args_clip, args_denoisermv, **kwargs):
        """
        Args:
            args: pipeline
            args_clip: clip
            args_denoisermv: denoisermv network
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
        self.denoisermv = instantiate(args_denoisermv, _recursive_=False)
        Log.info(self.denoisermv)

        if self.args.noise2d_scale > 0:
            Log.info(f"Use noise scale = {self.args.noise2d_scale} on 2d inputs") #0.1

        # Functions for En/Decoding from motion to x, including normalization
        self.data_endecoder: EnDecoderBase = instantiate(args.endecoder_opt.decoder, _recursive_=False)
        self.encoder_motion2d = self.data_endecoder.encode
        self.decoder_motion2d = self.data_endecoder.decode

        self.data_endecoder_global: EnDecoderBase = instantiate(args.endecoder_opt.decoder_global, _recursive_=False)
        self.encoder_motion2d_global = self.data_endecoder_global.encode
        self.decoder_motion2d_global = self.data_endecoder_global.decode


        # ----- Freeze ----- #
        self.freeze_clip()
        # ----- Freeze 2D diffusion ----- #
        if self.args.is_freeze2d:
            self.freeze_denoiser2d()

    def freeze_clip(self):
        Log.info("Freezing CLIP")
        self.clip.eval()
        self.clip.requires_grad_(False)

    def freeze_denoiser2d(self):
        Log.info("Freezing 2D denoiser")
        self.denoisermv.freeze()

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
        motion = inputs["gt_motion2d"]  # (B, V, L, J, 3)
        motion_global = inputs["gt_motion2d_global"]  # (B, V, L, J, 3)
        T_w2c = inputs["T_w2c"]  # (B, V, 4, 4)
        scheduler = self.tr_scheduler
        B, V, L, J, _ = motion.shape
        # *. Encoding
        x = self.encoder_motion2d(motion)  # (B, V, C, L)
        x_g = self.encoder_motion2d_global(motion_global)  # (B, V, C, L)

        # *. Add noise
        noise, _ = get_view_noise((B, V, J, L), T_w2c, self.args.is_worldnoise, None,self.args.recenter)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)

        # Conditions
        assert self.training
        f_condition = {
            "f_text": inputs["text"], 
        }

        if self.args.is_2dinput:
            f_cond_2d = x_g[:, 0] 
            if self.args.noise2d_scale > 0:
                f_cond_2d = f_cond_2d + torch.randn_like(f_cond_2d) * self.args.noise2d_scale
            f_condition["f_cond2d"] = f_cond_2d  

        if self.args.is_bb2dpose:
            f_cond_b2d = x[:, 0]  
            if self.args.noise2d_scale > 0:
                f_cond_b2d = f_cond_b2d + torch.randn_like(f_cond_b2d) * self.args.noise2d_scale*5
            f_condition["f_condb2d"] = f_cond_b2d  
        
        if self.args.is_camemb:
            f_condition["f_RT"] = inputs["RT_emb"] 
        if self.args.is_Kemb:
            f_condition["f_K"] = inputs["K_emb"] 
        if self.args.is_pointmap:
            f_condition["f_pm"] = inputs["pointmap"] 

        f_condition = randomly_set_null_condition(f_condition, self.args.is_hastext)
        clip_text = self.clip.encode_text(f_condition["f_text"], enable_cfg=False, with_projection=True)
        f_condition["f_text"] = clip_text.f_text 

        # Forward
        model_kwargs = self.build_model_kwargs(x=noisy_x, timesteps=t, length=length, f_condition=f_condition)
        model_output = self.denoisermv(**model_kwargs)
        model_pred = model_output.sample  
        mask = model_output.mask  

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

    
def randomly_set_null_condition(f_condition, is_hastext, uncond_prob=0.1):
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
            if is_hastext:
                text_ = ["" if m else t for m, t in zip(text_mask, text)]
            else:
                text_ = [""] * B
            f_condition[k] = text_
        else:
            f_condition[k] = f_condition[k].clone()
            mask = torch.rand(B, f_condition[k].shape[1]) < uncond_prob
            f_condition[k][mask] = 0.0

    return f_condition


def get_view_noise(noise_shape, T_w2c, is_worldnoise=True, generator=None,recenter=False):
    """Get view noise (BV, J*2, L)
    Args:
        is_worldnoise: if True, the noise is multiview-consistent
    """
    B, V, J, max_L = noise_shape
    if is_worldnoise:
        if generator is not None:
            noise3d = randlike_shape((B, max_L * J, 3), generator)
        else:
            noise3d = torch.randn((B, max_L * J, 3), device=T_w2c.device)
        view_noise3d = apply_T_on_points(  # (BVL, J, 3)
            repeat(noise3d, "b (l j) c -> (b v l) j c", v=V, l=max_L),
            repeat(T_w2c, "b v c d -> (b v l) c d", l=max_L),
        )
        view_noise = view_noise3d[..., [0, 1]]

        if recenter:
            zeros = torch.zeros_like(noise3d)[:,:1] 
            view_zero3d = apply_T_on_points(  # (BVL, J, 3) [1200, 1, 3] 8, 1, 3
                    repeat(zeros, "b j c -> (b v) j c", b=B, v=V),
                    rearrange(T_w2c, "b v c d -> (b v) c d", b=B, v=V),
                )
            zeros_noise = repeat(view_zero3d, "(b v) j c -> b v l j c", b=B, v=V, l=max_L)[...,[0], [0, 1]] #[2, 4, 1, 3]
            zeros_noise = repeat(zeros_noise, "b v l c -> b v l j c", j = J)#[2, 4, 1, 3]
            view_noise_ = rearrange(view_noise, "(b v l) j c -> b v l j c", b=B, v=V, l=max_L)
            view_noise_ = view_noise_-zeros_noise
            view_noise = rearrange(view_noise_, "b v l j c -> b v (j c) l", b=B, v=V, l=max_L)
        else:
            view_noise = rearrange(view_noise, "(b v l) j c -> b v (j c) l", b=B, v=V, l=max_L)
    else:
        if generator is not None:
            view_noise = randlike_shape((B, V, J * 2, max_L), generator)
            noise3d = randlike_shape((B, max_L * J, 3), generator)
        else:
            view_noise = torch.randn((B, V, J * 2, max_L), device=T_w2c.device)
            noise3d = torch.randn((B, max_L * J, 3), device=T_w2c.device)
    return view_noise, noise3d  # view_noise is (B, V, J*2, L), noise3d is (B, L*J, 3)
