import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log

from diffusers.schedulers import DDPMScheduler
from hmr4d.utils.diffusion.pipeline_helper import PipelineHelper
from hmr4d.model.mas.utils.motion2d_endecoder import EnDecoderBase
from hmr4d.model.mas.pipeline.mas import localmotion_withnr_to_globalmotion
from hmr4d.dataset.motionx.utils import normalize_keypoints_to_patch,normalize_kp_2d,normalize_kp_2d_linear
from hmr4d.model.mas.pipeline.mv import get_view_noise

from hmr4d.utils.geo.triangulation import triangulate_ortho, triangulate_persp
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    project_p2d,
    cvt_to_bi01_p2d,
    cvt_from_bi01_p2d,
    cvt_p2d_from_i_to_c,
)
from hmr4d.utils.diffusion.utils import randlike_shape
from hmr4d.utils.check_utils import check_equal_get_one
from hmr4d.dataset.motionx.utils import adjust_K
import copy
import numpy as np
import cv2

class MV_Offset_FromPredPipeline(nn.Module, PipelineHelper):
    def __init__(self, args, args_clip, args_denoiser2d, args_denoisermv, **kwargs):
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
        Log.info("--------- denoiser 2d ---------")
        Log.info(self.denoiser2d)
        Log.info("----- denoiser multi-view -----")
        self.denoisermv = instantiate(args_denoisermv, _recursive_=False)
        Log.info(self.denoisermv)
        Log.info("--------------------------------")

        # NOTE: Do not check weights equal here, the weights are not loaded here.

        # Functions for En/Decoding from motion to x, including normalization
        self.data_endecoder: EnDecoderBase = instantiate(args.endecoder_opt.decoder, _recursive_=False)
        self.encoder_motion2d = self.data_endecoder.encode
        self.decoder_motion2d = self.data_endecoder.decode

        self.data_endecoder_global: EnDecoderBase = instantiate(args.endecoder_opt.decoder_global, _recursive_=False)
        self.encoder_motion2d_global = self.data_endecoder_global.encode
        self.decoder_motion2d_global = self.data_endecoder_global.decode


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
        print("This pipeline only supports inference!")
        raise NotImplementedError

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

    # ========== Sample 2D ========== #
    def forward_sample(self, inputs):
        print("Multi-view generation uses forward_sample3d")
        raise NotImplementedError

    # ========== Sample 3D ========== #
    #default
    def forward_sample3d(self, inputs):
        # Setup
        outputs = dict()
        generator = inputs["generator"]
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.te3d_scheduler

        length = inputs["length"]
        B = inputs["length"].shape[0]
        gt_motion2d = inputs["gt_motion2d"]  
        gt_motion2d_global = inputs["gt_motion2d_global"]  
        max_L = gt_motion2d.shape[2]

        w_gt_3d = inputs["gt_motion"]
        T_w2c = inputs["T_w2c"] 
        Ks = inputs["Ks"]  
        is_pinhole = inputs["is_pinhole"][0].item()
        patch_size = inputs["patch_size"][0].item()
        B, V = T_w2c.shape[:2]
        BV = B * V
        BL = B * max_L
        BVL = B * V * max_L
        J = self.data_endecoder.J

        triangulate_func = triangulate_persp if is_pinhole else triangulate_ortho

        # Functions
        def cvt_x_to_c_p2d_global(x):
            try:
                normed_p2d = self.decoder_motion2d_global(x) 
            except RuntimeError:
                # separate process zero root
                normed_p2d = self.decoder_motion2d_global(x[..., 2:, :]) 
                
            normed_p2d = rearrange(normed_p2d, "b v l j c -> (b v l) j c", b=B, v=V, l=max_L) 
            i_p2d = normalize_keypoints_to_patch(normed_p2d, crop_size=patch_size, inv=True) 
            if len(Ks.size())== 4:
                c_p2d = cvt_p2d_from_i_to_c(i_p2d, repeat(Ks, "b v c d -> (b v l) c d", l=max_L))  
            elif len(Ks.size())== 5:
                c_p2d = cvt_p2d_from_i_to_c(i_p2d, rearrange(Ks, "b v l c d -> (b v l) c d"))  

            c_p2d = rearrange(c_p2d, "(b v l) j c -> b v (l j) c", b=B, v=V, l=max_L) 
            return c_p2d
        def cvt_x_to_c_p2d(x):
            try:
                normed_p2d = self.decoder_motion2d(x)  
            except RuntimeError:
                # separate process zero root
                normed_p2d = self.decoder_motion2d(x[..., 2:, :])  
                
            normed_p2d[...,1:-1,:] = normed_p2d[...,1:-1,:]*(normed_p2d[...,-1:,:]+1)/2+normed_p2d[...,:1,:] 

            normed_p2d = rearrange(normed_p2d, "b v l j c -> (b v l) j c", b=B, v=V, l=max_L)  
            i_p2d = normalize_keypoints_to_patch(normed_p2d, crop_size=patch_size, inv=True) 
            if len(Ks.size())== 4:
                c_p2d = cvt_p2d_from_i_to_c(i_p2d, repeat(Ks, "b v c d -> (b v l) c d", l=max_L))
            elif len(Ks.size())== 5:
                c_p2d = cvt_p2d_from_i_to_c(i_p2d, rearrange(Ks, "b v l c d -> (b v l) c d")) 

            c_p2d = rearrange(c_p2d, "(b v l) j c -> b v (l j) c", b=B, v=V, l=max_L) 
            return c_p2d

        
        def cvt_w_p3d_to_x_bbox_wh(w_p3d):
            c_p3d = apply_T_on_points(
                repeat(w_p3d, "b (l j) c -> (b v l) j c", v=V, l=max_L),
                repeat(T_w2c, "b v c d -> (b v l) c d", l=max_L),
            )  
            if len(Ks.size())== 4:
                i_p2d = project_p2d(
                    c_p3d, repeat(Ks, "b v c d -> (b v l) c d", l=max_L), is_pinhole=is_pinhole
                )  
            elif len(Ks.size())== 5:
                i_p2d = project_p2d(
                    c_p3d, rearrange(Ks, "b v l c d -> (b v l) c d"), is_pinhole=is_pinhole
                )  
            normed_p2d = normalize_keypoints_to_patch(i_p2d, crop_size=patch_size)  
            normed_p2d = rearrange(normed_p2d, "(b v l) j c -> b v l j c", b=B, v=V, l=max_L)

            i_p2d = rearrange(i_p2d, "(b v l) c d -> b v l c d", b=B, v=V, l=max_L)
            normed_motion2ds = []
            bboxs = []
            for b in range(B):
                t_norm = []
                t_bbox = []
                for v in range(V):
                    normed_pred_motion2d,_,bbox =normalize_kp_2d_linear(i_p2d[b,v].cpu()) 
                    t_norm.append(normed_pred_motion2d)
                    t_bbox.append(bbox)
                t_norm = torch.stack(t_norm)
                t_bbox = torch.stack(t_bbox)
                normed_motion2ds.append(t_norm)
                bboxs.append(t_bbox)
            normed_motion2ds = torch.stack(normed_motion2ds).to(i_p2d.device) 
            bboxs = torch.stack(bboxs).to(i_p2d.device) #b v l 3

            w_h = normalize_keypoints_to_patch(bboxs[...,2] * 200, crop_size=patch_size)[...,None,None] 
            w_h = torch.cat([w_h,w_h],dim=-1)
            
            normed_motion2ds[...,0,:] = normed_p2d[...,0,:]
            normed_motion2ds = torch.cat([normed_motion2ds,w_h],dim=-2)
            
            try:
                x = self.encoder_motion2d(normed_motion2ds) 
            except RuntimeError:
                # Separate process zero root
                x = self.encoder_motion2d(normed_motion2ds)  # (B, V, J*2, L)
            return x  

        
        # 1. Prepare target variable x, which will be denoised progressively
        x, noise3d = get_view_noise(
            (B, V, J, max_L), T_w2c, self.args.is_worldnoise, generator, self.args.recenter
        )  # in the data space after normalization

        
        # 2. Conditions
        # Encode CLIP embedding
        text = inputs["text"]

        # assign text with zero when no cfg
        if not enable_cfg or not self.args.is_hastext:
            text = ["" for _ in range(len(text))]

        clip_text = self.clip.encode_text(text, enable_cfg=enable_cfg, with_projection=True)
        # make f_text number with N_views
        f_text = clip_text.f_text  # (B, D)

        gt_x_g = self.encoder_motion2d_global(gt_motion2d_global)  # (B, V, C, L)
        gt_x = self.encoder_motion2d(gt_motion2d)  # (B, V, C, L)

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.num_inference_steps * scheduler.order
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(scheduler, generator)  # for scheduler.step()
        pred_progress = {}  # for visualization
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            f_condition = {
                "f_text": f_text, 
            }
            if self.args.is_2dinput:
                f_condition["f_cond2d"] = gt_x_g[:, 0] 
            if self.args.is_bb2dpose:
                f_condition["f_condb2d"] = gt_x[:, 0] 
            if self.args.is_camemb:
                f_condition["f_RT"] = inputs["RT_emb"]  
            if self.args.is_Kemb:
                f_condition["f_K"] = inputs["K_emb"]  
            if self.args.is_pointmap:
                f_condition["f_pm"] = inputs["pointmap"]  

            model_kwargs = self.build_model_kwargs(
                x=x,
                timesteps=t,
                length=length,
                f_condition=f_condition,
                enable_cfg=enable_cfg,
            )

            x0_, denoiser_out = self.cfg_denoise_func(self.denoisermv, model_kwargs, scheduler, enable_cfg)
            # (B, V, J*2, L)

            view_noise, _ = get_view_noise(
                (B, V, J, max_L), T_w2c, self.args.is_worldnoise, generator,self.args.recenter
            )  # (B, V, J*2, L)
            extra_step_kwargs["view_noise"] = view_noise  # (B, V, J*2, L)

            scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
            x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            
            c_p2d = cvt_x_to_c_p2d(x0_)  
            gt_c_p2d = cvt_x_to_c_p2d_global(gt_x_g) 

            
            gt_c_p2d_ = cvt_x_to_c_p2d_global(gt_x_g)  
            combine_c_p2d = copy.deepcopy(c_p2d)
            combine_c_p2d[:,0,:,:] = gt_c_p2d_[:,0,:,:]
            
            combine_c_p2d = rearrange(combine_c_p2d, "b v (l j) c-> b v l j c", l=max_L,j=J)
            combine_c_p2d = combine_c_p2d[...,:-1,:]
            combine_c_p2d = rearrange(combine_c_p2d, "b v l j c-> b v (l j) c", l=max_L,j=J-1)
            w_p3d = triangulate_func(T_w2c, combine_c_p2d)  # (B, L*J, 3)

            # for vis
            w_p3d_ = rearrange(w_p3d, "b (l j) c -> b l j c", l=max_L, c=3)  # (B, L, J, 3)
            c_p2d_ = rearrange(c_p2d, "b v (l j) c -> b v l j c", v=V, l=max_L, c=2)[...,:-1,:]  # (B, V, L, J, 2)
            gt_c_p2d_ = rearrange(gt_c_p2d, "b v (l j) c -> b v l j c", v=V, l=max_L, c=2)[...,:-1,:]  # (B, V, L, J, 2)

            
            # *. Update x0_
            if self.args.is_consisblock:
                x0_ = cvt_w_p3d_to_x_bbox_wh(w_p3d)
                x0_ = torch.cat([gt_x[:,:1,:,:], x0_[:,1:,:,:]], dim=1)
#
                # Use posterior p(x{t-1} | xt, x0) with projected x0
                scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
                x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            # *. Update and store intermediate results
            x = xprev_  
            
            if i % self.record_interval == 0:
                if "pred_motion" not in pred_progress.keys():
                    pred_progress["pred_motion"] = []
                    pred_progress["pred_motion2d"] = []
                pred_progress["pred_motion"].append(w_p3d_)  
                gt_c_p2d_[...,0,:] = 0
                c_p2d_ = torch.cat([gt_c_p2d_[:, :1], c_p2d_[:, 1:]], dim=1)
                pred_progress["pred_motion2d"].append(c_p2d_)  

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        # Post-processing
        outputs["pred_motion"] = w_p3d_  
        outputs["pred_global_motion"] = w_p3d_  
        for k in pred_progress.keys():
            pred_progress[k] = torch.stack(pred_progress[k], dim=1)  
        outputs["pred_progress"] = pred_progress

        
        return outputs
    
    
