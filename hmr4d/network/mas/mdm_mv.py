import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmr4d.network.mas.mdm import length_to_mask, TimestepEmbedder
from hmr4d.network.base_arch.transformer_layer.encoder_mv import MVTransformerEncoderLayer
from hmr4d.network.base_arch.transformer_layer.decoder_mv import MVTransformerDecoderLayer
from hmr4d.network.base_arch.transformer_layer.sd_layer import zero_module
from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.configs import MainStore, builds
from hmr4d.network.resnet.resnet import ResNet

def length_to_mask_pm(length,pm_mask,f_pm_):
    if pm_mask is None:
        mask = torch.full((length.shape[0], f_pm_.shape[-2]), True, device=length.device)
        return mask
    mask = pm_mask>0
    return mask

class PMDM2DMV(nn.Module):
    def __init__(
        self,
        # x
        input_dim=44,
        # condition
        text_dim=512,
        cam_dim=4,
        RT_dim=6,
        K_dim=300*4,
        freeze_k=True,
        is_2dinput=True,
        is_bb2dpose=False,
        pointmap_dim = 0,
        # intermediate
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        # training
        dropout=0.1,
        extra_root=False,
        with_projection=True,
        **kargs,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.text_dim = text_dim
        self.RT_dim = RT_dim
        self.K_dim = K_dim
        self.freeze_k = freeze_k
        self.cam_dim = cam_dim
        self.is_2dinput = is_2dinput
        self.is_bb2dpose = is_bb2dpose
        self.pointmap_dim = pointmap_dim

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dropout = dropout
        self.extra_root = extra_root

        self.with_projection = with_projection
        self._build_model()

    def _build_model(self):
        self._build_input()
        self._build_condition()
        self._build_transformer()
        self._build_output()
        if self.extra_root:
            self.root_process = OutputProcess(4, self.latent_dim)

    def _build_input(self):
        self.input_process = InputProcess(self.input_dim, self.latent_dim)

    def _build_condition(self):
        # TODO: upgrade residual module
        self.embed_text = nn.Linear(self.text_dim, self.latent_dim)
        self.embed_RT = zero_module(nn.Linear(self.RT_dim, self.latent_dim))
        self.embed_K = zero_module(nn.Linear(self.K_dim, self.latent_dim))
        if self.cam_dim>0:
            self.embed_cam = zero_module(nn.Linear(self.cam_dim, self.latent_dim))
        if self.is_2dinput:
            self.embed_cond2d = zero_module(nn.Linear(self.latent_dim, self.latent_dim))
        if self.is_bb2dpose:
            self.embed_condb2d = zero_module(nn.Linear(self.latent_dim, self.latent_dim))
        if self.pointmap_dim>0: #temp
            self.encoder_PM = ResNet([2, 2, 2, 2])
    def _build_transformer(self):
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.pointmap_dim==0:
            seqTransEncoderLayer = MVTransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation="gelu",
            )
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        else:
            seqTransDecoderLayer = MVTransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation="gelu",
            )
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

    def _build_output(self):
        self.output_process = OutputProcess(self.input_dim, self.latent_dim)

    def forward(self, x, timesteps, length, f_text=None, f_cam=None, f_cond2d=None,f_condb2d=None, f_RT=None,f_K=None,f_pm=None,pm_mask=None):
        """
        Args:
            x: (B, V, C, L), a noisy motion sequence
            timesteps: (B,)
            length: (B), valid length of x
            f_text: (B, C)
            f_cam: (B, V, C)
            f_cond2d: (B, C, L)
        """
        B, V, _, L = x.shape
        # Set timesteps
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        assert len(timesteps) == x.shape[0]

        emb = self.embed_timestep(timesteps)  
        emb = emb[:, :, None] 
        if f_text is not None:
            f_ = self.embed_text(f_text)  
            emb = emb + f_[None, :, None] 
        if f_RT is not None:
            f_ = self.embed_RT(f_RT)  
            emb = emb + f_[None]  
        if f_K is not None:
            f_ = self.embed_K(f_K)  
            emb = emb + f_[None] 
        if f_cam is not None:
            f_ = self.embed_cam(f_cam) 
            emb = emb + f_[None] 
        if f_pm is not None: 
            f_pm_ = f_pm.reshape(B*V,self.pointmap_dim,self.pointmap_dim,3).permute(0, 3,1, 2).contiguous() 
            f_pm_ = self.encoder_PM(f_pm_)  
            f_pm_ = f_pm_.permute(0, 2,3, 1).contiguous().reshape(B,V,-1,self.latent_dim) 
        x = self.input_process(x)  

        if f_cond2d is not None:
            assert self.is_2dinput, "The network does not have 2d condition layer!"
            f_2d = self.input_process(f_cond2d)  
            f_2d = self.embed_cond2d(f_2d)  
            if self.with_projection:
                x = x + f_2d[:, :, None]  
        if f_condb2d is not None:
            assert self.is_bb2dpose, "The network does not have 2d condition layer!"
            f_b2d = self.input_process(f_condb2d)  
            f_b2d = self.embed_condb2d(f_b2d) 
            if self.with_projection:
                x = x + f_b2d[:, :, None]  

        # adding the timestep embed
        xseq = torch.cat((emb, x), dim=0) 
        xseq = self.sequence_pos_encoder(xseq)  

        maskseq = length_to_mask(length + 1, xseq.shape[0])  
        if self.pointmap_dim==0:
            output_latent = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)[1:] 
        else:
            pmask_pm = length_to_mask_pm(length,pm_mask,f_pm_) 
            f_pm_ = f_pm_.permute(2, 0,1, 3).contiguous()  
            output = self.seqTransDecoder(
                tgt=xseq,
                memory=f_pm_,
                tgt_key_padding_mask=~maskseq,
                memory_key_padding_mask=~pmask_pm,
            )  # [seqlen + 1, bs, d]
            output_latent = output[1:]  # [seqlen, bs, d]

        output = self.output_process(output_latent)  # [bs, v, d, seqlen]
        
        return MdmUnetOutput(sample=output, mask=maskseq[:, None, None, 1:])

    def freeze(self):
        if self.pointmap_dim==0:
            for layer in self.seqTransEncoder.layers:
                layer.freeze()
        else:
            for layer in self.seqTransDecoder.layers:
                layer.freeze()
        self.input_process.eval()
        self.input_process.requires_grad_(False)
        self.embed_timestep.eval()
        self.embed_timestep.requires_grad_(False)
        self.embed_text.eval()
        self.embed_text.requires_grad_(False)
        self.embed_RT.eval()
        self.embed_RT.requires_grad_(False)
        if self.freeze_k:
            self.embed_K.eval()
            self.embed_K.requires_grad_(False)
        self.output_process.eval()
        self.output_process.requires_grad_(False)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, d_model] -> [max_len, 1, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        if len(x.shape) == 4:
            # for mv data
            x = x + self.pe[: x.shape[0], None]
        else:
            raise NotImplementedError
        return self.dropout(x)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1)  # [bs, d, seqlen] -> [seqlen, bs, d]
        elif len(x.shape) == 4:
            x = x.permute(3, 0, 1, 2)  # [bs, v, d, seqlen] -> [seqlen, bs, v, d]
        x = self.poseEmbedding(x)  # [seqlen, bs, v, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        output = self.poseFinal(output)  # [seqlen, bs, v, d]
        output = output.permute(1, 2, 3, 0)  # [seqlen, bs, v, d] -> [bs, v, d, seqlen]
        return output


# Add to MainStore


cfg_mdmmv_offset = builds(
    PMDM2DMV,
    input_dim=46,
    latent_dim=512,
    cam_dim = 0,
    num_layers=8,
    num_heads=4,
    RT_dim=3,
    K_dim=4,
    pointmap_dim=224,
    is_bb2dpose=True,
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="mdmmv_offset", node=cfg_mdmmv_offset, group=f"network/mas")

cfg_mdmmv_offset_coco = builds(
    PMDM2DMV,
    input_dim=38,
    latent_dim=512,
    cam_dim = 0,
    num_layers=8,
    num_heads=4,
    RT_dim=3,
    K_dim=4,
    pointmap_dim=224,
    is_bb2dpose=True,
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="mdmmv_offset_coco", node=cfg_mdmmv_offset_coco, group=f"network/mas")

