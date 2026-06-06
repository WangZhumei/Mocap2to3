import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmr4d.network.base_arch.transformer_layer.sd_layer import zero_module
from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.network.resnet.resnet import ResNet
from hmr4d.configs import MainStore, builds


def length_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def length_to_mask_pm(length,pm_mask,f_pm_):
    if pm_mask is None:
        mask = torch.full((length.shape[0], f_pm_.shape[1]), True, device=length.device)
        return mask
    mask = pm_mask>0
    return mask


class PMDM2D(nn.Module):
    def __init__(
        self,
        # x
        input_dim=44,
        # condition
        text_dim=512,
        RT_dim=6,
        K_dim=300*4,
        pointmap_dim = 0,
        # intermediate
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        # training
        dropout=0.1,
        **kargs,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.text_dim = text_dim
        self.RT_dim = RT_dim
        self.K_dim = K_dim
        self.pointmap_dim = pointmap_dim
        #self.img_w = img_w

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dropout = dropout

        self._build_model()

    def _build_model(self):
        self._build_input()
        self._build_condition()
        self._build_transformer()
        self._build_output()

    def _build_input(self):
        self.input_process = InputProcess(self.input_dim, self.latent_dim)

    def _build_condition(self):
        # TODO: upgrade residual module
        self.embed_text = nn.Linear(self.text_dim, self.latent_dim)
        self.embed_RT = zero_module(nn.Linear(self.RT_dim, self.latent_dim))
        if self.K_dim>0:
            self.embed_K = zero_module(nn.Linear(self.K_dim, self.latent_dim))
        if self.pointmap_dim>0: #temp
            self.encoder_PM = ResNet([2, 2, 2, 2])

    def _build_transformer(self):
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.pointmap_dim==0:
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation="gelu",
            )
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        else:
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation="gelu",
            )
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)
    def _build_output(self):
        self.output_process = OutputProcess(self.input_dim, self.latent_dim)

    def forward(self, x, timesteps, length, f_text=None, f_RT=None,f_K=None,f_pm=None,pm_mask=None):
        """
        Args:
            x: (B, C, L), a noisy motion sequence
            timesteps: (B,)
            length: (B), valid length of x
            f_text: (B, C)
        """
        B, _, L = x.shape
        # Set timesteps
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        assert len(timesteps) == x.shape[0]
        emb = self.embed_timestep(timesteps) 
        if f_text is not None:
            f_ = self.embed_text(f_text)
            emb = emb + f_[None] 
        if f_RT is not None:
            f_ = self.embed_RT(f_RT) 
            emb = emb + f_[None]  
        if f_K is not None:
            f_ = self.embed_K(f_K)  
            emb = emb + f_[None]  

        if f_pm is not None:
            f_pm_ = f_pm.permute(0, 3,1, 2)  
            f_pm_ = self.encoder_PM(f_pm_)  
            f_pm_ = f_pm_.permute(0, 2,3, 1).reshape(B,-1,self.latent_dim)  
        x = self.input_process(x)  

        # adding the timestep embed
        xseq = torch.cat((emb, x), dim=0)  
        xseq = self.sequence_pos_encoder(xseq)  

        maskseq = length_to_mask(length + 1, xseq.shape[0])  
        if self.pointmap_dim==0:
            output = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)[1:] 
        else:
            pmask_pm = length_to_mask_pm(length,pm_mask,f_pm_) 
            f_pm_ = f_pm_.permute(1, 0, 2)  
            output = self.seqTransDecoder(
                tgt=xseq,
                memory=f_pm_,
                tgt_key_padding_mask=~maskseq,
                memory_key_padding_mask=~pmask_pm,
            ) 
            output = output[1:]  

        output = self.output_process(output) 
        return MdmUnetOutput(sample=output, mask=maskseq[:, None, 1:])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # [bs, d, seqlen] -> [seqlen, bs, d]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.permute(1, 2, 0)  # [seqlen, bs, d] -> [bs, d, seqlen]
        return output


# Add to MainStore


cfg_mdm_offset = builds(
    PMDM2D,
    input_dim=46,
    latent_dim=512,
    num_layers=8,
    num_heads=4,
    RT_dim=3,
    K_dim=4,
    pointmap_dim=224,
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="mdm_offset", node=cfg_mdm_offset, group=f"network/mas")


cfg_mdm_offset_coco = builds(
    PMDM2D,
    input_dim=38,
    latent_dim=512,
    num_layers=8,
    num_heads=4,
    RT_dim=3,
    K_dim=4,
    pointmap_dim=224,
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="mdm_offset_coco", node=cfg_mdm_offset_coco, group=f"network/mas")
