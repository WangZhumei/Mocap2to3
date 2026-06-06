import torch
from pathlib import Path
from hmr4d.utils.pylogger import Log
from pytorch_lightning.utilities.memory import recursive_detach


def trusted_torch_load(*args, **kwargs):
    """Load trusted local project artifacts across PyTorch versions."""
    kwargs.setdefault("weights_only", False)
    return torch.load(*args, **kwargs)


def gaussian_smooth(x, sigma=3, dim=-1):
    kernel_smooth = _gaussian_kernel1d(sigma=sigma, order=0, radius=int(4 * sigma + 0.5))
    kernel_smooth = torch.from_numpy(kernel_smooth).float()[None, None].to(x)  # (1, 1, K)
    rad = kernel_smooth.size(-1) // 2

    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")  # (NB, 1, f)
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)  # (..., f)
    x = x.transpose(-1, dim)
    return x

def get_valid_mask(max_len, valid_len, device="cpu"):
    mask = torch.zeros(max_len, dtype=torch.bool).to(device)
    mask[:valid_len] = True
    return mask
def load_pretrained_model(model, ckpt_path, ckpt_type=None):
    """
    Load ckpt to model with strategy
    """
    assert ckpt_path

    # Option1: use model's own load_pretrained_model method
    if hasattr(model, "load_pretrained_model"):
        model.load_pretrained_model(ckpt_path, ckpt_type)
        return

    # Other options:
    Log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = trusted_torch_load(ckpt_path, "cpu")

    if ckpt_type is None:  # default loading to model
        model.load_state_dict(ckpt, strict=True)
    elif ckpt_type == "sahmr":
        model.load_pretrained_network(ckpt)


# @monitor_process_wrapper
# def get_resume_ckpts(cfg: DictConfig):
#     '''Get the latest checkpoints or return `None` if not exists.'''
#     pattern: str = '{}*.ckpt'.format(cfg['ckpt_path'])
#     ckpts = sorted(glob.glob(pattern))
#     if len(ckpts) > 0:
#         return ckpts[-1]
#     else:
#         return None


def find_last_ckpt_path(dirpath):
    """
    Assume ckpt is named as e{}* or last*, following the convention of pytorch-lightning.
    """
    dirpath = Path(dirpath)
    # Priority 1: last.ckpt
    auto_last_ckpt_path = dirpath / "last.ckpt"
    if auto_last_ckpt_path.exists():
        return auto_last_ckpt_path

    # Priority 2
    model_paths = []
    for p in sorted(list(dirpath.glob("*.ckpt"))):
        if "last" in p.name:
            continue
        model_paths.append(p)
    if len(model_paths) > 0:
        return model_paths[-1]
    else:
        Log.info("No checkpoint found, set model_path to None")
        return None
def repeat_to_max_len(x, max_len, dim=0):
    """Repeat last frame to max_len along dim"""
    assert isinstance(x, torch.Tensor)
    if x.shape[dim] == max_len:
        return x
    elif x.shape[dim] < max_len:
        x = x.clone()
        x = x.transpose(0, dim)
        x = torch.cat([x, repeat(x[-1:], "b ... -> (b r) ...", r=max_len - x.shape[0])])
        x = x.transpose(0, dim)
        return x
    else:
        raise ValueError(f"Unexpected length v.s. max_len: {x.shape[0]} v.s. {max_len}")


def repeat_to_max_len_dict(x_dict, max_len, dim=0):
    for k, v in x_dict.items():
        x_dict[k] = repeat_to_max_len(v, max_len, dim=dim)
    return x_dict



def select_state_dict_by_prefix(state_dict, prefix, new_prefix=""):
    """
    For each weight that start with {old_prefix}, remove the {old_prefic} and form a new state_dict.
    Args:
        state_dict: dict
        prefix: str
        new_prefix: str, if exists, the new key will be {new_prefix} + {old_key[len(prefix):]}
    Returns:
        state_dict_new: dict
    """
    state_dict_new = {}
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            new_key = new_prefix + k[len(prefix) :]
            state_dict_new[new_key] = state_dict[k]
    return state_dict_new


def detach_to_cpu(in_dict):
    return recursive_detach(in_dict, to_cpu=True)
