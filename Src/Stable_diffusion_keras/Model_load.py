import torch
import warnings


def make_compatible(state_dict):
    keys = list(state_dict.keys())
    changed = False
    for key in keys:
        if "causal_attention_mask" in key:
            del state_dict[key]
            changed = True
        elif "_proj_weight" in key:
            new_key = key.replace('_proj_weight', '_proj.weight')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True
        elif "_proj_bias" in key:
            new_key = key.replace('_proj_bias', '_proj.bias')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True

    if changed:
        warnings.warn(("Given checkpoint data were modified dynamically by make_compatible"
                       " function on model_loader.py. Maybe this happened because you're"
                       " running newer codes with older checkpoint files. This behavior"
                       " (modify old checkpoints and notify rather than throw an error)"
                       " will be removed soon, so please download latest checkpoints file."))

    return state_dict

def kcv_diffusion():
    state_dict = h5py.File('/root/.keras/datasets/kcv_diffusion_model.h5')
    state_dict = make_compatible(state_dict)
    return state_dict.keys()

def diffusion_model_v2_1():
    state_dict = h5py.File('/root/.keras/datasets/diffusion_model_v2_1.h5')
    state_dict = make_compatible(state_dict)
    return state_dict.keys()


def kcv_decoder():
    state_dict = h5py.File('/root/.keras/datasets/kcv_decoder.h5')
    state_dict = make_compatible(state_dict)
    return state_dict.keys()

def text_encoder():
    state_dict = h5py.File('/root/.keras/datasets/kcv_encoder.h5')
    state_dict = make_compatible(state_dict)
    return state_dict.keys()

def vae_encoder():
    state_dict = h5py.File('/root/.keras/datasets/vae_encoder.h5')
    state_dict = make_compatible(state_dict)
    return state_dict.keys()




def preload_model():
    return {
        'kcv_diffusion': kcv_diffusion(),
        'diffusion_model_v2_1': diffusion_model_v2_1(),
        'kcv_decoder': kcv_decoder(),
        'text_encoder': text_encoder(),
    }
