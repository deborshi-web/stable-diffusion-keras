import torch
import numpy as np
import os


def get_time_embedding(timestep):
    freqs = tf.pow(10000,tf.range(start=0, limit=160) / 160)
    x = tf.constant([timestep]* freqs)
    return tf.concat([tf.cos(x), tf.sin(x)],axis=-1,name='cos')

def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=np.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas_cumprod

def get_file_path(filename, url=None):
    module_location = os.path.dirname(os.path.abspath(__file__))
    parent_location = os.path.dirname(module_location)
    file_location = os.path.join(parent_location, "data", filename)
    return file_location

def move_channel(image, to):
  if to == "first":
      return tf.transpose(images,perm=(0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
  elif to == "last":
      return tf.transpose(images,perm=(0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
  else:
      raise ValueError("to must be one of the following: first, last")

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x
