import torch
from torch import nn
from torch.nn import functional as F
#from decoder import AttentionBlock, ResidualBlock


class Encoder(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        tf.keras.layers.Conv2D(3, 128, padding='valid'),
        ResidualBlock(128, 128),
        ResidualBlock(128, 128),
        tf.keras.layers.Conv2D(128, 128, padding='valid'),
        ResidualBlock(128, 256),
        ResidualBlock(256, 256),
        tf.keras.layers.Conv2D(256, 256, padding='valid'),
        ResidualBlock(256, 512),
        ResidualBlock(512, 512),
        tf.keras.layers.Conv2D(512, 512, padding='valid'),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512),
        AttentionBlock(512,512),
        ResidualBlock(512, 512),
        tf.keras.layers.GroupNormalization(32, 512),
        tf.keras.activations.silu(x),
        tf.keras.layers.Conv2D(512, 8, padding='valid'),
        tf.keras.layers.Conv2D(8, 8, padding='valid'),


    def forward(self, x, noise):
      x=tf.keras.random.normal((2,2))

      paddings=tf.constant(([[0,1],[0,1]]))
      tf.pad(x, paddings, mode='CONSTANT', constant_values=0)
      mean, log_variance = tf.split(x,num_or_size_splits=2,axis=1)
      log_variance = tf.clip_by_value(log_variance, -30, 20)
      variance = tf.exp(log_variance)
      stdev = tf.sqrt(variance)
      x = mean + stdev * noise
      x *= 0.18215
      return x
