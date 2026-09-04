import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import stable_diffusion_pytorch.tokenizer
from stable_diffusion_pytorch import KLMSSampler
from stable_diffusion_pytorch import KEulerSampler
from stable_diffusion_pytorch import KEulerAncestralSampler
import stable_diffusion_pytorch.util
from stable_diffusion_pytorch import model_loader
from stable_diffusion_pytorch.diffusion import Diffusion

def generate(
        prompts,
        uncond_prompts,
        input_images=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        height=512,
        width=512,
        sampler="k_lms",
        n_inference_steps=50,
        models={},
        seed=None,
        device=None,
        idle_device=None
):

        generator = tf.random.Generator

        if seed is None:
            generator.from_seed(seed)

        tokenizer = Tokenizer()
        clip = CLIP().embedding.token_embedding


        # use the dtype of the model weights as our dtype
        dtype = CLIP().embedding
        if do_cfg:
            cond_tokens = Tokenizer().encode_batch(prompts)
            cond_tokens = tf.constant(cond_tokens)
            cond_context = CLIP().embedding.token_embedding(cond_tokens)
            uncond_tokens = Tokenizer().encode_batch(uncond_prompts)
            uncond_tokens = tf.constant(uncond_tokens)
            uncond_context = CLIP().embedding.token_embedding(uncond_tokens)
            context = tf.keras.layers.concatenate([cond_context, uncond_context])
        else:
            tokens = Tokenizer().encode_batch(prompts)
            tokens = tf.constant(tokens)
            context = CLIP().embedding.token_embedding(tokens)

        del tokenizer, clip

        if sampler == "k_lms":
            sampler = KLMSSampler(n_inference_steps=n_inference_steps)
        elif sampler == "k_euler":
            sampler = KEulerSampler(n_inference_steps=n_inference_steps)
        elif sampler == "k_euler_ancestral":
            sampler = KEulerAncestralSampler(n_inference_steps=n_inference_steps,
                                             generator=generator)
        else:
            raise ValueError(
                "Unknown sampler value %s. "
                "Accepted values are {k_lms, k_euler, k_euler_ancestral}"
                % sampler
            )

        noise_shape = (len(prompts), 4, height // 8, width // 8)

        if input_images:
            encoder = Encoder().forward(tf.keras.random.normal((2,2)),tf.keras.random.normal((2,2)))
            processed_input_images = []
            for input_image in input_images:
                if type(input_image) is str:
                    input_image = Image.open(input_image)

                input_image = input_image.resize((width, height))
                input_image = np.array(input_image)
                input_image = tf.constant(input_image, dtype=dtype)
                input_image = util.rescale(input_image, (0, 255), (-1, 1))
                processed_input_images.append(input_image)
            input_images_tensor = tf.stack(processed_input_images)
            input_images_tensor = util.move_channel(input_images_tensor, to="first")

            _, _, height, width = input_images_tensor.shape
            noise_shape = (len(prompts), 4, height // 8, width // 8)

            encoder_noise = tf.keras.random.normal(noise_shape, generator=generator, device=device, dtype=dtype)
            latents = encoder(input_images_tensor, encoder_noise)

            latents_noise = tf.keras.random.normal(noise_shape, generator=generator, device=device, dtype=dtype)
            sampler.set_strength(strength=strength)
            latents += latents_noise * sampler.initial_scale

            del encoder, processed_input_images, input_images_tensor, latents_noise
        else:
            latents = tf.keras.random.normal(noise_shape)
            latents *= sampler.initial_scale

        diffusion = diffusion = Diffusion()

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = util.get_time_embedding(timestep)

            input_latents = latents * sampler.get_input_scale()
            if do_cfg:
                input_latents = tf.repeat(input_latents,5)

        del diffusion

        def move_channel(image, to):
          if to == "first":
              return tf.transpose(images,perm=(0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
          elif to == "last":
              return tf.transpose(images,perm=(0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
          else:
              raise ValueError("to must be one of the following: first, last")


        decoder = decoder = Decoder().forward(tf.keras.random.normal((32,32,32,32)))
        images = Decoder().forward(latents)
        del decoder

        zeros=np.zeros((1,1))
        images = move_channel(images, to="last")
        image=tf.uint8.from_tensors(images)
        Image.fromarray(zeros)
