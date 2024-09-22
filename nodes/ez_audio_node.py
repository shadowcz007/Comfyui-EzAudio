import os,re
import sys,time
from pathlib import Path
import torchaudio
import hashlib
import torch
import folder_paths
import comfy.utils

import random
import numpy as np
import librosa
from accelerate import Accelerator
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DDIMScheduler


# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

# 添加当前插件的nodes路径，使ChatTTS可以被导入使用
sys.path.append(current_directory)


MAX_SEED = np.iinfo(np.int32).max

from EzAudio.src.models.conditioners import MaskDiT
from EzAudio.src.modules.autoencoder_wrapper import Autoencoder
from EzAudio.src.inference import inference
from EzAudio.src.utils import load_yaml_with_includes


def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)

ckpts_path=get_model_dir('ez_audio')

# Load model and configs
def load_models(config_name, ckpt_path, vae_path,t5_path, device):
    params = load_yaml_with_includes(config_name)

    # Load codec model
    autoencoder = Autoencoder(ckpt_path=vae_path,
                              model_type=params['autoencoder']['name'],
                              quantization_first=params['autoencoder']['q_first']).to(device)
    autoencoder.eval()

    # Load text encoder
    # print(t5_path)
    tokenizer = T5Tokenizer.from_pretrained(t5_path,local_files_only=True)
    text_encoder = T5EncoderModel.from_pretrained(t5_path).to(device)
    text_encoder.eval()

    # Load main U-Net model
    unet = MaskDiT(**params['model']).to(device)
    unet.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    unet.eval()

    accelerator = Accelerator(mixed_precision="fp16")
    unet = accelerator.prepare(unet)

    # Load noise scheduler
    noise_scheduler = DDIMScheduler(**params['diff'])

    latents = torch.randn((1, 128, 128), device=device)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
    _ = noise_scheduler.add_noise(latents, noise, timesteps)

    return autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params



# 需要了解python的class是什么意思
class EZLoadModelNode:
    def __init__(self):
        self.model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                       "device": (["auto","cpu"],),
                        }
                }
    
    RETURN_TYPES = ("EZMODEL",)
    RETURN_NAMES = ("ez_model",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Audio/EzAudio"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def run(self,device):
        # Model and config paths
        config_name = os.path.join(ckpts_path,'ezaudio-xl.yml')
        ckpt_path = os.path.join(ckpts_path,'s3','ezaudio_s3_xl.pt')
        vae_path =os.path.join(ckpts_path,'vae','1m.pt')
        t5_path = os.path.join(ckpts_path,'t5')
        # save_path = 'output/'
        # os.makedirs(save_path, exist_ok=True)
        if device=='auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.model==None:
            self.model = load_models(config_name, ckpt_path, vae_path,t5_path, device)
        return (self.model,)


class EZGenerateAudioNode:
    def __init__(self):
        self.model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                        "text":  ("STRING", 
                                     {
                                       "default":"", 
                                       "multiline": True,
                                       "dynamicPrompts": True # comfyui 动态提示
                                       }
                                    ),
                         "length": ("INT", {"default": 6, "min": 0, "max": 1000, "step": 1}), 
                         "guidance_scale": ("FLOAT", {"default": 3, "min": 0, "max": 10, "step": 0.1}), 
                         "guidance_rescale": ("FLOAT", {"default": 3, "min": 0, "max": 10, "step": 0.1}), 
                         "ddim_steps": ("INT", {"default": 6, "min": 0, "max": 1000, "step": 1}), 
                          "eta": ("FLOAT", {"default": 3, "min": 0, "max": 10, "step": 0.1}), 
                          "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "step": 1}), 
                          "ez_model": ("EZMODEL",),
                          "device": (["auto","cpu"],),
                        }
                }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Audio/EzAudio"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def run(self,text, length,
                   guidance_scale, guidance_rescale, ddim_steps, eta,
                   seed,
                   ez_model,
                   device):
        
        autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params=ez_model

        neg_text = None
        length = length * params['autoencoder']['latent_sr']

        gt, gt_mask = None, None

        if text == '':
            guidance_scale = None
            print('empyt input')

        if device=='auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
        pred = inference(autoencoder, unet,
                     gt, gt_mask,
                     tokenizer, text_encoder,
                     params, noise_scheduler,
                     text, neg_text,
                     length,
                     guidance_scale, guidance_rescale,
                     ddim_steps, eta, seed,
                     device)
        
        print(pred.shape)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)

        # {'waveform': tensor([], size=(1, 1, 0)), 'sample_rate': 44100}

        return ({'waveform': pred, 'sample_rate': params['autoencoder']['sr']},)







