import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import mmap
import json

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from flask import Flask, request

import requests
import io

import base64

app = Flask(__name__)

# load safety model
#safety_model_id = "CompVis/stable-diffusion-safety-checker"
#safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
#safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def load_file(filename, device):
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)

    size = os.stat(filename).st_size
    storage = torch.ByteStorage.from_file(filename, shared=False, size=size).untyped()
    offset = n + 8
    return {name: create_tensor(storage, info, offset) for name, info in metadata.items() if name != "__metadata__"}




def create_tensor(storage, info, offset):
    DTYPES = {"F32": torch.float32, "F16": torch.float16, "I64": torch.int64}
    dtype = DTYPES[info["dtype"]]
    shape = info["shape"]
    start, stop = info["data_offsets"]
    return torch.asarray(storage[start + offset : stop + offset], dtype=torch.uint8).view(dtype=dtype).reshape(shape)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    if ckpt.endswith('.safetensors'):
        sd = load_file(ckpt, device="cpu")
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

class OptionModel:
    prompt: str = ""
    ddim_steps: int = 50
    plms: bool = False
    dpm_solver: bool = False
    laion400m: bool = False
    fixed_code: bool = False
    ddim_eta: float = 0.0
    config: str = "configs/stable-diffusion/v1-inference.yaml"
    ckpt: str = "models/ldm/stable-diffusion-v1/model.safetensors"
    n_iter: int = 1
    H: int = 512
    W: int = 512
    C: int = 4
    f: int = 8
    n_samples: int = 1
    n_rows: int = 0
    scale: float = 7.5
    model: str = "model"
    seed: int = 42
    precision: str = "autocast" #"full"

opt=OptionModel()
model=""
start_code=""
precision_scope=""
data=""
sampler=""
wm_encoder=""

def init(request: OptionModel):
    global model
    global start_code
    global precision_scope
    global opt
    global data
    global sampler
    global wm_encoder
    global opt

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "WebtoonToday"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]


    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext


@app.route("/", methods=['POST'])
def txt2img():
    global model
    global start_code
    global precision_scope
    global opt
    global data
    global sampler
    global wm_encoder
    global opt

    req = request.get_json()

    filename = req['filename']
    
    if 'props' in req:
        if 'width' in req['props']:
            opt.W = req['props']['width'] 
        if 'height' in req['props']:
            opt.H = req['props']['height']
        if 'steps' in req['props']:
            opt.ddim_steps = req['props']['steps']
    if 'seed' in req:
        opt.seed = req['seed']
    
    data = [1 * [req['prompt']]]

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()

                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(1 * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    #x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    x_checked_image = x_samples_ddim

                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    x_sample =x_checked_image_torch[0]
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    buffer.seek(0)
                    path = _upload_image_to_cdn(buffer, filename)
                    
                    return {
                        'code': 200,
                        'image': path.replace('https://s3.ap-northeast-2.amazonaws.com/', 'https://')
                    }

# Upload an image to cdn and returns the corresponding url if succeeded
def _upload_image_to_cdn(image_content, filename: str):
    cdn_url = None

    response = requests.put(
        "https://api.webtoon.today/image",
        params={"filename": 'image-generation/'+filename+".png", "fullPath": True},
        timeout=10,
    )

    if not response.ok:
        raise ValueError(response.text)

    signed_url = None
    try:
        signed_url = response.json()["data"]["uploadURL"]
    except KeyError as e:
        raise e

    response = requests.put(
        signed_url, data=image_content
    )

    if not response.ok:
        raise ValueError(response.text)

    cdn_url = signed_url.split('?')[0]

    return cdn_url


init(OptionModel())

if __name__ == "__main__":

    app.run(host='0.0.0.0', port=8000)

