import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from tni_triplet_sampler import TextNoiseImageTripletSample
from utils import log_info, get_model_id_by_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", type=str, default='tni_gen')
    parser.add_argument("--model", type=str, default='sd15')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument("--seed", type=int, default=123, help="Random seed. 0 means ignore")
    parser.add_argument("--anno_file", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument("--sample_count", type=int, default=10, help="0 means using annotation count")
    parser.add_argument("--sample_batch_size", type=int, default=5)
    parser.add_argument("--sample_output_dir", type=str, default="./output6_test_tni")
    parser.add_argument("--sample_steps_arr", nargs='*', type=int, default=[25])

    args = parser.parse_args()
    # add device
    gpu_ids = args.gpu_ids
    device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    args.device = device
    log_info(f"gpu_ids : {gpu_ids}")
    log_info(f"device  : {device}")

    # set random seed
    seed = args.seed  # if seed is 0. then ignore it.
    log_info(f"args.seed : {seed}")
    if seed:
        log_info(f"  torch.manual_seed({seed})")
        log_info(f"  np.random.seed({seed})")
        torch.manual_seed(seed)
        np.random.seed(seed)
    if seed and torch.cuda.is_available():
        log_info(f"  torch.cuda.manual_seed_all({seed})")
        torch.cuda.manual_seed_all(seed)
    log_info(f"final seed: torch.initial_seed(): {torch.initial_seed()}")

    cudnn.benchmark = True
    return args

def sample(args):
    if args.model == 'sd15':
        negative_prompt = "painting, unreal, twisted"
        steps           = 25
        guidance_scale  = 6
    elif args.model == '2rf_sd15':
        negative_prompt = "painting, unreal, twisted",
        steps           = 25,
        guidance_scale  = 1.5,
    elif args.model == 'instaflow_09b':
        negative_prompt = None
        steps           = 1
        guidance_scale  = 0.0
    else:
        raise ValueError(f"Invalid model: {args.model}")

    model_id = get_model_id_by_name(args.model)
    prompt = "A hyper-realistic photo of a cute cat."

    from pipeline_rf import RectifiedFlowPipeline
    from diffusers import StableDiffusionPipeline

    if "XCLIU" in model_id:
        # set resume_download=None, to avoid warning message:
        #   /.../lib/python3.8/site-packages/huggingface_hub/file_download.py:1132:
        #   FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0.
        #   Downloads always resume when possible.
        #   If you want to force a new download, use `force_download=True`.
        pipe = RectifiedFlowPipeline.from_pretrained(model_id, torch_dtype=torch.float16, resume_download=None)
        log_info(f"pipe = RectifiedFlowPipeline.from_pretrained()")
        # switch to torch.float32 for higher quality
    else:
        # assume it is pure SD
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, resume_download=None)
        log_info(f"pipe = StableDiffusionPipeline.from_pretrained()")

    pipe.safety_checker = None
    pipe.to("cuda")

    # 2-rectified flow is a multi-step text-to-image generative model.
    # It can generate with extremely few steps, e.g, 2-8
    # For guidance scale, the optimal range is [1.0, 2.0], which is smaller than normal Stable Diffusion.
    # You may set negative_prompts like normal Stable Diffusion
    pipe_output = pipe(prompt=prompt,
                       negative_prompt=negative_prompt,
                       num_inference_steps=steps,
                       guidance_scale=guidance_scale)
    images = pipe_output.images
    b_name = os.path.basename(model_id)
    f_path = f"./image_{b_name}_step{steps}.png"
    images[0].save(f_path)
    log_info(f"model_id        : {model_id}")
    log_info(f"prompt          : {prompt}")
    log_info(f"negative_prompt : {negative_prompt}")
    log_info(f"steps           : {steps}")
    log_info(f"guidance_scale  : {guidance_scale}")
    log_info(f"Image count     : {len(images)}")
    log_info(f"Image saved     : {f_path}")

def main():
    args = parse_args()
    if args.todo == 'sample':
        sample(args)
    elif args.todo == 'tni_gen':
        runner = TextNoiseImageTripletSample(args)
        runner.run()
    else:
        raise ValueError(f"Invalid todo: {args.todo}")

if __name__ == "__main__":
    main()
