"""
Text-Noise-Image triplet sampler
"""
import os
import json
import time

import torch
import numpy as np

from pipeline_sd_tni import StableDiffusionPipelineTni
from utils import log_info, get_model_id_by_name, get_time_ttl_and_eta


class TextNoiseImageTripletSample:
    def __init__(self, args):
        self.args = args

    def read_annotations(self):
        """
        Each annotation is like:
            {
                "image_id": 179765,
                "id": 38,
                "caption": "A black Honda motorcycle parked in front of a garage."
            },
        :return:
        """
        args = self.args
        j_file = args.anno_file
        if not os.path.exists(j_file):
            raise ValueError(f"File not exist: {j_file}")
        log_info(f"Reading JSON file: {j_file}...")
        with open(j_file, 'r') as fptr:
            j_str = fptr.read()
        log_info(f"Parsing JSON file: {j_file}...")
        j_dict = json.loads(j_str)
        anno_arr = j_dict.get("annotations", None)
        image_arr = j_dict.get("images", [])
        if anno_arr is None:
            raise ValueError(f"JSON file has no 'annotations' item: {j_file}")
        log_info(f"  annotations    : {len(anno_arr)}")
        log_info(f"  images         : {len(image_arr)}")
        log_info(f"Summarize JSON   : {j_file}...Done")
        return anno_arr

    def run(self):
        log_info(f"TextNoiseImageTripletSample::run()...")
        anno_arr = self.read_annotations()
        args = self.args
        sample_count = args.sample_count
        if sample_count <= 0:
            sample_count = len(anno_arr)
        b_sz = args.sample_batch_size
        b_cnt = sample_count // b_sz
        if b_cnt * b_sz < sample_count:
            b_cnt += 1
        prompt_cnt = len(anno_arr)
        log_info(f"args.sample_count: {args.sample_count}")
        log_info(f"sample_count     : {sample_count}")
        log_info(f"b_sz             : {b_sz}")
        log_info(f"b_cnt            : {b_cnt}")
        log_info(f"prompt_cnt       : {prompt_cnt}")

        model_id = get_model_id_by_name(args.model)
        if model_id is None:
            raise ValueError(f"Not found model_id by name: {args.model}")
        steps = args.sample_steps_arr[0]
        guidance_scale = 6
        log_info(f"args.model       : {args.model}")
        log_info(f"model_id         : {model_id}")
        log_info(f"steps            : {steps}")
        log_info(f"guidance_scale   : {guidance_scale}")
        log_info(f"pipe = StableDiffusionPipelineTni.from_pretrained()...")
        pipe = StableDiffusionPipelineTni.from_pretrained(model_id, torch_dtype=torch.float16, resume_download=None)
        pipe.safety_checker = None
        pipe.to(args.device)
        st_time = time.time()
        log_itv = args.log_interval
        for b_idx in range(b_cnt):
            i = b_idx * b_sz % prompt_cnt
            j = i + b_sz
            if j <= prompt_cnt:
                anno_batch = anno_arr[i:j]
            else:
                j = j % prompt_cnt
                anno_batch = anno_arr[i:] + anno_arr[:j]
            log_info(f"B{b_idx:03d}/{b_cnt}:[{i:05d}-{j:05d}]/{prompt_cnt}")
            prompt_batch = [a["caption"] for a in anno_batch]
            pipe_output = pipe(prompt=prompt_batch,
                               num_inference_steps=steps,
                               guidance_scale=guidance_scale,
                               b_idx=b_idx)
            self.save_images(pipe_output["images"], b_idx, b_sz)
            self.save_annotations(anno_batch, b_idx, b_sz)
            self.save_numpy(pipe_output["prompt_pos_emb"], b_idx, b_sz, "prompt_pos_emb")
            self.save_numpy(pipe_output["prompt_neg_emb"], b_idx, b_sz, "prompt_neg_emb")
            self.save_numpy(pipe_output["gaussian_noise"], b_idx, b_sz, "gaussian_noise")
            self.save_numpy(pipe_output["generate_latent"], b_idx, b_sz, "generate_latent")
            if log_itv > 0 and (b_idx % log_itv == 0 or b_idx + 1 == b_cnt):
                elp, eta = get_time_ttl_and_eta(st_time, b_idx+1, b_cnt)
                log_info(f"B{b_idx:03d}/{b_cnt}:elp:{elp}, eta:{eta}")

    def save_images(self, images, b_idx, b_sz):
        img_cnt = len(images)
        img_dir = os.path.join(self.args.sample_output_dir, "image")
        if not os.path.exists(img_dir):
            log_info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        img_path = None
        for i in range(img_cnt):
            img_id = b_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            images[i].save(img_path)
        log_info(f"Saved: {img_path}")

    def save_annotations(self, anno_arr, b_idx, b_sz):
        cnt = len(anno_arr)
        save_dir = os.path.join(self.args.sample_output_dir, "annotations")
        if not os.path.exists(save_dir):
            log_info(f"os.makedirs({save_dir})")
            os.makedirs(save_dir)
        f_path = None
        for i in range(cnt):
            item_id = b_idx * b_sz + i
            f_path = os.path.join(save_dir, f"{item_id:05d}.txt")
            json_str = json.dumps(anno_arr[i], indent=4)
            with open(f_path, 'w') as fptr:
                fptr.write(json_str)
        # for
        log_info(f"Saved: {f_path}")

    def save_numpy(self, tensor, b_idx, b_sz, dir_name):
        cnt = len(tensor)
        save_dir = os.path.join(self.args.sample_output_dir, dir_name)
        if not os.path.exists(save_dir):
            log_info(f"os.makedirs({save_dir})")
            os.makedirs(save_dir)
        f_path = None
        for i in range(cnt):
            item_id = b_idx * b_sz + i
            f_path = os.path.join(save_dir, f"{item_id:05d}.npy")
            np.save(f_path, tensor[i].numpy())
        # for
        log_info(f"Saved: {f_path}")

# class
