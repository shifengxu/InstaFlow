"""
Text-Noise-Image triplet sampler
"""
import os
import time
import torch
from torch import optim
import torch.utils.data as tu_data

from pipeline_rf import RectifiedFlowPipeline
from tni_triplet_dataset import TextNoiseImageDataset
from utils import log_info, get_model_id_by_name, get_time_ttl_and_eta


class TextNoiseImageTripletTrain:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.pipe = None
        self.unet = None
        self.optimizer = None
        self.start_time = None  # the time when training starts
        self.batch_counter = 0  # batch counter
        self.batch_total = 0    # total batch in all epochs
        self.guidance_scale = 1.5

    def get_data_loader(self, train_shuffle=False):
        args = self.args
        batch_size = args.batch_size
        data_limit = args.data_limit
        data_dir   = args.data_dir
        num_workers = 4
        train_ds = TextNoiseImageDataset(data_dir, data_limit=data_limit)
        train_loader = tu_data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
        )
        log_info(f"train dataset and data loader:")
        log_info(f"  root       : {train_ds.data_dir}")
        log_info(f"  len        : {len(train_ds)}")
        log_info(f"  batch_cnt  : {len(train_loader)}")
        log_info(f"  batch_size : {batch_size}")
        log_info(f"  shuffle    : {train_shuffle}")
        log_info(f"  num_workers: {num_workers}")
        return train_loader

    def set_model(self):
        args = self.args
        log_info(f"TextNoiseImageTripletTrain::set_model()...")
        model_id = get_model_id_by_name(args.model)
        if model_id is None:
            raise ValueError(f"Not found model_id by name: {args.model}")
        steps = args.sample_steps_arr[0]
        log_info(f"  args.model     : {args.model}")
        log_info(f"  model_id       : {model_id}")
        log_info(f"  steps          : {steps}")
        log_info(f"  pipe = StableDiffusionPipelineTni.from_pretrained()...")
        pipe = RectifiedFlowPipeline.from_pretrained(model_id, torch_dtype=torch.float16, resume_download=None)
        pipe.safety_checker = None
        pipe.to(args.device)
        pipe.vae.requires_grad = False
        self.pipe = pipe
        self.unet = pipe.unet

        lr = args.lr
        params = self.unet.parameters()
        # beta1, eps, w_decay = 0.9, 0.00000001, 0.0
        # optimizer = optim.Adam(params, lr=lr, betas=(beta1, 0.999), eps=eps, weight_decay=w_decay)
        optimizer = optim.SGD(params, lr=lr, momentum=0.9)
        log_info(f"  optimizer ---------")
        log_info(f"  name     : {type(optimizer).__name__}")
        log_info(f"  lr       : {lr}")
        self.optimizer = optimizer
        log_info(f"TextNoiseImageTripletTrain::set_model()...Done")

    def train(self):
        log_info(f"TextNoiseImageTripletTrain::train()...")
        args = self.args
        data_loader = self.get_data_loader()
        self.set_model()
        log_int  = args.log_interval
        save_int = args.save_ckpt_interval
        sample_cnt = args.sample_count
        e_cnt = args.n_epochs       # epoch count
        b_cnt = len(data_loader)    # batch count
        self.batch_total = e_cnt * b_cnt
        self.start_time = time.time()
        lr = args.lr
        log_info(f"TextNoiseImageTripletTrain::train()")
        log_info(f"  log_interval  : {log_int}")
        log_info(f"  save_interval : {save_int}")
        log_info(f"  sample_cnt    : {sample_cnt}")
        log_info(f"  sample_steps  : {args.sample_steps_arr}")
        log_info(f"  b_sz          : {args.batch_size}")
        log_info(f"  lr            : {lr}")
        log_info(f"  b_cnt         : {b_cnt}")
        log_info(f"  e_cnt         : {e_cnt}")
        log_info(f"  batch_total   : {self.batch_total}")
        log_info(f"  loss_dual     : {args.loss_dual}")
        log_info(f"  loss_lambda   : {args.loss_lambda}")
        self.unet.train()
        for epoch in range(1, e_cnt+1):
            log_info(f"Epoch {epoch}/{e_cnt} ---------- lr={lr:.7f}")
            loss_sum, loss_cnt = 0., 0
            for i, (gs_noise, gen_latent, p_pos_emb, p_neg_emb) in enumerate(data_loader):
                # gaussian_noise, generate_latent, prompt_pos_emb, prompt_neg_emb
                self.batch_counter += 1
                gs_noise = gs_noise.to(args.device).to(torch.float16)
                gen_latent = gen_latent.to(args.device).to(torch.float16)
                p_pos_emb = p_pos_emb.to(args.device).to(torch.float16)
                p_neg_emb = p_neg_emb.to(args.device).to(torch.float16)

                with torch.autocast("cuda"):
                    loss, decay = self.train_batch(gs_noise, gen_latent, p_pos_emb, p_neg_emb)
                loss_sum += loss
                loss_cnt += 1
                if log_int > 0 and ((i+1) % log_int == 0 or i+1 == b_cnt):
                    log_info(f"E{epoch}.B{i:03d}/{b_cnt} loss:{loss:6.4f}")
                if i < sample_cnt and epoch % 10 == 0:
                    self.unet.eval()
                    self.sample_online(epoch, i, args.batch_size, gs_noise, p_pos_emb, p_neg_emb)
                    self.unet.train()
            # for
            loss_avg = loss_sum / loss_cnt
            elp, eta = get_time_ttl_and_eta(self.start_time, self.batch_counter, self.batch_total)
            log_info(f"E{epoch}.training_loss_avg: {loss_avg:.6f}. elp:{elp}, eta:{eta}")
            if 0 < epoch < e_cnt and save_int > 0 and epoch % save_int == 0:
                self.save_ckpt(epoch, True)
        # for
        self.save_ckpt(e_cnt, False)

    def train_batch(self, gaussian_noise, generate_latent, prompt_pos_emb, prompt_neg_emb):
        prompt_embeds = torch.cat([prompt_neg_emb, prompt_pos_emb])

        self.optimizer.zero_grad()

        if self.args.loss_dual:
            loss, loss_adj = self.calc_loss_dual(gaussian_noise, generate_latent, prompt_embeds)
            loss_sum = loss + loss_adj * self.args.loss_lambda
        else:
            loss_sum = loss = self.calc_loss(gaussian_noise, generate_latent, prompt_embeds)
            loss_adj = torch.tensor(0.)

        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()
        # todo apply EMA
        return loss.item(), loss_adj.item()

    def calc_loss_dual(self, gaussian_noise, generate_latent, prompt_embeds):
        b_sz, ch, h, w = gaussian_noise.size()
        t1 = torch.randint(low=1, high=1001, size=(b_sz,), device=self.device)
        t1_ratio = torch.div(t1, 1000.).to(torch.float16)
        t1_ratio = t1_ratio.view(-1, 1, 1, 1)

        latent_model_input1 = t1_ratio * gaussian_noise + (1. - t1_ratio) * generate_latent
        latent_model_input1 = torch.cat([latent_model_input1] * 2)
        t1 = torch.cat([t1] * 2)

        unet_output = self.unet(latent_model_input1, t1, encoder_hidden_states=prompt_embeds)
        grad_pred1 = unet_output.sample  # predicted gradient
        v_pred_neg, v_pred_text = grad_pred1.chunk(2)
        grad_pred1 = v_pred_neg + self.guidance_scale * (v_pred_text - v_pred_neg)

        t2 = torch.randint(low=1, high=1001, size=(b_sz,), device=self.device)
        t2_ratio = torch.div(t2, 1000.).to(torch.float16)
        t2_ratio = t2_ratio.view(-1, 1, 1, 1)

        latent_model_input2 = t2_ratio * gaussian_noise + (1. - t2_ratio) * generate_latent
        latent_model_input2 = torch.cat([latent_model_input2] * 2)
        t2 = torch.cat([t2] * 2)

        unet_output2 = self.unet(latent_model_input2, t2, encoder_hidden_states=prompt_embeds)
        grad_pred2 = unet_output2.sample  # predicted gradient
        v_pred_neg, v_pred_text = grad_pred2.chunk(2)
        grad_pred2 = v_pred_neg + self.guidance_scale * (v_pred_text - v_pred_neg)

        grad_real = generate_latent - gaussian_noise
        loss1 = (grad_pred1 - grad_real).square().mean()
        loss2 = (grad_pred2 - grad_real).square().mean()
        loss = (loss1 + loss2) / 2.
        loss_adj = (grad_pred1 - grad_pred2).square().mean()
        return loss, loss_adj

    def calc_loss(self, gaussian_noise, generate_latent, prompt_embeds):
        b_sz, ch, h, w = gaussian_noise.size()
        t = torch.randint(low=1, high=1001, size=(b_sz,), device=self.device)
        t_ratio = torch.div(t, 1000.).to(torch.float16)
        t_ratio = t_ratio.view(-1, 1, 1, 1)

        latent_model_input = (1. - t_ratio) * gaussian_noise + t_ratio * generate_latent
        latent_model_input = torch.cat([latent_model_input] * 2)
        t = torch.cat([t] * 2)

        unet_output = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)
        pred_output = unet_output.sample  # predicted gradient
        v_pred_neg, v_pred_text = pred_output.chunk(2)
        grad_pred = v_pred_neg + self.guidance_scale * (v_pred_text - v_pred_neg)

        grad_real = generate_latent - gaussian_noise
        loss = (grad_pred - grad_real).square().mean()
        return loss

    def save_ckpt(self, epoch, epoch_in_file_name=False):
        ckpt_path = self.args.save_ckpt_path
        save_ckpt_dir, base_name = os.path.split(ckpt_path)
        if not os.path.exists(save_ckpt_dir):
            log_info(f"os.makedirs({save_ckpt_dir})")
            os.makedirs(save_ckpt_dir)
        if epoch_in_file_name:
            stem, ext = os.path.splitext(base_name)
            ckpt_path = os.path.join(save_ckpt_dir, f"{stem}_E{epoch:03d}{ext}")
        log_info(f"Save ckpt: {ckpt_path} . . .")
        torch.save(self.unet.state_dict(), ckpt_path)
        log_info(f"Save ckpt: {ckpt_path} . . . Done")

    def sample_online(self, epoch, b_idx, b_sz, latents, prompt_pos_emb, prompt_neg_emb):
        steps = self.args.sample_steps_arr[0]
        pipe_output = self.pipe(prompt=None,
                                latents=latents,
                                prompt_embeds=prompt_pos_emb,
                                negative_prompt_embeds=prompt_neg_emb,
                                num_inference_steps=steps,
                                guidance_scale=self.guidance_scale)
        images = pipe_output.images
        img_cnt = len(images)
        root_dir = self.args.sample_output_dir
        sub_dir = os.path.join(root_dir, f"sample_online_steps{steps}")
        f_path = None
        for i in range(len(images)):
            img_id = b_idx * b_sz + i
            img_dir = os.path.join(sub_dir, f"img{img_id:05d}")
            if not os.path.exists(img_dir):
                log_info(f"os.makedirs({img_dir})")
                os.makedirs(img_dir)
            f_path = os.path.join(img_dir, f"epoch{epoch:05d}.png")
            images[i].save(f_path)
        # for
        if b_idx == 0: log_info(f"sample_online() Saved {img_cnt} images: {f_path}")

# class
