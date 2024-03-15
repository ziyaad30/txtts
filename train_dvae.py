import copy
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.custom_dataset import DvaeMelDataset
from models.dvae.dvae import DiscreteVAE
from utils.utils import plot_spectrogram_to_numpy, summarize
from utils.utils import oldest_checkpoint_path, latest_checkpoint_path


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def get_grad_norm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(name)
    total_norm = total_norm ** (1. / 2)
    return total_norm


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, cfg_path='configs/config.json'):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.cfg = json.load(open(cfg_path))

        self.sample_rate = self.cfg['vae_train']['sample_rate']
        self.n_mels = self.cfg['vae_train']['n_mels']

        self.dvae = DiscreteVAE(channels=self.n_mels,
                                num_tokens=8192,
                                hidden_dim=512,
                                num_resnet_blocks=3,
                                codebook_dim=512,
                                num_layers=2,
                                positional_dims=1,
                                kernel_size=3,
                                use_transposed_convs=False)
        self.dataset = DvaeMelDataset(self.cfg)
        self.dataloader = DataLoader(self.dataset, **self.cfg['vae_dataloader'])
        self.train_steps = self.cfg['vae_train']['train_steps']
        self.val_freq = self.cfg['vae_train']['eval_interval']
        if self.accelerator.is_main_process:
            self.logs_folder = Path(self.cfg['vae_train']['logs_dir'])
            self.logs_folder.mkdir(exist_ok=True, parents=True)
            self.mel_folder = Path('mels')
            self.mel_folder.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.logs_folder))
        #   self.ema_updater = EMA(0.999)
        self.optimizer = AdamW(self.dvae.parameters(),
                               lr=self.cfg['vae_train']['lr'],
                               betas=(0.9, 0.9999),
                               weight_decay=0.01)
        self.dvae, self.dataloader, self.optimizer = self.accelerator.prepare(self.dvae,
                                                                              self.dataloader,
                                                                              self.optimizer)
        self.dataloader = cycle(self.dataloader)
        self.step = 1
        self.gradient_accumulate_every = 1
        self.load()

    def _get_target_encoder(self, model):
        target_encoder = copy.deepcopy(model)
        set_requires_grad(target_encoder, False)
        for p in target_encoder.parameters():
            p.DO_NOT_TRAIN = True
        return target_encoder

    def save(self):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.dvae),
        }
        torch.save(data, str(self.logs_folder / f'dvae_{self.step}.pth'))
        keep_ckpts = self.cfg['vae_train']['keep_ckpts']
        old_ckpt = oldest_checkpoint_path(f"{self.logs_folder}", f"dvae_[0-9]*", preserved=keep_ckpts)
        if os.path.exists(old_ckpt):
            print(f"Removed {old_ckpt}")
            os.remove(old_ckpt)

    def load(self):
        try:
            print('Loading saved model...')
            dvae_model_path = latest_checkpoint_path(f"{self.logs_folder}", f"dvae_[0-9]*")
            # dvae_model_path = "pretrained_models/dvae.pth"  # This is original dvae.pth
            print(f'Loading model path {dvae_model_path}')
            dvae_checkpoint = torch.load(dvae_model_path, map_location="cpu")
            self.step = dvae_checkpoint['step'] + 1
            if 'model' in dvae_checkpoint:
                dvae_checkpoint = dvae_checkpoint['model']
            self.dvae.load_state_dict(dvae_checkpoint, strict=True)
        except Exception as e:
            print(e)

    def train(self):
        with tqdm(initial=self.step, total=self.train_steps, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < self.train_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    mel = next(self.dataloader)
                    mel = mel.to(self.device).squeeze(1)
                    with self.accelerator.autocast():
                        recon_loss, commitment_loss, mel_recon = self.dvae(mel)
                        recon_loss = torch.mean(recon_loss)
                        loss = recon_loss + 0.25 * commitment_loss
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)
                grad_norm = get_grad_norm(self.dvae)
                self.accelerator.clip_grad_norm_(self.dvae.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                self.accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accelerator.wait_for_everyone()

                if self.accelerator.is_main_process and self.step % self.val_freq == 0:
                    with torch.no_grad():
                        eval_model = self.accelerator.unwrap_model(self.dvae)
                        eval_model.eval()

                        mel_recon_ema = eval_model.infer(mel)[0]
                        eval_model.train()
                    scalar_dict = {"vae/loss": total_loss,
                                   "vae/loss_mel": recon_loss,
                                   "vae/loss_commitment": commitment_loss,
                                   "vae/grad_norm": grad_norm}
                    image_dict = {
                        "vae/spec": plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
                        "vae/spec_pred": plot_spectrogram_to_numpy(mel_recon[0, :, :].detach().unsqueeze(-1).cpu()),
                        "vae/spec_pred_ema": plot_spectrogram_to_numpy(
                            mel_recon_ema[0, :, :].detach().unsqueeze(-1).cpu()),
                    }
                    summarize(
                        writer=self.writer,
                        global_step=self.step,
                        images=image_dict,
                        scalars=scalar_dict
                    )
                if self.accelerator.is_main_process and self.step % self.cfg['vae_train']['save_freq'] == 0:
                    self.save()

                self.step += 1
                pbar.update(1)
        self.save()
        self.accelerator.print('training complete')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
