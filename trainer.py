import json
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from vocos.feature_extractors import MelSpectrogramFeatures

from MyTTS import MyTts
from common.tts_dataset import GptDataset
from text.bpe_tokenizer import VoiceBpeTokenizer
from utils.arch_utils import TorchMelSpectrogram
from utils.utils import latest_checkpoint_path, oldest_checkpoint_path, summarize, plot_spectrogram_to_numpy
from xtts.dvae import DiscreteVAE
from xtts.gpt_dataset import GptTtsDataset

my_xtts = "model"


def save_model(state_dict, step, pre_fix, model_dir):
    torch.save(
        {
            "model": state_dict,
            "step": step,
        },
        f=f"{model_dir}/{pre_fix}_{step}.pth"
    )
    old_ckpt = oldest_checkpoint_path(f"{model_dir}", f"{pre_fix}_[0-9]*", preserved=2)
    if os.path.exists(old_ckpt):
        print(f"Removed {old_ckpt}")
        os.remove(old_ckpt)


def load_ckpt(model, optimizer, scheduler, pre_fix, model_dir):
    model_path = latest_checkpoint_path(f"{model_dir}", f"{pre_fix}_[0-9]*")
    checkpoint_dict = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint_dict['model']
    try:
        step = checkpoint_dict['step'] + 1
    except:
        step = 1
    model.load_state_dict(state_dict, strict=True)
    try:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        scheduler.load_state_dict(checkpoint_dict["scheduler"])
    except:
        print('>>> Not loading scheduler and optimizer')
        pass
    return model, step, optimizer, scheduler


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


def eval_model(model, writer, step, mel_extractor, speaker, text):
    model.gpt.init_gpt_for_inference(kv_cache=True, use_deepspeed=False)
    model.cuda()
    model.gpt.eval()

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker,
        gpt_cond_len=10,
        max_ref_length=10,
        sound_norm_refs=False
    )

    out = model.inference(
        text=text,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.75,
        length_penalty=1,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.85,
    )

    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)

    mel = mel_extractor(out["wav"])[0]

    with torch.no_grad():
        image_dict = {
            "dvae/mel": plot_spectrogram_to_numpy(mel.squeeze()),
        }
        audios_dict = {
            "gpt/generated": out["wav"].detach().cpu(),
        }
        summarize(
            writer=writer,
            global_step=step,
            images=image_dict,
            audios=audios_dict
        )

    try:
        del model.gpt.gpt_inference
        del model.gpt.gpt.wte
    except Exception as e:
        print(e)

    model.gpt.train()


@torch.no_grad()  # torch no grad to avoid gradients from the pre-processing and DVAE codes extraction
def format_batch_on_device(batch, dvae, torch_mel_spectrogram_style_encoder, torch_mel_spectrogram_dvae):
    """Compute spectrograms on the device."""
    batch["text_lengths"] = batch["text_lengths"]
    batch["wav_lengths"] = batch["wav_lengths"]
    batch["text_inputs"] = batch["padded_text"]
    batch["cond_idxs"] = batch["cond_idxs"]
    # compute conditioning mel specs transform waves from torch.Size([B, num_cond_samples, 1, T] to torch.Size([B
    # * num_cond_samples, 1, T] because if is faster than iterate the tensor
    B, num_cond_samples, C, T = batch["conditioning"].size()
    conditioning_reshaped = batch["conditioning"].view(B * num_cond_samples, C, T)
    paired_conditioning_mel = torch_mel_spectrogram_style_encoder(conditioning_reshaped)
    # transform torch.Size([B * num_cond_samples, n_mel, T_mel]) in torch.Size([B, num_cond_samples, n_mel, T_mel])
    n_mel = torch_mel_spectrogram_style_encoder.n_mel_channels  # paired_conditioning_mel.size(1)
    T_mel = paired_conditioning_mel.size(2)
    paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond_samples, n_mel, T_mel)
    # get the conditioning embeddings
    batch["cond_mels"] = paired_conditioning_mel
    # compute codes using DVAE
    dvae_wav = batch["wav"]
    dvae_mel_spec = torch_mel_spectrogram_dvae(dvae_wav)
    codes = dvae.get_codebook_indices(dvae_mel_spec)

    batch["audio_codes"] = codes
    # delete useless batch tensors
    del batch["padded_text"]
    del batch["wav"]
    del batch["conditioning"]
    return batch


def work(config_path):
    config = json.load(open(config_path))

    batch_size = config['gpt_train']['batch_size']
    accum = config['gpt_train']['gradient_steps']

    loss_text_weight = config['gpt_train']['loss_text_weight']
    loss_mel_weight = config['gpt_train']['loss_mel_weight']

    learning_rate = config['gpt_train']['lr']
    milestones = config['gpt_train']['milestones']

    eval_save = config['gpt_train']['eval_step']
    log_step = config['gpt_train']['log_step']
    total_epochs = config['gpt_train']['total_epochs']

    log_dir = config['gpt_train']['log_dir']
    pre_fix = config['gpt_train']['pre_fix']
    model_dir = config['gpt_train']['model_dir']
    mel_dir = config['gpt_train']['mel_dir']

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(mel_dir):
        os.makedirs(mel_dir)

    # tokenizer = VoiceBpeTokenizer()

    # torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
    #     filter_length=2048,
    #     hop_length=256,
    #     win_length=1024,
    #     normalize=False,
    #     sampling_rate=22050,
    #     mel_fmin=0,
    #     mel_fmax=8000,
    #     n_mel_channels=80,
    #     mel_norm_file="./train_models/mel_stats.pth",
    # )

    """dvae = DiscreteVAE(
        channels=80,
        normalization=None,
        positional_dims=1,
        num_tokens=1024,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
    )
    dvae.cuda()
    dvae.eval()
    dvae_checkpoint = torch.load("./train_models/dvae.pth", map_location=torch.device("cpu"))
    dvae.load_state_dict(dvae_checkpoint, strict=False)"""

    mel_extractor = MelSpectrogramFeatures()

    # torch_mel_spectrogram_dvae = TorchMelSpectrogram(mel_norm_file="./train_models/mel_stats.pth",
    # sampling_rate=22050)

    # dataset = GptTtsDataset("./dataset/valid.txt", tokenizer)

    dataset = GptDataset(config)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, drop_last=True,
                        num_workers=0, shuffle=False)

    print(f">>> Training files loaded: {config['gpt_train']['train_file']}")

    xtts = MyTts(config)

    print('>>> Initializing optimizer and scheduler...')
    optimizer = torch.optim.Adam(xtts.gpt.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-6,
                                 weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    try:
        xtts, step, optimizer, scheduler = load_ckpt(xtts, optimizer, scheduler, pre_fix, model_dir)
        print(f">>> loaded model from step: {step}")
    except:
        print(f">>> start training new model from step: 1")
        step = 1

    xtts.gpt.cuda()
    xtts.gpt.train()

    writer = SummaryWriter(log_dir=log_dir)

    epoch = 0

    for epoch in range(epoch, total_epochs + 1):
        for idx, batch in enumerate(loader):
            # batch = format_batch_on_device(batch,
            # dvae, torch_mel_spectrogram_style_encoder, torch_mel_spectrogram_dvae)

            loss_dict = {}

            # cond_mels = batch["cond_mels"].cuda()
            # text_inputs = batch["text_inputs"].cuda()
            # text_lengths = batch["text_lengths"].cuda()
            # audio_codes = batch["audio_codes"].cuda()
            # wav_lengths = batch["wav_lengths"].cuda()
            # cond_idxs = batch["cond_idxs"].cuda()
            # cond_lens = batch["cond_lens"]

            # loss_text, loss_mel, mel_logits = xtts.gpt(
            #     text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
            # )

            text_inputs = batch['text_inputs'].cuda()
            text_lengths = batch['text_lengths'].cuda()
            audio_codes = batch['audio_codes'].cuda()
            wav_lengths = batch['wav_lengths'].cuda()
            cond_mels = batch['cond_mels'].cuda()
            cond_idxs = None
            cond_lens = batch['cond_lens'].cuda()

            """with torch.set_grad_enabled(True):"""

            loss_text, loss_mel, mel_logits = xtts.gpt(
                text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
            )

            loss = loss_text * loss_text_weight + loss_mel * loss_mel_weight
            # loss = loss / accum
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(xtts.gpt.parameters(), max_norm=1)

            """if step % accum == 0:
                print('>> optimizing...')"""

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            loss_dict["loss_text_ce"] = loss_text * loss_text_weight
            loss_dict["loss_mel_ce"] = loss_mel * loss_mel_weight
            loss_dict["loss"] = loss_dict["loss_text_ce"] + loss_dict["loss_mel_ce"]
            lr = scheduler.get_last_lr()[0]

            print(
                'Epoch: {}, '
                'Step: {}, '
                'loss_text_ce: {:.7f}, '
                'loss_mel_ce: {:.7f}, '
                'total_loss: {:.7f}, '
                'grad_norm: {:.7f}, '
                'lr: {:.7f}'
                .format(
                    epoch, step, loss_dict["loss_text_ce"], loss_dict["loss_mel_ce"], loss_dict["loss"],
                    grad_norm,
                    lr
                )
            )

            if step % log_step == 0:
                scalar_dict = {
                    "gpt/loss_mel": loss_mel * loss_text_weight,
                    "gpt/loss_text": loss_text * loss_mel_weight,
                    "gpt/total_loss": loss,
                    "gpt/grad_norm": grad_norm,
                    "gpt/lr": scheduler.get_last_lr()[0]
                }

                summarize(
                    writer=writer,
                    global_step=step,
                    scalars=scalar_dict
                )

            if step % eval_save == 0 and step > 0:
                print('>> saving...')
                save_model(xtts.state_dict(), step, pre_fix, model_dir)

                eval_model(xtts, writer, step, mel_extractor,
                           speaker="./dataset/wavs/ljs_davis_10.wav",
                           text="the porter asked what he had got, and the answer was, a male subject.")

            step += 1
        epoch += 1


if __name__ == "__main__":
    work("configs/my_config.json")
