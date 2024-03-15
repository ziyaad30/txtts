import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from torch import LongTensor

from audio.injectors import TortoiseDiscreteTokenInjector, TortoiseMelSpectrogramInjector


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def get_prompt_slice(audio, max_audio_length=20, min_audio_length=3, sample_rate=24000, is_eval=False):
    max_sample_length = max_audio_length * sample_rate
    min_sample_length = min_audio_length * sample_rate
    rel_clip = audio
    # if eval uses a middle size sample when it is possible to be more reproducible
    if is_eval:
        sample_length = int((min_sample_length + max_sample_length) / 2)
    else:
        sample_length = random.randint(min_sample_length, max_sample_length)
    gap = rel_clip.shape[-1] - sample_length
    if gap < 0 and is_eval:
        sample_length = rel_clip.shape[-1]
    elif gap < 0:
        sample_length = rel_clip.shape[-1] // 2
    gap = rel_clip.shape[-1] - sample_length

    # if eval start always from the position 0 to be more reproducible
    if is_eval:
        rand_start = 0
    else:
        rand_start = random.randint(0, gap)

    rand_end = rand_start + sample_length
    rel_clip = rel_clip[:, rand_start:rand_end]
    return rel_clip


class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, dvae_path, is_eval=False):
        self.is_eval = is_eval
        if not self.is_eval:
            self.path = config['gpt_train']['train_file']
        else:
            self.path = config['gpt_train']['valid_file']

        self.audiopath_and_text = parse_filelist(self.path)
        self.tok = tokenizer

        self.sample_rate = config['vae_train']['sample_rate']
        self.n_mels = config['vae_train']['n_mels']
        self.power = config['vae_train']['power']

        try:
            self.mel_fmax = config['vae_train']['mel_fmax']
        except:
            self.mel_fmax = None

        self.dvae_path = dvae_path
        self.code_inj = TortoiseDiscreteTokenInjector({'in': 'mel', 'out': 'codes'}, self.dvae_path,
                                                      channels=self.n_mels)
        self.inj = TortoiseMelSpectrogramInjector({'in': 'wav', 'out': 'mel'},
                                                  n_mel_channels=self.n_mels,
                                                  sampling_rate=self.sample_rate,
                                                  mel_fmax=self.mel_fmax)
        self.mel_path = 'mels'

    def __getitem__(self, index):
        # Fetch text and add start/stop tokens.
        audiopath_and_text = self.audiopath_and_text[index]
        wav_file, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.tok.encode(text)
        text_tokens = LongTensor(text)

        audio, sr = torchaudio.load(wav_file)

        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)

        base_name = Path(wav_file).stem

        if not os.path.exists(f'{self.mel_path}/{base_name}.mel.pth'):
            print(f'{self.mel_path}/{base_name}.mel.pth')
            mel = self.inj({'wav': audio.unsqueeze(0)})['mel']
            torch.save(mel.cpu().detach(), f'{self.mel_path}/{base_name}.mel.pth')

        mel = torch.load(f'{self.mel_path}/{base_name}.mel.pth')

        if not os.path.exists(f'{self.mel_path}/{base_name}.melvq.pth'):
            print(base_name)
            code = self.code_inj({'mel': mel.to('cuda')})['codes']
            torch.save(code, f'{self.mel_path}/{base_name}.melvq.pth')

        mel_raw = mel[0]
        mel_codes = torch.load(f'{self.mel_path}/{base_name}.melvq.pth')[0]

        split = random.randint(int(mel_raw.shape[1] // 3), int(mel_raw.shape[1] // 3 * 2))
        if random.random() > 0.5:
            mel_refer = mel_raw[:, split:]
        else:
            mel_refer = mel_raw[:, :split]

        if mel_refer.shape[1] > 200:
            mel_refer = mel_refer[:, :200]

        if mel_raw.shape[1] > 400:
            mel_raw = mel_raw[:, :400]
            mel_codes = mel_codes[:100]

        return text_tokens, mel_codes, mel_raw, mel_refer

    def __len__(self):
        return len(self.audiopath_and_text)


class DiffusionCollater:
    def __init__(self):
        pass

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)
        mel_code_lens = [len(x[1]) for x in batch]
        max_mel_code_len = max(mel_code_lens)
        mel_lens = [x[2].shape[1] for x in batch]
        max_mel_len = max(mel_lens)
        mel_refer_lens = [x[3].shape[1] for x in batch]
        max_mel_refer_len = max(mel_refer_lens)
        texts = []
        mel_codes = []
        mels = []
        mel_refers = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for b in batch:
            text_token, mel_code, mel, mel_refer = b
            texts.append(F.pad(text_token, (0, max_text_len - len(text_token)), value=0))
            mel_codes.append(F.pad(mel_code, (0, max_mel_code_len - len(mel_code)), value=0))
            mels.append(F.pad(mel, (0, max_mel_len - mel.shape[1]), value=0))
            mel_refers.append(F.pad(mel_refer, (0, max_mel_refer_len - mel_refer.shape[1]), value=0))

        padded_text = torch.stack(texts)
        padded_mel_code = torch.stack(mel_codes)
        padded_mel = torch.stack(mels)
        padded_mel_refer = torch.stack(mel_refers)
        return {
            'padded_text': padded_text,
            'padded_mel_code': padded_mel_code,
            'padded_mel': padded_mel,
            'mel_lengths': LongTensor(mel_lens),
            'padded_mel_refer': padded_mel_refer,
            'mel_refer_lengths': LongTensor(mel_refer_lens)
        }
