import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import LongTensor
from torch.utils.data import Dataset
from vocos.feature_extractors import MelSpectrogramFeatures

from text.bpe_tokenizer import VoiceBpeTokenizer
from xtts.dvae import DiscreteVAE


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class GptDataset(Dataset):
    def __init__(self, config):
        self.path = config['gpt_train']['train_file']
        self.dvae_path = config['gpt_train']['dvae_path']
        self.tokenizer = VoiceBpeTokenizer()
        self.seed = config['gpt_train']['seed']
        self.mel_path = config['gpt_train']['mel_dir']
        self.audiopath_and_text = parse_filelist(self.path)

        random.seed(self.seed)
        random.shuffle(self.audiopath_and_text)

        self.dvae = DiscreteVAE(
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
        ).cuda()

        self.dvae.eval()
        dvae_checkpoint = torch.load(self.dvae_path, map_location=torch.device("cpu"))
        self.dvae.load_state_dict(dvae_checkpoint, strict=False)

        self.mel_extractor = MelSpectrogramFeatures(n_mels=80).cuda()

    def get_text(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = torch.IntTensor(tokens)
        assert not torch.any(tokens == 1), f"UNK token found in {text} -> {self.tokenizer.decode(tokens)}"
        # The stop token should always be sacred.
        assert not torch.any(tokens == 0), f"Stop token found in {text}"
        return tokens

    def __getitem__(self, index):
        audiopath_and_text = self.audiopath_and_text[index]
        wav_file, text = audiopath_and_text[0], audiopath_and_text[1]

        tseq = self.get_text(text)

        audio, sr = torchaudio.load(wav_file)

        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        if sr != 24000:
            audio = torchaudio.transforms.Resample(sr, 24000)(audio).cuda()
        else:
            audio = audio.cuda()

        base_name = Path(wav_file).stem

        mel = self.mel_extractor(audio)

        if not os.path.exists(f'{self.mel_path}/{base_name}.mel.pth'):
            print(f"Getting {base_name}")
            mel_path = f'mels/{base_name}.mel.pth'
            torch.save(mel.cpu().detach(), mel_path)
            raw_mel = torch.load(mel_path)[0]
        else:
            mel_path = f'{self.mel_path}/{base_name}.mel.pth'
            raw_mel = torch.load(mel_path)[0]

        if not os.path.exists(f'{self.mel_path}/{base_name}.melvq.pth'):
            code = self.dvae.get_codebook_indices(mel)
            quant_path = f'mels/{base_name}.melvq.pth'
            torch.save(code.tolist(), quant_path)
            qmel = LongTensor(torch.load(quant_path)[0])
        else:
            quant_path = f'{self.mel_path}/{base_name}.melvq.pth'
            qmel = LongTensor(torch.load(quant_path)[0])

        wav_length = torch.tensor(audio.shape[-1], dtype=torch.long)

        split = random.randint(int(raw_mel.shape[1] // 3), int(raw_mel.shape[1] // 3 * 2))
        if random.random() > 0.5:
            raw_mel = raw_mel[:, :split]
        else:
            raw_mel = raw_mel[:, split:]

        if tseq.shape[0] > 400 or qmel.shape[0] > 600:
            print('tseq or qmel exceeds limits!')
            return None

        return tseq, qmel, raw_mel, wav_length

    def __len__(self):
        return len(self.audiopath_and_text)

    def collate_fn(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)

        qmel_lens = [len(x[1]) for x in batch]
        max_qmel_len = max(qmel_lens)

        raw_mel_lens = [x[2].shape[1] for x in batch]
        max_raw_mel_len = max(raw_mel_lens)

        wav_lens = [x[3] for x in batch]
        max_wav_len = max(wav_lens)

        texts = []
        qmels = []
        raw_mels = []

        for b in batch:
            text, qmel, raw_mel, wav_length = b
            text = F.pad(text, (0, max_text_len - len(text)), value=0)
            texts.append(text)
            qmels.append(F.pad(qmel, (0, max_qmel_len - len(qmel)), value=0))
            raw_mels.append(F.pad(raw_mel, (0, max_raw_mel_len - raw_mel.shape[1]), value=0))

        padded_qmel = torch.stack(qmels)
        padded_raw_mel = torch.stack(raw_mels)
        padded_texts = torch.stack(texts)
        return {
            'text_inputs': padded_texts,
            'text_lengths': LongTensor(text_lens),
            'audio_codes': padded_qmel,
            # 'qmel_lengths': LongTensor(qmel_lens),
            'cond_mels': padded_raw_mel,
            'cond_lens': LongTensor(raw_mel_lens),
            'wav_lengths': LongTensor(wav_lens)
        }
