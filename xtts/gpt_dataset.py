import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import IntTensor

from MyTTS import load_audio
from text.bpe_tokenizer import VoiceBpeTokenizer


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def get_prompt_slice(gt_path, max_sample_length, min_sample_length, sample_rate, is_eval=False):
    rel_clip = load_audio(gt_path, sample_rate)
    # if eval uses a middle size sample when it is possible to be more reproducible
    if is_eval:
        sample_length = int((min_sample_length + max_sample_length) / 2)
    else:
        sample_length = random.randint(min_sample_length, max_sample_length)
    gap = rel_clip.shape[-1] - sample_length
    if gap < 0:
        sample_length = rel_clip.shape[-1] // 2
    gap = rel_clip.shape[-1] - sample_length

    # if eval start always from the position 0 to be more reproducible
    if is_eval:
        rand_start = 0
    else:
        rand_start = random.randint(0, gap)

    rand_end = rand_start + sample_length
    rel_clip = rel_clip[:, rand_start:rand_end]
    rel_clip = F.pad(rel_clip, pad=(0, max_sample_length - rel_clip.shape[-1]))
    cond_idxs = [rand_start, rand_end]
    return rel_clip, rel_clip.shape[-1], cond_idxs


class GptTtsDataset(torch.utils.data.Dataset):
    def __init__(self, config, eval=False):
        self.tokenizer = VoiceBpeTokenizer()
        self.path = config['gpt_train']['train_file']
        self.audiopaths_and_text = parse_filelist(self.path )
        self.sample_rate = 22050
        self.max_conditioning_length = 132300
        self.min_conditioning_length = 66150
        self.max_wav_len = 242550
        self.max_text_len = 200
        self.is_eval = eval
        self.use_masking_gt_prompt_approach = True

    def __getitem__(self, index):
        # Fetch text and add start/stop tokens.
        audiopath_and_text = self.audiopaths_and_text[index]
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]

        wav = load_audio(audiopath, self.sample_rate)

        text = self.tokenizer.encode(text)
        tseq = IntTensor(text)

        assert not torch.any(tseq == 1), f"UNK token found in {text} -> {self.tokenizer.decode(tseq)}"
        # The stop token should always be sacred.
        assert not torch.any(tseq == 0), f"Stop token found in {text}"

        if self.use_masking_gt_prompt_approach:
            # get a slice from GT to condition the model
            cond, _, cond_idxs = get_prompt_slice(
                audiopath, self.max_conditioning_length, self.min_conditioning_length, self.sample_rate, self.is_eval
            )
            # if you use masking do not use cond_len
            cond_len = torch.nan
        else:
            cond, cond_len, _ = get_prompt_slice(
                audiopath, self.max_conditioning_length, self.min_conditioning_length, self.sample_rate, self.is_eval
            )
            # if you do not use masking use cond_len
            cond_idxs = torch.nan

        # Basically, this audio file is nonexistent or too long to be supported by the dataset.
        if (
                wav is None
                or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len)
                or (self.max_text_len is not None and tseq.shape[0] > self.max_text_len)
        ):
            return 'wav error'

        res = {
            # 'real_text': text,
            "text": tseq,
            "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
            "filenames": audiopath,
            "conditioning": cond.unsqueeze(1),
            "cond_lens": torch.tensor(cond_len, dtype=torch.long)
            if cond_len is not torch.nan
            else torch.tensor([cond_len]),
            "cond_idxs": torch.tensor(cond_idxs) if cond_idxs is not torch.nan else torch.tensor([cond_idxs]),
        }
        return res

    def __len__(self):
        return len(self.audiopaths_and_text)

    def collate_fn(self, batch):
        # convert list of dicts to dict of lists
        B = len(batch)

        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        # stack for features that already have the same shape
        batch["wav_lengths"] = torch.stack(batch["wav_lengths"])
        batch["text_lengths"] = torch.stack(batch["text_lengths"])
        batch["conditioning"] = torch.stack(batch["conditioning"])
        batch["cond_lens"] = torch.stack(batch["cond_lens"])
        batch["cond_idxs"] = torch.stack(batch["cond_idxs"])

        if torch.any(batch["cond_idxs"].isnan()):
            batch["cond_idxs"] = None

        if torch.any(batch["cond_lens"].isnan()):
            batch["cond_lens"] = None

        max_text_len = batch["text_lengths"].max()
        max_wav_len = batch["wav_lengths"].max()

        # create padding tensors
        text_padded = torch.IntTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, max_wav_len)

        # initialize tensors for zero padding
        text_padded = text_padded.zero_()
        wav_padded = wav_padded.zero_()
        for i in range(B):
            text = batch["text"][i]
            text_padded[i, : batch["text_lengths"][i]] = torch.IntTensor(text)
            wav = batch["wav"][i]
            wav_padded[i, :, : batch["wav_lengths"][i]] = torch.FloatTensor(wav)

        batch["wav"] = wav_padded
        batch["padded_text"] = text_padded
        return batch

