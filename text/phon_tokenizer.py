import os

import torch
# load phonemizer
from phonemizer.backend import EspeakBackend

from text.cleaners import phoneme_cleaners
from text.symbols import symbols

if os.name == 'nt':
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    _ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'  # For Windows
    EspeakWrapper.set_library(_ESPEAK_LIBRARY)

# Mappings from symbol to numeric ID and vice versa:
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}


def phoneme_text(text):
    backend = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=False,
                            punctuation_marks=';:,.!?¡¿—…"«»“”()', language_switch='remove-flags')
    text = backend.phonemize([text], strip=True)[0]
    return text.strip()


def phoneme_to_sequence(text):
    sequence = []

    for symbol in text:
        if symbol in symbol_to_id.keys():
            symbol_id = symbol_to_id[symbol]
            sequence += [symbol_id]
        else:
            sequence.append(symbol_to_id['[UNK]'])

    # Append EOS token
    # sequence.append(symbol_to_id['~'])

    return sequence


def text_to_sequence(text):
    sequence = []

    # Phonemize text
    text = phoneme_text(text)
    # print(text)

    # Convert text to symbols
    for symbol in text:
        if symbol in symbol_to_id.keys():
            symbol_id = symbol_to_id[symbol]
            sequence += [symbol_id]
        else:
            sequence.append(symbol_to_id['[UNK]'])

    # Append EOS token
    # sequence.append(symbol_to_id['~'])
    return sequence


class PhonTokenizer:
    def __init__(self):
        print('Init PhonTokenizer')

    def preprocess_text(self, txt):
        txt = phoneme_cleaners(txt)
        return txt

    def encode(self, txt):
        # txt = self.preprocess_text(txt)
        ids = phoneme_to_sequence(txt)
        # print(self.decode(ids))
        return ids

    def vocab_size(self):
        return len(symbols)


if __name__ == '__main__':
    tokenizer = PhonTokenizer()
    text = "after absorption into the cells the elements of the starch (or glucose) are, by the living protoplasm, in some unknown way."
    ids = tokenizer.encode(text)
    print(ids)

    tokens = torch.IntTensor(ids)
    if torch.any(tokens == 1):
        raise Exception(f"[UNK] token found in ===> {text}")

    print("number_text_tokens", tokenizer.vocab_size())
