import re
import string

import torch

from text.cleaners import english_cleaners3

token_symbols = [
    "[STOP]",
    "[UNK]",
    " "
]

_token_symbol_set = set(token_symbols)
_tokens = [s for s in token_symbols]

valid_symbols = [
    "!",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "?",
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "CH",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "SH",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "ZH",
]

_valid_symbol_set = set(valid_symbols)
_arpabet = [s for s in valid_symbols]

_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Export all symbols:
new_symbols = list(_tokens) + _arpabet + list(_letters)

_symbol_to_id = {s: i for i, s in enumerate(new_symbols)}
_id_to_symbol = {i: s for i, s in enumerate(new_symbols)}


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence([s for s in text.split()])


cmu_dict = {}
with open('text/en_dictionary') as f:
    for entry in f:
        tokens = []
        for t in entry.split():
            tokens.append(t)
        cmu_dict[tokens[0]] = tokens[1:]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence([s for s in text.split()])


def sequence_to_text(sequence):
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            result += s
    return result


def text_to_sequence(text):
    sequence = []
    text = text.upper()
    text = text.replace('!', ' !')
    text = text.replace('.', ' .')
    text = text.replace(',', ' ,')
    text = text.replace(';', ' ;')
    text = text.replace('?', ' ?')
    text = text.replace(':', ' :')
    text = re.split(r'(\s)', text)

    for phn in text:
        found = False
        for word, pronunciation in cmu_dict.items():
            if word == phn:
                found = True
                for p in pronunciation:
                    sequence += _arpabet_to_sequence(p)
                break

        if not found:
            if phn not in string.punctuation:
                if phn == ' ':
                    sequence += _symbols_to_sequence(' ')
                else:
                    raise Exception(f'"{phn}" NOT FOUND IN DICTIONARY! ---> {text}')
            else:
                sequence = sequence[:-1]
                sequence += _symbols_to_sequence(phn)

    return sequence


class TextBpeTokenizer:
    def __init__(self):
        print('Init TextBpeTokenizer')

    def preprocess_text(self, txt):
        txt = english_cleaners3(txt)
        return txt

    def encode(self, txt):
        txt = self.preprocess_text(txt)
        ids = text_to_sequence(txt)
        # print(self.decode(ids))
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = sequence_to_text(seq)
        return txt

    def vocab_size(self):
        return len(new_symbols)


if __name__ == '__main__':
    tokenizer = TextBpeTokenizer()
    ids = tokenizer.encode("This is a test.")
    print(tokenizer.decode(ids))
    print("number_text_tokens", tokenizer.vocab_size())