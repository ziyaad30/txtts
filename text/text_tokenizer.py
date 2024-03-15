import re
import string

import torch
from sklearn.model_selection import train_test_split

from text.cleaners import arpa_cleaners

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
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "CH",
    "DH",
    "EH",
    "ER",
    "EY",
    "HH",
    "IH",
    "IX",
    "IY",
    "JH",
    "NG",
    "OW",
    "OY",
    "SH",
    "TH",
    "TS",
    "UH",
    "UW",
    "ZH",
]

_valid_symbol_set = set(valid_symbols)
_arpabet = [s for s in valid_symbols]

_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Export all symbols:
new_symbols = list(_tokens) + _arpabet + list(_letters)

_symbol_to_id = {s: i for i, s in enumerate(new_symbols)}
_id_to_symbol = {i: s for i, s in enumerate(new_symbols)}


def remove_extraneous_punctuation(word):
    replacement_punctuation = {
        '{': '(',
        '}': ')',
        '[': '(',
        ']': ')',
        '—': '-',
        '`': '\'',
        'ʼ': '\''
    }
    replace = re.compile("|".join([re.escape(k) for k in sorted(replacement_punctuation, key=len, reverse=True)]),
                         flags=re.DOTALL)
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)

    # TODO: some of these are spoken ('@', '%', '+', etc). Integrate them into the cleaners.
    extraneous = re.compile(r'^[@#%_=\$\^&\*\+\\]$')
    word = extraneous.sub('', word)
    return word


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence([s for s in text.split()])


cmu_dict = {}
with open('./text/en_dictionary') as f:
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
                    try:
                        sequence += _arpabet_to_sequence(p)
                    except Exception as e:
                        sequence += _arpabet_to_sequence('[UNK]')
                break

        if not found:
            if phn not in string.punctuation:
                if phn == ' ':
                    sequence += _symbols_to_sequence(' ')
                else:
                    if str(phn).__contains__('(') or str(phn).__contains__(')'):
                        phn = phn.replace('(', '')
                        phn = phn.replace(')', '')
                        for word, pronunciation in cmu_dict.items():
                            if word == phn:
                                for p in pronunciation:
                                    sequence += _symbols_to_sequence(p)

                    elif str(phn).__contains__('.'):
                        print(f'stop found in {phn}')
                        with open(f"unknown.txt", 'a', encoding='utf-8') as f:
                            f.write(phn + '\n')
                        phn = phn.replace('.', '')
                        for word, pronunciation in cmu_dict.items():
                            if word == phn:
                                for p in pronunciation:
                                    sequence += _arpabet_to_sequence(p)

                    else:
                        # print(phn)
                        with open(f"unknown.txt", 'a', encoding='utf-8') as f:
                            f.write(phn + '\n')
                        sequence += _arpabet_to_sequence('[UNK]')
            else:
                sequence = sequence[:-1]
                sequence += _symbols_to_sequence(phn)

    return sequence


class TextBpeTokenizer:
    def __init__(self):
        print('Init TextBpeTokenizer')

    def preprocess_text(self, txt):
        txt = arpa_cleaners(txt)
        # txt = remove_extraneous_punctuation(txt)
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

    """ids = tokenizer.encode("after absorption into the cells the elements of the starch (or glucose) are, by the living protoplasm, in some unknown way")
    print(tokenizer.decode(ids))

    print("number_text_tokens", tokenizer.vocab_size())"""

    print("number_text_tokens", tokenizer.vocab_size())

    with open('../david_dataset/train.txt', encoding="utf8") as f:
        lines = f.readlines()

    train_data, valid_data = train_test_split(lines, test_size=0.2, random_state=1234, shuffle=True)

    for text in train_data:
        line = text.strip()
        wav, txt2 = line.split("|")
        ids = tokenizer.encode(txt2)
        print(tokenizer.decode(ids))

    for text in valid_data:
        line = text.strip()
        wav, txt2 = line.split("|")
        ids = tokenizer.encode(txt2)
        print(tokenizer.decode(ids))
