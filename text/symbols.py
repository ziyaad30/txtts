token_symbols = [
    "[STOP]",
    "[UNK]",
    " "
]

_token_symbol_set = set(token_symbols)
_tokens = [s for s in token_symbols]
_pad = '_'
_eos = '~'
_punctuation = ';:,.!?¡¿—…"«»“”-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = list(_tokens) + [_pad] + list(_eos) + list(_punctuation) + list(_letters) + list(_letters_ipa)

"""for i, s in enumerate(symbols):
    print(s, i)"""