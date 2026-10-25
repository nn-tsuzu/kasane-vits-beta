_pad = '_'
_punctuation = ';:,.!?¡¿—-…"«»“”/\\()「」『』[]|‖̯・ '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒɛ̃œ̃ʔd̥ʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ˥˦˧˨˩ˀ'͡''̚'"

# Export all symbols:
symbols = list(dict.fromkeys(
    [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
))

# Special symbol ids
SPACE_ID = symbols.index(" ")
