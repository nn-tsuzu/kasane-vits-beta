import re, unicodedata, regex

# English
def US_English_converter(text):
    from .Multi_Phonemizer import G2P_US_English_to_Phoneme
    text = G2P_US_English_to_Phoneme.phonemize(text)
    return text

def GB_English_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "en")
    return text 

# Asia
def Chinese_converter(text):
    from .Chinese_Phonemizer import G2P_Chinese_to_Phoneme
    model = G2P_Chinese_to_Phoneme()
    text = model(text)
    return text

def Cantonese_converter(text):
    from .Cantonese_Phonemizer import G2P_Cantonese_to_Phoneme
    text = G2P_Cantonese_to_Phoneme.phonemize(text)
    return text

def Japanese_converter(text):
    from .Japanese_Phonemizer import G2P_Japanese_to_Phoneme
    text = G2P_Japanese_to_Phoneme.phonemize(text)
    return text 

def Korean_converter(text):
    from hanipa import G2P_Korean_to_Phoneme
    g2p = G2P_Korean_to_Phoneme()
    text = g2p(text)
    return text  

def Arabic_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "ar")
    return text   

def Farsi_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "fa")
    return text  

def Persian_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "fa")
    return text  

# europe
def German_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "de")
    return text  

def Dutch_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "nl")
    return text  

def French_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "fr")
    return text  

def Italian_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "it")
    return text  

def Spanish_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "es")
    return text  

def Luxembourgish_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "lb")
    return text  

def Czech_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "cs")
    return text  

# russia and nordic
def Swedish_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "sv")
    return text  

def Russian_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "ru")
    return text  

# africa
def Swahili_converter(text):
    from .Multi_Phonemizer import G2P_multilang_to_Phoneme
    text = G2P_multilang_to_Phoneme.phonemize(text, "sw")
    return text  


SYMBOL_MAP = {
    # 引用符・括弧類の統一
    '《': '«', '》': '»',
    '【': '「', '】': '」',
    '“': '"', '”': '"', '„': '"',
    
    # ダッシュ・長音・マイナス類の統一（ハイフンに集約）
    '—': '-', '–': '-', '−': '-', '〜': '-',
    "‐": "-",
    
    # 特殊なカンマやピリオド類の統一
    '，': ',', '。': '.', '、': ',', '．': '.',
    '؟': '?', '；': ';',
}
ch_keywords = [
    "她", "你", "很",
    "看看", "好好", "剛剛", "刚刚", 
    "請問", "请问", 
    "一下", "起来", "起來", "下来", "下來",
]
ja_keywords = ["々", "ゝ", "塩"]
SYMBOL_MAP_TABLE = str.maketrans(SYMBOL_MAP)

class auto_g2p:
    """
    Text-to-IPA Converter with Automatic Language Selection!!
    言語を自動選択してくれるText-to-IPAコンバーター
    """
    
    def __init__(self):
        self.g2p_map = {
            "en-us": US_English_converter, # us English
            "en-gb": GB_English_converter, # uk English
            "AMB": self.hanzi_kanji_checker, # Hanzi, Kanji, Hanja
            "zh-cn": Chinese_converter, # China-Chinese (Simplified Chinese)
            "zh-tw": Chinese_converter, # Taiwan-Chinese (Traditional Chinese)
            "zh": Chinese_converter, # China-Chinese (Simplified Chinese)
            "yue": Cantonese_converter, # Cantonese
            "ja": Japanese_converter, # Japanese
            "ko": Korean_converter, # Korean
            "fa": Farsi_converter, # Farsi
            "ar": Arabic_converter, # Arabic
            "de": German_converter, # German
            "nl": Dutch_converter, # Dutch
            "fr": French_converter, # Freanch
            "it": Italian_converter, # Italian
            "es": Spanish_converter, # Spanich
            "cs": Czech_converter, # Czech
            "lb": Luxembourgish_converter, # Luxembourgish
            "sv": Swedish_converter, # Swedish
            "ru": Russian_converter, # Russian
            "sw": Swahili_converter, # Swahili
        }

    def __call__(self, text:str, language=None):
        # By Convert to ipa by tokens, can support text that contain same languages. 
        # 1. nomalize text
        # 2. split text by special tokens
        # 3. convert to ipa by tokens
        text = unicodedata.normalize("NFKC", text)
        text = text.translate(SYMBOL_MAP_TABLE)
        text = re.sub(r'\s+', ' ', text)

        tokens = regex.split(r'(\p{P})', text)
        tokens = list(filter(None, tokens))
        if tokens[-1] == "":
            del tokens[-1]

        return self.text_convertor(tokens, language)
    
    def text_convertor(self, tokens, language=None):
        result = []
        language_list = [self.auto_detect_language(t, language) for t in tokens]

        for token, lang in zip(tokens, language_list):
            if token == "" or token == None:
                continue

            if lang == "Punctuation" or token == " ":
                result.append(token)
                continue

            text = self.g2p_map[lang](token)
            result.extend(text)

        result = "".join(result)
        result = result.replace("  ", " ")
        return result

    def hanzi_kanji_checker(self, tokens):
        if any(word in tokens for word in ch_keywords):
            try:
                output = Chinese_converter(tokens)
            except:
                output = Japanese_converter(tokens)
        else:
            try: 
                output = Japanese_converter(tokens)
            except: 
                output = Chinese_converter(tokens)
        return output

    def auto_detect_language(self, text, language=None) -> str:
        if language is None:
            if re.search(r"[ぁ-んァ-ン]", text):
                return "ja"
            if re.search(r"[가-힣]", text):
                return "ko"
            if re.search(r"[一-龯]", text):
                if re.search(r"[嘅咗喺冇佢哋啲咩呢嘛吖噉乜]", text):
                    return "yue" # cantonese
                else:
                    return "AMB" # CH or JP
            if re.search(r"[А-Яа-яЁё]", text):
                return "ru"
            if re.search(r"[گچپژکی]", text):
                return "fa"
            if re.search(r"[\u0600-\u06FF]", text):
                return "ar"
            if re.search(r"[čřšžěů]", text):
                return "cs"
            if re.search(r"[ñ¿¡]", text):
                return "es"
            if re.search(r"[ß]", text):
                return "de"
            if re.search(r"[å]", text):
                return "sv"
            if re.search(r"[ë]", text):
                return "lb"
            if re.search(r"ij", text):
                return "nl"
            if re.search(r"[àâçèéêëîïôùûÿ]", text):
                return "fr"
            if re.search(r"[àèéìòù]", text):
                return "it"
            if re.search(r"ng'", text):
                return "sw"
            if re.search(r"[a-zA-Z]", text) or re.search(r"\d+", text):
                return "en-gb"
            return "Punctuation"
        else:
            if re.search(r"[a-zA-Z]|[àèéìòù]|[àâçèéêëîïôùûÿ]|[å]|[ß]|[ñ¿¡]|[čřšžěů]|[\u0600-\u06FF]|[А-Яа-яЁё]|[一-龯]|[嘅咗喺冇佢哋啲咩呢嘛吖噉乜]|[가-힣]|[ぁ-んァ-ン]", text) or re.search(r"\d+", text):
                return language
            else:
                return "Punctuation"


if __name__ == "__main__":
    g2p = auto_g2p()
    t  = g2p(text="你好，世界！こんにちは。Programing 쪼와요~")
    print("mix: ", t) 

    t  = g2p(text="１週間して、そのニュースは本当になった。", language="ja")
    print("Japanese: ", t)

    t  = g2p("und sie, die jezt nur unsere Parodisten sind, durch eine gründliche Anweisung, den Daumen und die Fingerspizen zusammen zu bringen, so daß sie mindestens eine Schreibfeder führen können, zu uns herauf ziehen mögen", language="de")   
    print("german: ", t)

    # hangugəɾɯl` maɾɦajo.
    # hangugʌɾɯl malhaj͡o.