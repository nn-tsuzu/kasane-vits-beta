import torch
from text.phonemize.symbols import symbols
from models import SynthesizerTrn, MultiPeriodDiscriminator
import utils

original_model_symbols = [
    '_' , '#' , '$' , '[' , '?' , ']' , '^' ,  
    'a' , 'b' , 'by', 'ch', 'cl', 'd' , 'dy',
    'e' , 'f' , 'g' , 'gy', 'h' , 'hy', 'i' ,
    'j' , 'k' , 'ky', 'm' , 'my', 'n' , 'N' , 
    'ny', 'o' , 'p' , 'py', 'r' , 'ry', 's' , 
    'sh', 't' , 'ts', 'ty', 'u' , 'v'  , 'w', 
    'y' , 'z'
]
news_model_symbols = symbols

# 新音素: 元音素
symbol_mapping = {
    "a":"a", "i":"i", "ɯ":"u", "e":"e", "o":"o",
    # 子音
    "b":"b", "t":"t", "s":"s", "ty":"ty", "v":"v",     
    "by":"by", "w":"w", "z":"z",      
    "cl":"cl",  "d":"dy", "g":"g", "h":"h",  "hy":"hy",   
    "k":"k",  "m":"m", "n":"n", "N":"n", "ny":"ny",  
    "p":"p", "py":"py", "s":"s",     
    # 変更した子音
    "j":"y",
    "d͡ʑa":"ja", "d͡ʑi":"ji", "d͡ʑɯ":"jyu", "d͡ʑɯ":"ju", "d͡ʑe":"je", "d͡ʑo":"jo",
    "ɸ":"f",
    "ɾ":"r",
    "ɾʲ":"ry",
    "kʲ":"ky",
    "gʲ":"gy",
    "ɲ":"ny",
    "ɕ":"sh",
    "t͡ɕ":"ch",
    "t͡s":"ts",
    "mʲ":"my",
    # ん
    "ŋk":"nk",
    "ŋg":"ng",
    "mp":"np",
    "mb":"nb",
    "mm":"nm",
}


def transfer_embeddings(
    model,
    pretrained_state,
    old_symbols,
    new_symbols,
    symbol_mapping=None,
):
    """
    model:
        Kasane-TTS のモデル

    pretrained_state:
        元モデルの state_dict

    old_symbols:
        元モデルの symbols リスト

    new_symbols:
        Kasane-TTS の symbols リスト

    symbol_mapping:
        {新しい音素: 元モデルの音素}
        の対応表
    """

    symbol_mapping = symbol_mapping or {}

    key = "enc_p.emb.weight"

    old_weight = pretrained_state[key]
    new_weight = model.enc_p.emb.weight

    assert len(old_symbols) == old_weight.shape[0]
    assert len(new_symbols) == new_weight.shape[0]
    assert old_weight.shape[1] == new_weight.shape[1]

    assert len(set(old_symbols)) == len(old_symbols)
    assert len(set(new_symbols)) == len(new_symbols)

    old_ids = {
        symbol: i
        for i, symbol in enumerate(old_symbols)
    }

    copied = []
    initialized = []

    with torch.no_grad():
        for new_id, new_symbol in enumerate(new_symbols):

            old_symbol = symbol_mapping.get(
                new_symbol,
                new_symbol
            )

            if old_symbol in old_ids:
                old_id = old_ids[old_symbol]

                new_weight[new_id].copy_(
                    old_weight[old_id].to(
                        device=new_weight.device,
                        dtype=new_weight.dtype
                    )
                )

                copied.append(new_symbol)

            else:
                # モデル作成時のランダム初期値を維持
                initialized.append(new_symbol)

    print(f"Copied: {len(copied)}")
    print(f"New: {len(initialized)}")
    print(f"New symbols: {initialized}")
    return copied, initialized

def transfer_load(checkpoint_path, config_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # mb_istft_vitsの保存形式に合わせて取り出す
    if 'model' in checkpoint:
        pretrained_state = checkpoint['model']
    else:
        pretrained_state = checkpoint

    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
      len(symbols), 
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model
    )
    new_state = net_g.state_dict()

    # enc_p.emb.weight
    for key, value in pretrained_state.items():
        if key == "enc_p.emb.weight":
            continue
        if key in new_state and new_state[key].shape == value.shape:
            new_state[key] = value
    net_g.load_state_dict(new_state)

    copied, initialized = transfer_embeddings(
        model=net_g,
        pretrained_state=pretrained_state,
        old_symbols=original_model_symbols,
        new_symbols=news_model_symbols,
        symbol_mapping=symbol_mapping
    )
    print(f"移植できた音素: {len(copied)}")
    print(f"新規初期化の音素: {len(initialized)}")


def load_skip_embeds_mb_istft_vits_G(checkpoint_path, config_path, symbol_mapping=None):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # mb_istft_vitsの保存形式に合わせて取り出す
    if 'model' in ckpt:
        pretrained_dict = ckpt['model']
    else:
        pretrained_dict = ckpt

    hps = utils.get_hparams_from_file(config_path)
    model = SynthesizerTrn(
      len(symbols), 
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model
    )

    model_dict = model.state_dict()
    matched, skipped_shape, skipped_missing = [], [], []
    new_state_dict = {}

    for k, v in model_dict.items():
        if k in pretrained_dict:
            if pretrained_dict[k].size() == v.size():
                new_state_dict[k] = pretrained_dict[k]
                matched.append(k)
            else:
                # 元重みより重みのsizeが増えている場合、その差ぶんの重みを疑似生成し末尾に加える
                new_state_dict[k] = v 
                skipped_shape.append(f"{k}: ckpt={pretrained_dict[k].size()} vs model={v.size()}")
                hidden_dim = v.shape[1]
                mean_val = v.mean().item()
                std_val = v.std().item()

                additional_weight = torch.randn(
                    (v.shape[0]-pretrained_dict[k].shape[0], hidden_dim), 
                    dtype=torch.bfloat16, 
                ) * std_val + mean_val

                # 元モデルのembed_tokensとadditional_weightを結合しサイズを拡張
                fixed_embed_weight = torch.cat([pretrained_dict[k], additional_weight], dim=0)                      
                new_state_dict[k] = fixed_embed_weight
                matched.append(fixed_embed_weight)
        else:
            new_state_dict[k] = v  # ランダム初期化のまま
            skipped_missing.append(k)

    model.load_state_dict(new_state_dict)
    print(f"\n流用成功: {len(matched)} layers")
    print(f"スキップ: {len(skipped_shape)} layers")

    for s in skipped_shape:
        print(f"   {s}")

    print(f"新規初期化: {len(skipped_missing)} layers")
    
    for s in skipped_missing:
        print(f"   {s}")
    return model, ckpt


def load_pretrained_mb_istft_vits_G(checkpoint_path, config_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # mb_istft_vitsの保存形式に合わせて取り出す
    if 'model' in ckpt:
        pretrained_dict = ckpt['model']
    else:
        pretrained_dict = ckpt

    hps = utils.get_hparams_from_file(config_path)
    model = SynthesizerTrn(
      len(symbols), 
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model
    )

    model_dict = model.state_dict()
    matched, skipped_shape, skipped_missing = [], [], []
    new_state_dict = {}

    for k, v in model_dict.items():
        if k in pretrained_dict:
            if pretrained_dict[k].size() == v.size():
                new_state_dict[k] = pretrained_dict[k]
                matched.append(k)
            else:
                # 元重みより重みのsizeが増えている場合、その差ぶんの重みを疑似生成し末尾に加える
                new_state_dict[k] = v 
                skipped_shape.append(f"{k}: ckpt={pretrained_dict[k].size()} vs model={v.size()}")

                hidden_dim = v.shape[1]
                mean_val = v.mean().item()
                std_val = v.std().item()
                print("pretrained_dict[k]: ", pretrained_dict[k].size())

                additional_weight = torch.randn(
                    (v.shape[0]-pretrained_dict[k].shape[0], hidden_dim), 
                    dtype=torch.bfloat16, 
                ) * std_val + mean_val
                print("additional_weight: ", additional_weight.size())

                # 元モデルのembed_tokensとadditional_weightを結合しサイズを拡張
                fixed_embed_weight = torch.cat([pretrained_dict[k], additional_weight], dim=0)
                print("fixed_embed_weight: ", fixed_embed_weight.size())
                      
                new_state_dict[k] = fixed_embed_weight
                matched.append(fixed_embed_weight)
        else:
            new_state_dict[k] = v  # ランダム初期化のまま
            skipped_missing.append(k)

    model.load_state_dict(new_state_dict)
    print(f"\n流用成功: {len(matched)} layers")
    print(f"スキップ: {len(skipped_shape)} layers")

    for s in skipped_shape:
        print(f"   {s}")

    print(f"新規初期化: {len(skipped_missing)} layers")
    
    for s in skipped_missing:
        print(f"   {s}")
    return model, ckpt


def load_pretrained_mb_istft_vits_D(checkpoint_path, config_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # mb_istft_vitsの保存形式に合わせて取り出す
    if 'model' in ckpt:
        pretrained_dict = ckpt['model']
    else:
        pretrained_dict = ckpt

    hps = utils.get_hparams_from_file(config_path)
    model = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    model_dict = model.state_dict()
    matched, skipped_shape, skipped_missing = [], [], []
    new_state_dict = {}

    for k, v in model_dict.items():
        if k in pretrained_dict:
            if pretrained_dict[k].size() == v.size():
                new_state_dict[k] = pretrained_dict[k]
                matched.append(k)
            else:
                # 元重みより重みのsizeが増えている場合、その差ぶんの重みを疑似生成し末尾に加える
                new_state_dict[k] = v 
                skipped_shape.append(f"{k}: ckpt={pretrained_dict[k].size()} vs model={v.size()}")
        else:
            new_state_dict[k] = v  # ランダム初期化のまま
            skipped_missing.append(k)

    model.load_state_dict(new_state_dict)
    print(f"\n流用成功: {len(matched)} layers")
    print(f"スキップ: {len(skipped_shape)} layers")

    for s in skipped_shape:
        print(f"   {s}")

    print(f"新規初期化: {len(skipped_missing)} layers")
    
    for s in skipped_missing:
        print(f"   {s}")
    return model, ckpt


# model G
model, ckpt = load_skip_embeds_mb_istft_vits_G(
    config_path="./configs/ljs_mb_istft_vits.json",
    checkpoint_path="./checkpoints/G_0.pth"
)
checkpoint = {
    'iteration': ckpt['iteration'], 
    'model': model.state_dict(),
    'learning_rate': ckpt['learning_rate'],
}
torch.save(checkpoint, "./checkpoints/G_finetune.pth")


# model D
model, ckpt = load_pretrained_mb_istft_vits_D(
    config_path="./configs/ljs_mb_istft_vits.json",
    checkpoint_path="./checkpoints/D_0.pth"
)
checkpoint = {
    'iteration': ckpt['iteration'], 
    'model': model.state_dict(),
}
torch.save(checkpoint, "./checkpoints/D_finetune.pth")
print("saved!")