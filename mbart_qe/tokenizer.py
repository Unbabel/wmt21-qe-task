import torch
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from transformers import AutoConfig, AutoTokenizer


LANGUAGE_MAPPING = {
    "en": "en_XX",
    "zh": "zh_CN",
    "de": "de_DE",
    "ru": "ru_RU",
    "et": "et_EE",    
    "ro": "ro_RO",
    "si": "si_LK",
    "ne": "ne_NP",
    "cs": "cs_CZ",
    "ja": "ja_XX",
    "ps": "ps_AF",
    "km": "km_KH",
    "fr": "fr_XX",
}

class Tokenizer:

    def __init__(self, pretrained_model) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.pad_index = self.tokenizer.pad_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.bos_index = self.tokenizer.eos_token_id
        self.stoi = self.tokenizer.get_vocab()
        self.itos = {v: k for k, v in self.stoi.items()}
        configs = AutoConfig.from_pretrained(pretrained_model)
        
        
    def batch_encode(self, sources: list, hypothesis: list, src_langs: list, tgt_langs: list) -> torch.Tensor:
        encoded_batch = {"input_ids": [], "attention_mask": [], "mt_input_ids": []}
        for src_lang, src in zip(src_langs, sources):
            self.tokenizer.src_lang = LANGUAGE_MAPPING[src_lang]
            tokenized_src = self.tokenizer(src, truncation=True, max_length=1024)
            encoded_batch["input_ids"].append(torch.tensor(tokenized_src["input_ids"]))
            encoded_batch["attention_mask"].append(torch.tensor(tokenized_src["attention_mask"]))
        
        for tgt_lang, mt in zip(tgt_langs, hypothesis):
            self.tokenizer.tgt_lang = LANGUAGE_MAPPING[tgt_lang]
            with self.tokenizer.as_target_tokenizer():
                encoded_batch["mt_input_ids"].append([self.bos_index,] + self.tokenizer(mt, truncation=True, max_length=1022).input_ids)
                
        model_input = {}
        for k, v in encoded_batch.items():
            if k == "mt_input_ids":
                #model_input["mt_labels"] = stack_and_pad_tensors([torch.tensor(l +[-100]) for l in v], padding_index=-100).tensor
                #model_input["mt_labels"] = model_input["mt_labels"][..., 1:].contiguous() # shift labels
                padded_input = stack_and_pad_tensors([torch.tensor(l) for l in v])
                model_input["mt_input_ids"] = padded_input.tensor
                model_input["mt_eos_ids"] = padded_input.lengths
            else:
                padded_input = stack_and_pad_tensors(v)
                model_input[k] = padded_input.tensor

        return model_input
