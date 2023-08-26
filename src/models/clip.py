from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTokenizerTransform:
    def __init__(self, tokenizer, max_length=77):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, text):
        tok_out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = tok_out['input_ids'][...]
        attention_mask = 1 - tok_out['attention_mask'][...]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    def __init__(self, encoder, device):
        super().__init__()
        self.transformer = encoder
        self.transformer = (
            self.transformer.eval().requires_grad_(False).to(device)
        )
        self.device = device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(
            input_ids=input_ids.to(self.device), output_hidden_states=True
        )
        return (
            clip_out.hidden_states[-1],
            cross_cond_padding.to(self.device),
            clip_out.pooler_output,
        )
