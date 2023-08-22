from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTokenizerTransform:
    def __init__(self, version='openai/clip-vit-large-patch14', max_length=77):
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = tok_out['input_ids'][indexer]
        attention_mask = 1 - tok_out['attention_mask'][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version='openai/clip-vit-large-patch14', device='cuda'):
        super().__init__()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer = (
            self.transformer.eval().requires_grad_(False).to(device)
        )

    @property
    def device(self):
        return self.transformer.device

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
