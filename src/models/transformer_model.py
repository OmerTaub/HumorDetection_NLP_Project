import torch
import torch.nn as nn


class TransformerBasedModel(nn.Module):
    """
    BERT-based Model: HuggingFace backbone + linear head for binary classification.
    - Defaults to 'bert-base-uncased'
    - Accepts (input_ids, attention_mask[, token_type_ids])
    - Keeps 'vocab_size' arg for compatibility (ignored)
    """
    def __init__(
        self,
        vocab_size=None,                 # kept for compatibility with existing code
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False,
        max_seq_length = None
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoConfig
        except ImportError as e:
            raise ImportError("Please install `transformers` (pip install transformers)") from e

        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        # init head
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # DistilBERT doesn't accept token_type_ids, only BERT models do
        backbone_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'return_dict': True
        }
        
        # Only add token_type_ids for BERT models (not DistilBERT/DistilRoBERTa)
        if token_type_ids is not None and not any(x in self.model_name.lower() for x in ['distilbert', 'distilroberta']):
            backbone_inputs['token_type_ids'] = token_type_ids
            
        outputs = self.backbone(**backbone_inputs)
        # Prefer pooled_output when available; fallback to masked mean
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None \
                 else self._mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        return logits

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor = None):
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, L, 1)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom
