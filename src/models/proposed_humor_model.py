import torch
import torch.nn as nn


class ProposedHumorModel(nn.Module):
    """
    Proposed Model: Humor-aware transformer with Setup–Punchline Attention (SPA) and Incongruity features.
    Design goals:
      1) Strong base semantics from pretrained transformer
      2) Capture setup→punchline interaction (cross-attention)
      3) Model incongruity via semantic shift between setup & punchline
    """
    def __init__(
        self,
        vocab_size=None,                         # kept for compatibility
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        spa_heads: int = 1,
        setup_ratio: float = 0.7,                # first 70% tokens considered "setup"
        freeze_backbone: bool = False,
        embedding_dim = None
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoConfig
        except ImportError as e:
            raise ImportError("Please install `transformers` (pip install transformers)") from e

        self.setup_ratio = setup_ratio
        self.model_name = model_name

        # Backbone
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        H = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(H)

        # Setup–Punchline Cross-Attention (queries=punchline, keys/values=setup)
        self.spa = nn.MultiheadAttention(embed_dim=H, num_heads=spa_heads, batch_first=True)

        # Incongruity MLP: compare mean(setup) vs mean(punchline)
        incong_in = H * 4  # [setup, punch, setup - punch, setup * punch]
        self.incongruity_mlp = nn.Sequential(
            nn.Linear(incong_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )

        # Final classifier over concatenated features
        # Features: CLS/pooled (H) + masked mean (H) + SPA pooled (H) + incongruity (64)
        fused_in = H * 3 + 64
        self.head = nn.Sequential(
            nn.Linear(fused_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        # init lightweight heads
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _mean_pool(x: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            return x.mean(dim=1)
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _split_setup_punch(self, length: int, mask: torch.Tensor = None):
        # Use attention_mask to estimate real length if provided
        if mask is not None:
            length = mask.sum(dim=1)  # (B,)
        return length

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # DistilBERT doesn't accept token_type_ids, only BERT models do
        backbone_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': True,
            'return_dict': True
        }
        
        # Only add token_type_ids for BERT models (not DistilBERT/DistilRoBERTa)
        if token_type_ids is not None and not any(x in self.model_name.lower() for x in ['distilbert', 'distilroberta']):
            backbone_inputs['token_type_ids'] = token_type_ids
            
        out = self.backbone(**backbone_inputs)
        last = out.last_hidden_state            # (B, L, H)
        
        # DistilBERT doesn't have pooler_output, so always use mean pooling for it
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = self._mean_pool(last, attention_mask)
        pooled = self.ln(self.dropout(pooled))  # (B, H)

        mean_all = self._mean_pool(last, attention_mask)  # (B, H)

        # ---- Setup vs Punchline segmentation ----
        B, L, H = last.shape
        if attention_mask is not None:
            true_len = attention_mask.sum(dim=1)                      # (B,)
            setup_len = (true_len.float() * self.setup_ratio).floor().clamp(min=1, max=L-1).long()
        else:
            true_len = torch.full((B,), L, device=last.device, dtype=torch.long)
            setup_len = (true_len.float() * self.setup_ratio).floor().clamp(min=1, max=L-1).long()

        # Build batch slices
        setup_feats = []
        punch_feats = []
        spa_pooled = []
        for b in range(B):
            slen = setup_len[b].item()
            tlen = true_len[b].item()
            # safe slice bounds
            sl = last[b:b+1, :slen, :]                  # (1, S, H)
            pl = last[b:b+1, slen:tlen, :]              # (1, P, H) possibly empty if very short
            if pl.size(1) == 0:                         # degenerate: no punchline tokens → fallback
                pl = last[b:b+1, max(1, tlen-1):tlen, :]

            # Mean pools
            setup_feats.append(sl.mean(dim=1))          # (1, H)
            punch_feats.append(pl.mean(dim=1))          # (1, H)

            # Cross-attend: Q=punch, K/V=setup
            attn_out, _ = self.spa(query=pl, key=sl, value=sl, need_weights=False)
            spa_pooled.append(attn_out.mean(dim=1))     # (1, H)

        setup_mean = torch.cat(setup_feats, dim=0)      # (B, H)
        punch_mean = torch.cat(punch_feats, dim=0)      # (B, H)
        spa_feat   = torch.cat(spa_pooled, dim=0)       # (B, H)

        # Incongruity features
        incong = torch.cat([setup_mean, punch_mean, setup_mean - punch_mean, setup_mean * punch_mean], dim=1)  # (B, 4H)
        incong = self.incongruity_mlp(self.dropout(incong))  # (B, 64)

        # Final fusion
        fused = torch.cat([pooled, mean_all, spa_feat, incong], dim=1)  # (B, 3H+64)
        logits = self.head(self.dropout(fused))                         # (B, 2)
        return logits
