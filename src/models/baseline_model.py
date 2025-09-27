import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    """
    Super Simple Baseline Model: CNN-only for humor detection
    Minimalistic traditional text classification approach with single CNN layer and global max pooling
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=32, num_classes=2, dropout_rate=0.2):
        super(BaselineModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Single CNN layer - super simple approach
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)            # (B, L, E)
        x = x.transpose(1, 2)                    # (B, E, L)
        
        # Single CNN layer with ReLU activation
        x = F.relu(self.conv(x))                 # (B, hidden_dim, L)
        
        # Global max pooling to get fixed-size representation
        x = F.max_pool1d(x, kernel_size=x.size(2))  # (B, hidden_dim, 1)
        x = x.squeeze(2)                         # (B, hidden_dim)
        
        # Classification with dropout
        logits = self.classifier(self.dropout(x))
        return logits
