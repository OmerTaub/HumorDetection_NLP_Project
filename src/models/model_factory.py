import torch
from .baseline_model import BaselineModel
from .transformer_model import TransformerBasedModel
from .proposed_humor_model import ProposedHumorModel


class ModelFactory:
    """Factory class for creating different model architectures"""
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == 'baseline':
            return BaselineModel(**kwargs)
        elif model_type == 'transformer':
            # kwargs may include vocab_size, model_name, etc.
            return TransformerBasedModel(**kwargs)
        elif model_type == 'proposed':
            return ProposedHumorModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_model_info():
        return {
            'baseline': "Super simple CNN-only baseline model (single conv layer + global max pooling)",
            'transformer': "BERT backbone + linear head for binary humor classification",
            'proposed': "Humor-aware transformer with Setup–Punchline Attention and incongruity modeling"
        }


# -------------------- Quick smoke test -------------------- #
def test_models():
    print("Testing Model Architectures")
    print("=" * 40)

    vocab_size = 100
    seq_length = 32
    batch_size = 16

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    models_to_test = [
        ('baseline', {'vocab_size': vocab_size}),
        # For transformer/proposed we keep vocab_size in kwargs for compatibility; real training should tokenize with the chosen HF model.
        ('transformer', {'vocab_size': vocab_size, 'model_name': 'bert-base-uncased'}),
        ('proposed', {'vocab_size': vocab_size, 'model_name': 'bert-base-uncased'})
    ]

    for i, (model_type, kwargs) in enumerate(models_to_test, 1):
        print(f"\n{i}. Testing {model_type.title()} Model")
        print("-" * 30)
        try:
            model = ModelFactory.create_model(model_type, **kwargs)
            with torch.no_grad():
                out = model(input_ids, attention_mask)
            print(f"{model_type.title()} Output Shape: {out.shape}")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Variable lengths
            for L in [16, 32, 64]:
                x = torch.randint(1, vocab_size, (2, L))
                m = torch.ones(2, L, dtype=torch.long)
                with torch.no_grad():
                    _ = model(x, m)
                print(f"  Length {L}: ✓")
        except Exception as e:
            print(f"Error testing {model_type}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    test_models()
