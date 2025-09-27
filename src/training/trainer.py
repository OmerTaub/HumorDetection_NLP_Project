import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import time
import json
from pathlib import Path
import sys
import os
from tqdm.auto import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.model_factory import ModelFactory
from src.evaluation.metrics import HumorDataLoader, HumorEvaluationMetrics
from src.models.simple_tokenizer import SimpleTokenizer


class ThreeModelTrainer:
    """Comprehensive trainer for all three model architectures (Baseline + BERT variants)"""
    def __init__(self, device='cpu'):
        self.device = device
        self.simple_tokenizer = None
        self.hf_tokenizer = None
        self._raw_splits = None  # store raw texts/labels for re-encoding per model

    # ---------- DATA PREP (Simple tokenizer: for baseline) ----------
    def prepare_data(self, dataset_name, vocab_size=100, max_seq_length=64):
        """
        Prepare data using SimpleTokenizer (for baseline model).
        Also caches raw splits for later HF tokenization.
        """
        print(f"Preparing data from dataset: {dataset_name}")

        loader = HumorDataLoader(
            data_dir=f'/home/omertaub/projects/nlp_proj_new/data/processed'
        )
        splits = loader.load_all_splits(dataset_name)

        train_texts, train_labels = splits['train']
        val_texts, val_labels = splits['val']
        test_texts, test_labels = splits['test']

        # cache raw for BERT models
        self._raw_splits = {
            'train': (train_texts, train_labels),
            'val':   (val_texts, val_labels),
            'test':  (test_texts, test_labels),
        }

        print(f"Loaded: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")

        # Simple tokenizer for baseline
        self.simple_tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        # IMPORTANT: Only use training data for vocabulary building to prevent data contamination
        self.simple_tokenizer.build_vocab(train_texts)

        train_data = self._encode_dataset_simple(train_texts, train_labels, max_seq_length)
        val_data   = self._encode_dataset_simple(val_texts,   val_labels,   max_seq_length)
        test_data  = self._encode_dataset_simple(test_texts,  test_labels,  max_seq_length)

        return train_data, val_data, test_data, len(self.simple_tokenizer.word_to_idx)

    def _encode_dataset_simple(self, texts, labels, max_seq_length):
        encoded = []
        for text, label in zip(texts, labels):
            ids = self.simple_tokenizer.encode(text, max_seq_length)
            mask = [1 if t != 0 else 0 for t in ids]
            encoded.append({
                'input_ids': ids,
                'attention_mask': mask,
                'label': label
            })
        return encoded

    # ---------- DATA PREP (HF tokenizer: for transformer/proposed) ----------
    def prepare_hf_data(self, model_name='bert-base-uncased', max_seq_length=64):
        """Prepare HF-tokenized datasets for BERT-based models."""
        if self._raw_splits is None:
            raise RuntimeError("Call prepare_data() first to cache raw splits.")

        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError("Please install `transformers` (pip install transformers)") from e

        if self.hf_tokenizer is None:
            print(f"Loading HF tokenizer: {model_name}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        def _enc(texts, labels):
            enc = self.hf_tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_seq_length,
                return_attention_mask=True
            )
            out = []
            for i in range(len(texts)):
                item = {
                    'input_ids': enc['input_ids'][i],
                    'attention_mask': enc['attention_mask'][i],
                    'label': labels[i]
                }
                if 'token_type_ids' in enc:
                    item['token_type_ids'] = enc['token_type_ids'][i]
                out.append(item)
            return out

        (tr_x, tr_y) = self._raw_splits['train']
        (va_x, va_y) = self._raw_splits['val']
        (te_x, te_y) = self._raw_splits['test']

        train_data = _enc(tr_x, tr_y)
        val_data   = _enc(va_x, va_y)
        test_data  = _enc(te_x, te_y)

        return train_data, val_data, test_data

    # ---------- DataLoader ----------
    def create_dataloader(self, data, batch_size=16, shuffle=True):
        from torch.utils.data import Dataset, DataLoader

        class HumorDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                item = self.data[idx]
                out = {
                    'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
                    'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
                    'label': torch.tensor(item['label'], dtype=torch.long)
                }
                if 'token_type_ids' in item:
                    out['token_type_ids'] = torch.tensor(item['token_type_ids'], dtype=torch.long)
                return out

        return DataLoader(HumorDataset(data), batch_size=batch_size, shuffle=shuffle)

    # ---------- Train/Eval ----------
    def train_epoch(self, model, dataloader, optimizer, criterion):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc="Training", unit="batch", leave=False)
        ema_loss = None

        for _, batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            labels = batch['label'].to(self.device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids) if token_type_ids is not None \
                     else model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            ema_loss = loss_val if ema_loss is None else (0.9 * ema_loss + 0.1 * loss_val)
            running_acc = correct / max(total, 1)
            current_lr = optimizer.param_groups[0].get('lr', None)
            pbar.set_postfix({
                "loss(ema)": f"{ema_loss:.4f}",
                "acc": f"{running_acc:.3f}",
                **({"lr": f"{current_lr:.2e}"} if current_lr is not None else {})
            })

        return total_loss / len(dataloader), correct / max(total, 1)

    def evaluate_model(self, model, dataloader, criterion):
        model.eval()
        total_loss = 0.0
        all_predictions, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                labels = batch['label'].to(self.device)

                logits = model(input_ids, attention_mask, token_type_ids) if token_type_ids is not None \
                         else model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        metrics = HumorEvaluationMetrics()
        metrics.update(all_labels, all_predictions)
        return avg_loss, metrics.get_all_metrics(), all_predictions, all_labels

    def train_model(self, model_type, train_data, val_data, test_data, vocab_size,
                    epochs=10, batch_size=16, learning_rate=0.001, max_seq_length=64,
                    model_name='bert-base-uncased', **model_kwargs):
        """Train a specific model. For BERT-based models, pass HF-tokenized datasets.
        
        Args:
            **model_kwargs: Additional model parameters (e.g., spa_heads, setup_ratio, freeze_backbone)
        """
        print(f"\nTraining {model_type.upper()} Model")
        print("=" * 50)

        # Build model args with defaults, then override with any provided kwargs
        if model_type == 'baseline':
            default_kwargs = {'vocab_size': vocab_size, 'num_classes': 2, 'dropout_rate': 0.2,
                            'embedding_dim': 32, 'hidden_dim': 32}
        elif model_type == 'transformer':
            default_kwargs = {'vocab_size': vocab_size, 'num_classes': 2, 'dropout_rate': 0.1,
                            'model_name': model_name}
        elif model_type == 'proposed':
            default_kwargs = {'vocab_size': vocab_size, 'num_classes': 2, 'dropout_rate': 0.2,
                            'model_name': model_name}
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Merge defaults with provided kwargs (kwargs take precedence)
        final_kwargs = {**default_kwargs, **model_kwargs}
        
        print(f"Model kwargs: {final_kwargs}")  # Debug print to verify parameters
        model = ModelFactory.create_model(model_type, **final_kwargs).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model created with {n_params:,} parameters")

        # Dataloaders
        train_loader = self.create_dataloader(train_data, batch_size, shuffle=True)
        val_loader   = self.create_dataloader(val_data,   batch_size, shuffle=False)
        test_loader  = self.create_dataloader(test_data,  batch_size, shuffle=False)

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        if model_type in {'transformer', 'proposed'}:
            # Safer defaults for BERT fine-tuning
            lr = 2e-5 if learning_rate >= 1e-4 else learning_rate
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        best_val_f1, best_state = 0.0, None
        patience_counter, max_patience = 0, 3
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_f1': []}
        start_time = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)
            tr_loss, tr_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            va_loss, va_metrics, _, _ = self.evaluate_model(model, val_loader, criterion)
            va_f1 = va_metrics['f1_score']
            scheduler.step(va_loss)

            print(f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")
            print(f"Val   Loss: {va_loss:.4f}, Val F1: {va_f1:.4f}")

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(va_loss)
            history['val_f1'].append(va_f1)

            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"New best model! Val F1: {va_f1:.4f}")
            else:
                patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"Loaded best model with Val F1: {best_val_f1:.4f}")

        print(f"\nFinal Evaluation on Test Set:")
        print("-" * 35)
        test_loss, test_metrics, test_preds, test_labels = self.evaluate_model(model, test_loader, criterion)

        metrics_obj = HumorEvaluationMetrics()
        metrics_obj.update(test_labels, test_preds)
        metrics_obj.print_metrics(f"{model_type.upper()} Test Results")

        train_secs = time.time() - start_time
        print(f"Training completed in {train_secs:.2f} seconds")

        model_dir = Path("models") / "three_architectures"
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / f"{model_type}_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'model_kwargs': model_kwargs,
            'vocab_size': vocab_size,
            'test_metrics': test_metrics,
            'training_history': history,
            'training_time': train_secs
        }, model_path)
        print(f"Model saved to: {model_path}")

        return model, test_metrics, history


def train_three_models(dataset_name="200k_jokes"):
    """Run comprehensive comparison of baseline + BERT models with proper tokenization per model."""
    print("Three Model Architecture Comparison")
    print("=" * 50)
    print(f"Dataset: {dataset_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    trainer = ThreeModelTrainer(device)

    # Prepare baseline (simple) data and cache raw splits
    base_train, base_val, base_test, vocab_size = trainer.prepare_data(
        dataset_name, vocab_size=100, max_seq_length=48
    )

    # Prepare HF-tokenized data for BERT models
    hf_train, hf_val, hf_test = trainer.prepare_hf_data(model_name='bert-base-uncased', max_seq_length=48)

    models_to_train = ['baseline', 'transformer', 'proposed']
    training_config_baseline = dict(epochs=1, batch_size=16, learning_rate=0.001, max_seq_length=48)
    training_config_bert     = dict(epochs=1, batch_size=16, learning_rate=2e-5,  max_seq_length=48)

    all_results, all_histories = {}, {}

    for model_type in models_to_train:
        try:
            if model_type == 'baseline':
                tr, va, te, cfg = base_train, base_val, base_test, training_config_baseline
            else:
                tr, va, te, cfg = hf_train, hf_val, hf_test, training_config_bert

            model, test_metrics, history = trainer.train_model(
                model_type, tr, va, te, vocab_size, **cfg
            )
            all_results[model_type] = test_metrics
            all_histories[model_type] = history
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            import traceback; traceback.print_exc()

    # Save results
    results_dir = Path("results") / "three_model_comparison"
    results_dir.mkdir(exist_ok=True, parents=True)
    with open(results_dir / f"{dataset_name}_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(results_dir / f"{dataset_name}_histories.json", 'w') as f:
        json.dump(all_histories, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
    print("=" * 60)
    if all_results:
        print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        for name, m in all_results.items():
            print(f"{name.capitalize():<15} "
                  f"{m['accuracy']:<10.4f} "
                  f"{m['precision']:<12.4f} "
                  f"{m['recall']:<10.4f} "
                  f"{m['f1_score']:<10.4f}")

        best_model = max(all_results.items(), key=lambda x: x[1]['f1_score'])
        print(f"\n🏆 Best Model: {best_model[0].upper()}")
        print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")

        # Compute param counts live
        print(f"\nModel Complexity (Parameters):")
        for name in models_to_train:
            try:
                if name == 'baseline':
                    mk = {'vocab_size': vocab_size, 'num_classes': 2, 'dropout_rate': 0.2,
                          'embedding_dim': 32, 'hidden_dim': 32}
                elif name == 'transformer':
                    mk = {'vocab_size': vocab_size, 'num_classes': 2, 'dropout_rate': 0.1,
                          'model_name': 'bert-base-uncased'}
                else:
                    mk = {'vocab_size': vocab_size, 'num_classes': 2, 'dropout_rate': 0.2,
                          'model_name': 'bert-base-uncased'}
                m = ModelFactory.create_model(name, **mk)
                print(f"  {name.capitalize()}: {sum(p.numel() for p in m.parameters()):,} params")
            except Exception:
                print(f"  {name.capitalize()}: (param count unavailable)")

    
    print(f"\nResults saved to: {results_dir}")
    print("Training completed! 🎉")
    return all_results, all_histories
