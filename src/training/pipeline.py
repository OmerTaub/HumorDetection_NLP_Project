# import json
# import os
# import time
# from datetime import datetime
# from pathlib import Path
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import sys
# import tqdm

# class TrainingLogger:
#     """Logger for training experiments."""
    
#     def __init__(self, log_dir="logs", experiment_name=None):
#         self.log_dir = Path(log_dir)
#         self.log_dir.mkdir(exist_ok=True)
        
#         if experiment_name is None:
#             experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
#         self.experiment_name = experiment_name
#         self.log_file = self.log_dir / f"{experiment_name}.log"
#         self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        
#         self.metrics_history = {
#             'train_loss': [],
#             'train_acc': [],
#             'val_loss': [],
#             'val_acc': [],
#             'epochs': []
#         }
        
#         self.log(f"Starting experiment: {experiment_name}")
#         self.log(f"Log file: {self.log_file}")
#         self.log(f"Metrics file: {self.metrics_file}")
    
#     def log(self, message):
#         """Log a message with timestamp."""
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         log_entry = f"[{timestamp}] {message}"
        
#         print(log_entry)
        
#         with open(self.log_file, 'a', encoding='utf-8') as f:
#             f.write(log_entry + '\n')
    
#     def log_epoch(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
#         """Log epoch results."""
#         self.metrics_history['epochs'].append(epoch)
#         self.metrics_history['train_loss'].append(train_loss)
#         self.metrics_history['train_acc'].append(train_acc)
        
#         message = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}"
        
#         if val_loss is not None and val_acc is not None:
#             self.metrics_history['val_loss'].append(val_loss)
#             self.metrics_history['val_acc'].append(val_acc)
#             message += f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        
#         self.log(message)
        
#         # Save metrics to file
#         with open(self.metrics_file, 'w') as f:
#             json.dump(self.metrics_history, f, indent=2)
    
#     def log_config(self, config):
#         """Log experiment configuration."""
#         self.log("Experiment Configuration:")
#         for key, value in config.items():
#             self.log(f"  {key}: {value}")
    
#     def log_results(self, results):
#         """Log final results."""
#         self.log("Final Results:")
#         for key, value in results.items():
#             self.log(f"  {key}: {value}")

# class HumorTrainingPipeline:
#     """Comprehensive training pipeline for humor detection models."""
    
#     def __init__(self, config=None):
#         default_config = self.get_default_config()
#         if config:
#             default_config.update(config)
#         self.config = default_config
#         self.logger = TrainingLogger(experiment_name=self.config.get('experiment_name'))
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         self.logger.log(f"Using device: {self.device}")
#         self.logger.log_config(self.config)
    
#     def get_default_config(self):
#         """Get default training configuration."""
#         return {
#             'experiment_name': f"humor_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#             'model_type': 'humor_cnn',  # 'lstm', 'cnn', 'humor_cnn'
#             'vocab_size': 500,
#             'embedding_dim': 64,
#             'hidden_dim': 64,
#             'num_filters': 64,
#             'dropout': 0.5,
#             'learning_rate': 0.001,
#             'batch_size': 16,
#             'epochs': 10,
#             'early_stopping_patience': 1000,
#             'save_best_model': True,
#             'max_seq_length': 64
#         }
    
#     def create_model(self, vocab_size):
#         """Create model based on configuration."""
#         sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
#         from src.models.deep_models import LSTMHumorClassifier, CNNHumorClassifier, HumorSpecificCNN
        
#         model_type = self.config['model_type']
        
#         if model_type == 'lstm':
#             model = LSTMHumorClassifier(
#                 vocab_size=vocab_size,
#                 embedding_dim=self.config['embedding_dim'],
#                 hidden_dim=self.config['hidden_dim'],
#                 dropout=self.config['dropout']
#             )
#         elif model_type == 'cnn':
#             model = CNNHumorClassifier(
#                 vocab_size=vocab_size,
#                 embedding_dim=self.config['embedding_dim'],
#                 num_filters=self.config['num_filters'],
#                 dropout=self.config['dropout']
#             )
#         elif model_type == 'humor_cnn':
#             model = HumorSpecificCNN(
#                 vocab_size=vocab_size,
#                 embedding_dim=self.config['embedding_dim'],
#                 num_filters=self.config['num_filters'],
#                 dropout=self.config['dropout']
#             )
#         else:
#             raise ValueError(f"Unknown model type: {model_type}")
        
#         self.logger.log(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
#         return model
    
#     def prepare_data(self, train_texts, train_labels, val_texts=None, val_labels=None, test_texts=None, test_labels=None):
#         """Prepare data for training."""
#         from src.models.deep_models import SimpleTokenizer
        
#         # Create tokenizer
#         self.tokenizer = SimpleTokenizer(vocab_size=self.config['vocab_size'])
#         # IMPORTANT: Only use training data for vocabulary building to prevent data contamination
#         # Do not include val_texts or test_texts in vocabulary building
#         self.tokenizer.build_vocab(train_texts)
#         vocab_size = len(self.tokenizer.word_to_idx)
        
#         self.logger.log(f"Built vocabulary with {vocab_size} words")
        
#         # Encode data
#         max_length = self.config['max_seq_length']
        
#         train_data = [(self.tokenizer.encode(text, max_length), label) 
#                      for text, label in zip(train_texts, train_labels)]
        
#         val_data = None
#         if val_texts and val_labels:
#             val_data = [(self.tokenizer.encode(text, max_length), label) 
#                        for text, label in zip(val_texts, val_labels)]
        
#         test_data = None
#         if test_texts and test_labels:
#             test_data = [(self.tokenizer.encode(text, max_length), label) 
#                         for text, label in zip(test_texts, test_labels)]
        
#         return train_data, val_data, test_data, vocab_size
    
#     def train_epoch(self, model, train_data, optimizer, criterion):
#         """Train for one epoch."""
#         model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
#         batch_size = self.config['batch_size']
        
#         for i in tqdm.tqdm(range(0, len(train_data), batch_size)):
#             batch = train_data[i:i + batch_size]
            
#             texts = [item[0] for item in batch]
#             labels = [item[1] for item in batch]
            
#             input_ids = torch.tensor(texts, dtype=torch.long).to(self.device)
#             target_labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            
#             optimizer.zero_grad()
#             logits = model(input_ids)
#             loss = criterion(logits, target_labels)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             predictions = torch.argmax(logits, dim=1)
#             correct += (predictions == target_labels).sum().item()
#             total += target_labels.size(0)
        
#         avg_loss = total_loss / (len(train_data) // batch_size + 1)
#         accuracy = correct / total
        
#         return avg_loss, accuracy
    
#     def evaluate_model(self, model, eval_data, criterion):
#         """Evaluate model on validation/test data."""
#         model.eval()
#         total_loss = 0
#         correct = 0
#         total = 0
#         batch_size = self.config['batch_size']
        
#         with torch.no_grad():
#             for i in range(0, len(eval_data), batch_size):
#                 batch = eval_data[i:i + batch_size]
                
#                 texts = [item[0] for item in batch]
#                 labels = [item[1] for item in batch]
                
#                 input_ids = torch.tensor(texts, dtype=torch.long).to(self.device)
#                 target_labels = torch.tensor(labels, dtype=torch.long).to(self.device)
                
#                 logits = model(input_ids)
#                 loss = criterion(logits, target_labels)
                
#                 total_loss += loss.item()
#                 predictions = torch.argmax(logits, dim=1)
#                 correct += (predictions == target_labels).sum().item()
#                 total += target_labels.size(0)
        
#         avg_loss = total_loss / (len(eval_data) // batch_size + 1)
#         accuracy = correct / total
        
#         return avg_loss, accuracy
    
#     def get_predictions(self, model, eval_data):
#         """Get model predictions."""
#         model.eval()
#         all_predictions = []
#         all_labels = []
#         batch_size = self.config['batch_size']
        
#         with torch.no_grad():
#             for i in range(0, len(eval_data), batch_size):
#                 batch = eval_data[i:i + batch_size]
                
#                 texts = [item[0] for item in batch]
#                 labels = [item[1] for item in batch]
                
#                 input_ids = torch.tensor(texts, dtype=torch.long).to(self.device)
                
#                 logits = model(input_ids)
#                 predictions = torch.argmax(logits, dim=1)
                
#                 all_predictions.extend(predictions.cpu().numpy())
#                 all_labels.extend(labels)
        
#         return all_predictions, all_labels
    
#     def train(self, train_texts, train_labels, val_texts=None, val_labels=None, test_texts=None, test_labels=None):
#         """Main training loop."""
#         start_time = time.time()
        
#         # Prepare data
#         train_data, val_data, test_data, vocab_size = self.prepare_data(
#             train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
#         )
        
#         # Create model
#         model = self.create_model(vocab_size)
#         model.to(self.device)
        
#         # Setup training
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
#         # Training loop
#         best_val_acc = 0
#         patience_counter = 0
#         best_model_state = None
        
#         for epoch in tqdm.tqdm(range(self.config['epochs'])):
#             # Train
#             train_loss, train_acc = self.train_epoch(model, train_data, optimizer, criterion)
            
#             # Validate
#             val_loss, val_acc = None, None
#             if val_data:
#                 val_loss, val_acc = self.evaluate_model(model, val_data, criterion)
                
#                 # Early stopping check
#                 if val_acc > best_val_acc:
#                     best_val_acc = val_acc
#                     patience_counter = 0
#                     if self.config['save_best_model']:
#                         best_model_state = model.state_dict().copy()
#                 else:
#                     patience_counter += 1
                
#                 if patience_counter >= self.config['early_stopping_patience']:
#                     self.logger.log(f"Early stopping at epoch {epoch + 1}")
#                     break
            
#             # Log epoch results
#             self.logger.log_epoch(epoch + 1, train_loss, train_acc, val_loss, val_acc)
        
#         # Load best model if available
#         if best_model_state:
#             model.load_state_dict(best_model_state)
#             self.logger.log(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
        
#         # Final evaluation
#         results = {}
#         if test_data:
#             test_predictions, test_labels = self.get_predictions(model, test_data)
            
#             # Calculate detailed metrics
#             sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
#             from src.evaluation.metrics import HumorEvaluationMetrics
            
#             metrics = HumorEvaluationMetrics()
#             metrics.update(test_labels, test_predictions)
#             results = metrics.get_all_metrics()
            
#             self.logger.log_results(results)
        
#         # Save model
#         model_dir = Path("models") / "trained"
#         model_dir.mkdir(exist_ok=True, parents=True)
        
#         model_path = model_dir / f"{self.config['experiment_name']}.pth"
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'config': self.config,
#             'results': results,
#             'vocab_size': vocab_size
#         }, model_path)
        
#         training_time = time.time() - start_time
#         self.logger.log(f"Training completed in {training_time:.2f} seconds")
#         self.logger.log(f"Model saved to: {model_path}")
        
#         return model, results, self.logger.metrics_history

# def run_comprehensive_experiments():
#     """Run comprehensive experiments with different model configurations."""
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
#     from src.evaluation.metrics import HumorDataLoader
    
#     print("Running Comprehensive Humor Detection Experiments")
#     print("=" * 55)
    
#     # Load data
#     loader = HumorDataLoader()
#     splits = loader.load_all_splits("extended_sample")
#     train_texts, train_labels = splits['train']
#     val_texts, val_labels = splits['val']
#     test_texts, test_labels = splits['test']
    
#     # Experiment configurations
#     experiments = [
#         {
#             'experiment_name': 'cnn_baseline',
#             'model_type': 'lstm',
#             'embedding_dim': 64,
#             'hidden_dim': 128,
#             'epochs': 15
#         },
#         {
#             'experiment_name': 'cnn_baseline',
#             'model_type': 'cnn',
#             'embedding_dim': 100,
#             'num_filters': 100,
#             'epochs': 15
#         },
#         {
#             'experiment_name': 'humor_specific_cnn',
#             'model_type': 'humor_cnn',
#             'embedding_dim': 100,
#             'num_filters': 100,
#             'epochs': 15
#         }
#     ]
    
#     all_results = {}
    
#     for exp_config in experiments:
#         print(f"\nRunning experiment: {exp_config['experiment_name']}")
#         print("-" * 40)
        
#         pipeline = HumorTrainingPipeline(exp_config)
#         model, results, history = pipeline.train(
#             train_texts, train_labels,
#             val_texts, val_labels,
#             test_texts, test_labels
#         )
        
#         all_results[exp_config['experiment_name']] = results
    
#     # Save comprehensive results
#     results_file = Path("results") / "comprehensive_experiments.json"
#     results_file.parent.mkdir(exist_ok=True)
    
#     with open(results_file, 'w') as f:
#         json.dump(all_results, f, indent=2)
    
#     print(f"\nAll results saved to: {results_file}")
    
#     # Print summary
#     print("\nExperiment Summary:")
#     print("=" * 30)
#     for exp_name, results in all_results.items():
#         print(f"{exp_name}:")
#         print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
#         print(f"  F1 Score: {results.get('f1_score', 0):.4f}")
#         print(f"  Precision: {results.get('precision', 0):.4f}")
#         print(f"  Recall: {results.get('recall', 0):.4f}")

# if __name__ == "__main__":
#     run_comprehensive_experiments()