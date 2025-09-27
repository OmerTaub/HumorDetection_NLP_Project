# import json
# import torch
# import sys
# import os
# from pathlib import Path
# from collections import defaultdict

# class CrossDatasetEvaluator:
#     """Framework for evaluating humor detection models across different datasets."""
    
#     def __init__(self, results_dir="results/cross_dataset"):
#         self.results_dir = Path(results_dir)
#         self.results_dir.mkdir(exist_ok=True, parents=True)
        
#         # Add path for imports
#         sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
#     def load_model(self, model_path):
#         """Load a trained model from checkpoint."""
#         from src.models.deep_models import LSTMHumorClassifier, CNNHumorClassifier, HumorSpecificCNN
#         from src.models.deep_models import SimpleTokenizer
        
#         checkpoint = torch.load(model_path, map_location='cpu')
#         config = checkpoint['config']
#         vocab_size = checkpoint['vocab_size']
        
#         # Create model based on type
#         if config['model_type'] == 'lstm':
#             model = LSTMHumorClassifier(
#                 vocab_size=vocab_size,
#                 embedding_dim=config['embedding_dim'],
#                 hidden_dim=config['hidden_dim'],
#                 dropout=config['dropout']
#             )
#         elif config['model_type'] == 'cnn':
#             model = CNNHumorClassifier(
#                 vocab_size=vocab_size,
#                 embedding_dim=config['embedding_dim'],
#                 num_filters=config['num_filters'],
#                 dropout=config['dropout']
#             )
#         elif config['model_type'] == 'humor_cnn':
#             model = HumorSpecificCNN(
#                 vocab_size=vocab_size,
#                 embedding_dim=config['embedding_dim'],
#                 num_filters=config['num_filters'],
#                 dropout=config['dropout']
#             )
#         else:
#             raise ValueError(f"Unknown model type: {config['model_type']}")
        
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()
        
#         return model, config
    
#     def create_synthetic_datasets(self):
#         """Create synthetic datasets with different humor characteristics for cross-dataset evaluation."""
        
#         # Dataset 1: Pun-heavy dataset
#         pun_dataset = {
#             'name': 'pun_heavy',
#             'texts': [
#                 "I told my wife she was drawing her eyebrows too high. She looked surprised.",
#                 "Time flies like an arrow. Fruit flies like a banana.",
#                 "I used to hate facial hair, but then it grew on me.",
#                 "What do you call a fake noodle? An impasta!",
#                 "I'm afraid for the calendar. Its days are numbered.",
#                 "The weather is fine today.",
#                 "Please submit the report by tomorrow.",
#                 "The meeting starts at 3 PM.",
#                 "Remember to lock the door.",
#                 "The store closes at 9 PM."
#             ],
#             'labels': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
#         }
        
#         # Dataset 2: Observational humor dataset
#         observational_dataset = {
#             'name': 'observational_humor',
#             'texts': [
#                 "I haven't slept for ten days, because that would be too long.",
#                 "I bought some batteries, but they weren't included.",
#                 "A day without sunshine is like night.",
#                 "I'm reading a book about anti-gravity. It's impossible to put down!",
#                 "Why don't scientists trust atoms? Because they make up everything!",
#                 "Water boils at 100 degrees Celsius.",
#                 "The capital of France is Paris.",
#                 "Regular exercise is important for health.",
#                 "Python is a programming language.",
#                 "The Earth orbits the Sun."
#             ],
#             'labels': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
#         }
        
#         # Dataset 3: Mixed formal/informal dataset
#         mixed_dataset = {
#             'name': 'mixed_formal_informal',
#             'texts': [
#                 "Why did the scarecrow win an award? He was outstanding in his field!",
#                 "What's the best thing about Switzerland? I don't know, but the flag is a big plus.",
#                 "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them.",
#                 "I wondered why the ball kept getting bigger. Then it hit me.",
#                 "Parallel lines have so much in common. It's a shame they'll never meet.",
#                 "The quarterly financial report shows steady growth.",
#                 "Please ensure all documentation is properly filed.",
#                 "The conference will be held in the main auditorium.",
#                 "All employees must complete the safety training.",
#                 "The system maintenance is scheduled for tonight."
#             ],
#             'labels': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
#         }
        
#         return [pun_dataset, observational_dataset, mixed_dataset]
    
#     def evaluate_model_on_dataset(self, model, config, dataset):
#         """Evaluate a model on a specific dataset."""
#         from src.models.deep_models import SimpleTokenizer
#         from src.evaluation.metrics import HumorEvaluationMetrics
        
#         # Note: In a real scenario, we would need the original tokenizer
#         # For this demo, we'll create a new one (which is not ideal but works for demonstration)
#         tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'])
#         tokenizer.build_vocab(dataset['texts'])  # This is a simplification
        
#         # Encode texts
#         max_length = config.get('max_seq_length', 64)
#         encoded_texts = [tokenizer.encode(text, max_length) for text in dataset['texts']]
        
#         # Get predictions
#         predictions = []
#         with torch.no_grad():
#             for encoded_text in encoded_texts:
#                 input_ids = torch.tensor([encoded_text], dtype=torch.long)
#                 logits = model(input_ids)
#                 pred = torch.argmax(logits, dim=1).item()
#                 predictions.append(pred)
        
#         # Calculate metrics
#         metrics = HumorEvaluationMetrics()
#         metrics.update(dataset['labels'], predictions)
        
#         return metrics.get_all_metrics()
    
#     def run_cross_dataset_evaluation(self, model_paths=None):
#         """Run cross-dataset evaluation for all models."""
#         if model_paths is None:
#             # Find all trained models
#             models_dir = Path("models/trained")
#             if models_dir.exists():
#                 model_paths = list(models_dir.glob("*.pth"))
#             else:
#                 print("No trained models found!")
#                 return
        
#         # Create synthetic datasets
#         datasets = self.create_synthetic_datasets()
        
#         # Results storage
#         results = {}
        
#         print("Cross-Dataset Evaluation")
#         print("=" * 40)
        
#         for model_path in model_paths:
#             model_name = model_path.stem
#             print(f"\nEvaluating model: {model_name}")
#             print("-" * (len(model_name) + 18))
            
#             try:
#                 # Load model
#                 model, config = self.load_model(model_path)
                
#                 # Evaluate on each dataset
#                 model_results = {}
#                 for dataset in datasets:
#                     dataset_name = dataset['name']
#                     print(f"  Testing on {dataset_name}...")
                    
#                     metrics = self.evaluate_model_on_dataset(model, config, dataset)
#                     model_results[dataset_name] = metrics
                    
#                     print(f"    Accuracy: {metrics['accuracy']:.4f}")
#                     print(f"    F1 Score: {metrics['f1_score']:.4f}")
                
#                 results[model_name] = model_results
                
#             except Exception as e:
#                 print(f"  Error evaluating {model_name}: {e}")
#                 import traceback
#                 traceback.print_exc()
        
#         # Save results
#         results_file = self.results_dir / "cross_dataset_evaluation.json"
#         with open(results_file, 'w') as f:
#             json.dump(results, f, indent=2)
        
#         print(f"\nCross-dataset evaluation results saved to: {results_file}")
        
#         # Generate summary report
#         self.generate_summary_report(results)
        
#         return results
    
#     def generate_summary_report(self, results):
#         """Generate a summary report of cross-dataset evaluation."""
        
#         # Calculate average performance across datasets
#         summary = {}
        
#         for model_name, model_results in results.items():
#             metrics_sums = defaultdict(float)
#             num_datasets = len(model_results)
            
#             for dataset_name, metrics in model_results.items():
#                 for metric_name, value in metrics.items():
#                     if isinstance(value, (int, float)):
#                         metrics_sums[metric_name] += value
            
#             # Calculate averages
#             summary[model_name] = {
#                 metric: total / num_datasets 
#                 for metric, total in metrics_sums.items()
#             }
        
#         # Print summary
#         print("\nCross-Dataset Performance Summary")
#         print("=" * 45)
        
#         for model_name, avg_metrics in summary.items():
#             print(f"\n{model_name}:")
#             print(f"  Average Accuracy: {avg_metrics.get('accuracy', 0):.4f}")
#             print(f"  Average F1 Score: {avg_metrics.get('f1_score', 0):.4f}")
#             print(f"  Average Precision: {avg_metrics.get('precision', 0):.4f}")
#             print(f"  Average Recall: {avg_metrics.get('recall', 0):.4f}")
        
#         # Find best performing model
#         if summary:
#             best_model = max(summary.items(), key=lambda x: x[1].get('f1_score', 0))
#             print(f"\nBest Overall Model: {best_model[0]}")
#             print(f"Average F1 Score: {best_model[1].get('f1_score', 0):.4f}")
        
#         # Save summary
#         summary_file = self.results_dir / "performance_summary.json"
#         with open(summary_file, 'w') as f:
#             json.dump(summary, f, indent=2)
        
#         print(f"Summary saved to: {summary_file}")
    
#     def analyze_model_specialization(self, results):
#         """Analyze which models perform better on which types of humor."""
        
#         print("\nModel Specialization Analysis")
#         print("=" * 35)
        
#         # For each dataset, find the best performing model
#         dataset_best = {}
        
#         # Get all dataset names
#         dataset_names = set()
#         for model_results in results.values():
#             dataset_names.update(model_results.keys())
        
#         for dataset_name in dataset_names:
#             best_f1 = 0
#             best_model = None
            
#             for model_name, model_results in results.items():
#                 if dataset_name in model_results:
#                     f1_score = model_results[dataset_name].get('f1_score', 0)
#                     if f1_score > best_f1:
#                         best_f1 = f1_score
#                         best_model = model_name
            
#             dataset_best[dataset_name] = {
#                 'best_model': best_model,
#                 'f1_score': best_f1
#             }
        
#         # Print specialization results
#         for dataset_name, best_info in dataset_best.items():
#             print(f"\n{dataset_name.replace('_', ' ').title()}:")
#             print(f"  Best Model: {best_info['best_model']}")
#             print(f"  F1 Score: {best_info['f1_score']:.4f}")
        
#         return dataset_best

# def create_additional_test_datasets():
#     """Create additional test datasets for more comprehensive evaluation."""
    
#     # Create directory for additional datasets
#     data_dir = Path("data/raw")
#     data_dir.mkdir(exist_ok=True, parents=True)
    
#     # Reddit-style jokes dataset
#     reddit_jokes = {
#         'texts': [
#             "I told my cat a joke about dogs. He didn't find it a-mew-sing.",
#             "Why don't eggs tell jokes? They'd crack each other up!",
#             "I'm terrified of elevators. I'll take steps to avoid them.",
#             "What do you call a bear with no teeth? A gummy bear!",
#             "I used to be a banker, but I lost interest.",
#             "Today is Monday and it's raining.",
#             "The grocery store has a sale on apples.",
#             "My commute to work takes 30 minutes.",
#             "The library is open until 8 PM today.",
#             "I need to do laundry this weekend."
#         ],
#         'labels': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
#     }
    
#     # Save as CSV
#     import csv
    
#     reddit_path = data_dir / "reddit_style_jokes.csv"
#     with open(reddit_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['text', 'label'])
#         for text, label in zip(reddit_jokes['texts'], reddit_jokes['labels']):
#             writer.writerow([text, label])
    
#     print(f"Created additional test dataset: {reddit_path}")
    
#     return reddit_path

# def main():
#     """Main function for cross-dataset evaluation."""
    
#     print("Setting up Cross-Dataset Evaluation Framework")
#     print("=" * 50)
    
#     # Create additional test datasets
#     create_additional_test_datasets()
    
#     # Initialize evaluator
#     evaluator = CrossDatasetEvaluator()
    
#     # Run cross-dataset evaluation
#     results = evaluator.run_cross_dataset_evaluation()
    
#     if results:
#         # Analyze model specialization
#         evaluator.analyze_model_specialization(results)
    
#     print("\nCross-dataset evaluation completed!")

# if __name__ == "__main__":
#     main()