import csv
import json
from pathlib import Path

class HumorEvaluationMetrics:
    """Evaluation metrics for humor detection models."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
    
    def update(self, y_true, y_pred):
        """Update metrics with new predictions."""
        if not isinstance(y_true, list):
            y_true = [y_true]
        if not isinstance(y_pred, list):
            y_pred = [y_pred]
            
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label == 1 and pred_label == 1:
                self.true_positives += 1
            elif true_label == 0 and pred_label == 1:
                self.false_positives += 1
            elif true_label == 0 and pred_label == 0:
                self.true_negatives += 1
            elif true_label == 1 and pred_label == 0:
                self.false_negatives += 1
    
    def accuracy(self):
        """Calculate accuracy."""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    def precision(self):
        """Calculate precision."""
        if (self.true_positives + self.false_positives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    def recall(self):
        """Calculate recall."""
        if (self.true_positives + self.false_negatives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    def f1_score(self):
        """Calculate F1 score."""
        prec = self.precision()
        rec = self.recall()
        if (prec + rec) == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)
    
    def get_all_metrics(self):
        """Get all metrics as a dictionary."""
        return {
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1_score(),
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives
        }
    
    def print_metrics(self, title="Evaluation Results"):
        """Print all metrics in a formatted way."""
        metrics = self.get_all_metrics()
        
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 0      1")
        print(f"Actual    0    {metrics['true_negatives']:4d}   {metrics['false_positives']:4d}")
        print(f"          1    {metrics['false_negatives']:4d}   {metrics['true_positives']:4d}")
    
    def save_metrics(self, filepath, additional_info=None):
        """Save metrics to a JSON file."""
        metrics = self.get_all_metrics()
        if additional_info:
            metrics.update(additional_info)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")

class HumorDataLoader:
    """Data loader for humor detection datasets."""
    
    def __init__(self, data_dir="data/processed"):
        self.data_dir = Path(data_dir)
    
    def load_split(self, dataset_name, split):
        """Load a specific split (train/val/test) of a dataset."""
        filepath = self.data_dir / f"{dataset_name}_{split}.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        texts = []
        labels = []
        
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                texts.append(row['text'])
                labels.append(int(row['label']))
        
        print(f"Loaded {len(texts)} examples from {filepath}")
        return texts, labels
    
    def load_all_splits(self, dataset_name):
        """Load all splits of a dataset."""
        splits = {}
        for split in ['train', 'val', 'test']:
            try:
                texts, labels = self.load_split(dataset_name, split)
                splits[split] = (texts, labels)
            except FileNotFoundError:
                print(f"Warning: {split} split not found for {dataset_name}")
        
        return splits

def evaluate_predictions(y_true, y_pred, title="Model Evaluation"):
    """Convenient function to evaluate predictions and print results."""
    metrics = HumorEvaluationMetrics()
    metrics.update(y_true, y_pred)
    metrics.print_metrics(title)
    return metrics.get_all_metrics()

# Example usage and testing
def test_evaluation_metrics():
    """Test the evaluation metrics with sample data."""
    print("Testing Evaluation Metrics")
    print("-" * 30)
    
    # Perfect predictions
    y_true = [1, 1, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0, 1, 0]
    
    metrics = HumorEvaluationMetrics()
    metrics.update(y_true, y_pred)
    metrics.print_metrics("Perfect Predictions")
    
    # Imperfect predictions
    y_true = [1, 1, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0, 0, 0]
    
    metrics.reset()
    metrics.update(y_true, y_pred)
    metrics.print_metrics("Imperfect Predictions")
    
    # Test data loader
    print("\nTesting Data Loader")
    print("-" * 20)
    
    try:
        loader = HumorDataLoader(data_dir=f'/home/omertaub/projects/nlp_proj_new/data/processed')
        splits = loader.load_all_splits("extended_sample")
        
        for split_name, (texts, labels) in splits.items():
            humor_count = sum(labels)
            total_count = len(labels)
            print(f"{split_name}: {total_count} examples ({humor_count} humor, {total_count - humor_count} non-humor)")
    
    except Exception as e:
        print(f"Data loader test failed: {e}")

if __name__ == "__main__":
    test_evaluation_metrics()