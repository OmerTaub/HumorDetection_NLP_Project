#!/usr/bin/env python3
"""
Focused Experimental Framework for Proposed Humor Model Architecture
===================================================================

This framework provides systematic experiments specifically designed for the 
proposed humor detection model, focusing on its key architectural innovations:

1. Setup-Punchline Attention (SPA) Analysis
2. Incongruity Modeling Evaluation  
3. Setup-Punchline Segmentation Optimization
4. Component Ablation Studies
5. Transfer Learning with Backbone Freezing

Author: Advanced NLP Course Project
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from scipy import stats
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Project imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models import ModelFactory, SimpleTokenizer
from src.evaluation.metrics import HumorDataLoader, HumorEvaluationMetrics
from src.training.trainer import ThreeModelTrainer


@dataclass
class ProposedModelConfig:
    """Configuration specifically for proposed humor model experiments"""
    experiment_id: str
    experiment_category: str  # 'spa_analysis', 'incongruity_study', etc.
    
    # Core model parameters
    model_type: str = "proposed"  # 'baseline', 'transformer', or 'proposed'
    model_name: str = "huawei-noah/TinyBERT_General_4L_312D"
    freeze_backbone: bool = False
    dropout_rate: float = 0.2
    
    # SPA-specific parameters
    spa_heads: int = 1
    setup_ratio: float = 0.7
    
    # Training parameters
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 2e-5
    max_seq_length: int = 48
    
    # Experiment settings
    random_seed: int = 42
    dataset: str = "puns_pos_neg"
    
    # Analysis flags
    save_attention_weights: bool = False
    analyze_setup_punchline: bool = False
    
    def to_dict(self):
        return asdict(self)


class FocusedExperimentLogger:
    """Specialized logging for proposed model experiments"""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = experiment_dir / 'experiment.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Results tracking
        self.results = {}
        self.attention_analyses = {}
        self.component_analyses = {}
        
    def log_spa_analysis(self, config: ProposedModelConfig, attention_stats: Dict):
        """Log Setup-Punchline Attention analysis"""
        self.attention_analyses[config.experiment_id] = {
            'config': config.to_dict(),
            'attention_stats': attention_stats,
            'timestamp': datetime.now().isoformat()
        }
        
    def log_component_analysis(self, config: ProposedModelConfig, component_importance: Dict):
        """Log component ablation analysis"""
        self.component_analyses[config.experiment_id] = {
            'config': config.to_dict(),
            'component_importance': component_importance,
            'timestamp': datetime.now().isoformat()
        }
        
    def save_all_results(self):
        """Save all experimental results"""
        with open(self.experiment_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        with open(self.experiment_dir / 'attention_analyses.json', 'w') as f:
            json.dump(self.attention_analyses, f, indent=2)
        with open(self.experiment_dir / 'component_analyses.json', 'w') as f:
            json.dump(self.component_analyses, f, indent=2)


class ProposedModelExperimentRunner:
    """Focused experimental framework for proposed humor model"""
    
    def __init__(self, base_dir: str = "experiments/proposed_model_focus", device: str = None):
        self.base_dir = Path(base_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer = ThreeModelTrainer(self.device)
        
        print(f"🎭 Focused Proposed Humor Model Experiment Runner")
        print(f"📱 Device: {self.device}")
        print(f"📁 Base directory: {self.base_dir}")
        
    def set_random_seeds(self, seed: int = 42):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def run_single_experiment(self, config: ProposedModelConfig, logger: FocusedExperimentLogger) -> Dict:
        """Run a single proposed model experiment"""
        
        self.set_random_seeds(config.random_seed)
        logger.logger.info(f"Starting experiment: {config.experiment_id}")
        logger.logger.info(f"Category: {config.experiment_category}")
        
        start_time = time.time()
        
        try:
            # Prepare data based on model type
            if config.model_type == 'baseline':
                # For baseline model, prepare_data returns simple tokenized data directly
                train_data, val_data, test_data, vocab_size = self.trainer.prepare_data(
                    config.dataset, max_seq_length=config.max_seq_length
                )
            else:
                # For transformer/proposed models, prepare simple data first then convert to HF format
                self.trainer.prepare_data(config.dataset, max_seq_length=config.max_seq_length)
                train_data, val_data, test_data = self.trainer.prepare_hf_data(
                    model_name=config.model_name, max_seq_length=config.max_seq_length
                )
                vocab_size = 100  # Not used for transformer models
            
            # Build model kwargs based on model type
            if config.model_type == 'baseline':
                model_kwargs = {
                    'dropout_rate': config.dropout_rate,
                    'num_classes': 2
                    # Note: baseline model doesn't use freeze_backbone, spa_heads, or setup_ratio
                }
            elif config.model_type == 'transformer':
                model_kwargs = {
                    'freeze_backbone': config.freeze_backbone,
                    'dropout_rate': config.dropout_rate,
                    'num_classes': 2
                    # Note: transformer model doesn't use spa_heads or setup_ratio
                }
            else:  # proposed model
                model_kwargs = {
                    'freeze_backbone': config.freeze_backbone,
                    'dropout_rate': config.dropout_rate,
                    'spa_heads': config.spa_heads,
                    'setup_ratio': config.setup_ratio,
                    'num_classes': 2
                }
            
            # Train model with model_kwargs passed as **kwargs
            model, test_metrics, history = self.trainer.train_model(
                model_type=config.model_type,  # Use model_type from config
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                vocab_size=vocab_size,  # Use actual vocab size from data preparation
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                max_seq_length=config.max_seq_length,
                model_name=config.model_name,
                **model_kwargs  # Pass all model-specific parameters
            )
            
            training_time = time.time() - start_time
            
            # Store results
            result = {
                'config': config.to_dict(),
                'metrics': test_metrics,
                'training_time': training_time,
                'history': history,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.results[config.experiment_id] = result
            logger.logger.info(f"Completed: {config.experiment_id} - F1: {test_metrics.get('f1_score', 'N/A'):.4f}")
            
            return {
                'success': True,
                'metrics': test_metrics,
                'training_time': training_time,
                'model': model,
                'history': history
            }
            
        except Exception as e:
            logger.logger.error(f"Experiment {config.experiment_id} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }

    def experiment_1_spa_attention_analysis(self) -> Dict:
        """
        Experiment 1: Setup-Punchline Attention (SPA) Mechanism Analysis
        
        Investigate how different numbers of attention heads affect the model's
        ability to capture setup-punchline interactions.
        """
        print("\n🔍 Experiment 1: Setup-Punchline Attention Analysis")
        print("=" * 60)
        
        experiment_dir = self.base_dir / "exp1_spa_analysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = FocusedExperimentLogger(experiment_dir)
        
        # Test different SPA head configurations
        spa_head_configs = [
            (1, "single_head"),
            (2, "light"),
            # (3, "moderate"),
        ]
        
        results = {}
        
        for spa_heads, description in spa_head_configs:
            config = ProposedModelConfig(
                experiment_id=f"spa_heads_{spa_heads}_{description}",
                experiment_category="spa_analysis",
                spa_heads=spa_heads,
                setup_ratio=0.7,  # Keep other params constant
                epochs=1,  # Focused experiments with fewer epochs
                save_attention_weights=True
            )
            
            result = self.run_single_experiment(config, logger)
            results[config.experiment_id] = result
        
        logger.save_all_results()
        self._analyze_spa_results(logger.results, experiment_dir)
        
        return results
    
    def experiment_2_setup_ratio_optimization(self) -> Dict:
        """
        Experiment 2: Setup-Punchline Segmentation Optimization
        
        Find the optimal ratio for dividing text into setup and punchline segments.
        """
        print("\n📏 Experiment 2: Setup-Punchline Segmentation Optimization")
        print("=" * 65)
        
        experiment_dir = self.base_dir / "exp2_setup_ratio" / datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = FocusedExperimentLogger(experiment_dir)
        
        # Test different setup ratios
        setup_ratios = [0.5, 0.6, 0.7, 0.75, 0.8]
        
        results = {}
        
        for ratio in setup_ratios:
            # Use smaller batch size for memory-intensive ratios
            batch_size = 4 if ratio >= 0.75 else 8
            
            config = ProposedModelConfig(
                experiment_id=f"setup_ratio_{int(ratio*100)}",
                experiment_category="setup_ratio_optimization",
                setup_ratio=ratio,
                spa_heads=1,  # Keep SPA heads constant
                epochs=1,
                batch_size=batch_size,
                analyze_setup_punchline=True
            )
            
            result = self.run_single_experiment(config, logger)
            results[config.experiment_id] = result
        
        logger.save_all_results()
        self._analyze_setup_ratio_results(logger.results, experiment_dir)
        
        return results
    
    
    def experiment_4_backbone_freezing_study(self) -> Dict:
        """
        Experiment 4: Backbone Freezing Strategy Analysis
        
        Analyze the effect of freezing the BERT backbone on proposed model components.
        """
        print("\n🧊 Experiment 4: Backbone Freezing Strategy Study")
        print("=" * 50)
        
        experiment_dir = self.base_dir / "exp4_backbone_freezing" / datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = FocusedExperimentLogger(experiment_dir)
        
        configs = []
        
        # Test freezing with different learning rates
        for freeze_backbone in [True, False]:
            for lr in [1e-5]:
                config = ProposedModelConfig(
                    experiment_id=f"freeze_{freeze_backbone}_lr_{lr:.0e}",
                    experiment_category="backbone_freezing",
                    freeze_backbone=freeze_backbone,
                    learning_rate=lr,
                    epochs=1
                )
                configs.append(config)
        
        results = {}
        for config in tqdm(configs, desc="Backbone freezing experiments"):
            result = self.run_single_experiment(config, logger)
            results[config.experiment_id] = result
        
        logger.save_all_results()
        self._analyze_backbone_freezing_results(logger.results, experiment_dir)
        
        return results
    
    def experiment_5_architecture_comparison(self) -> Dict:
        """
        Experiment 5: Proposed Model vs Baselines
        
        Compare the proposed model against transformer baseline with controlled settings.
        """
        print("\n🏆 Experiment 5: Architecture Comparison Study")
        print("=" * 45)
        
        experiment_dir = self.base_dir / "exp5_architecture_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = FocusedExperimentLogger(experiment_dir)
        
        # Prepare data once
        self.trainer.prepare_data("200k_jokes", max_seq_length=64)
        hf_train, hf_val, hf_test = self.trainer.prepare_hf_data(
            model_name="bert-base-uncased", max_seq_length=64
        )
        
        results = {}
        
        # Test both architectures with same settings
        for model_type in ['baseline','transformer', 'proposed']:
            for freeze_backbone in [True, False]:
                experiment_id = f"{model_type}_freeze_{freeze_backbone}"
                
                try:
                    start_time = time.time()
                    
                    if model_type == 'transformer':
                        model_kwargs = {
                            'model_name': 'bert-base-uncased',
                            'freeze_backbone': freeze_backbone,
                            'dropout_rate': 0.2,
                            'num_classes': 2
                        }
                    else:  # proposed
                        model_kwargs = {
                            'model_name': 'bert-base-uncased',
                            'freeze_backbone': freeze_backbone,
                            'dropout_rate': 0.2,
                            'spa_heads': 2,
                            'setup_ratio': 0.7,
                            'num_classes': 2
                        }
                    
                    model, test_metrics, history = self.trainer.train_model(
                        model_type=model_type,
                        train_data=hf_train,
                        val_data=hf_val,
                        test_data=hf_test,
                        vocab_size=100,
                        epochs=2,
                        batch_size=16,
                        learning_rate=2e-5,
                        max_seq_length=64,
                        model_name='bert-base-uncased'
                    )
                    
                    training_time = time.time() - start_time
                    
                    results[experiment_id] = {
                        'success': True,
                        'model_type': model_type,
                        'metrics': test_metrics,
                        'training_time': training_time,
                        'history': history
                    }
                    
                    logger.logger.info(f"Completed {experiment_id} - F1: {test_metrics.get('f1_score', 'N/A'):.4f}")
                    
                except Exception as e:
                    logger.logger.error(f"Failed {experiment_id}: {str(e)}")
                    results[experiment_id] = {'success': False, 'error': str(e)}
        
        # Save results
        with open(experiment_dir / 'comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self._analyze_architecture_comparison(results, experiment_dir)
        
        return results
    
    def _analyze_spa_results(self, results: Dict, output_dir: Path):
        """Analyze SPA attention head experiments"""
        
        data = []
        for exp_id, result in results.items():
            if 'config' in result and 'metrics' in result:
                config = result['config']
                metrics = result['metrics']
                data.append({
                    'spa_heads': config['spa_heads'],
                    'f1_score': metrics.get('f1_score', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'training_time': result.get('training_time', 0)
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # F1 Score vs SPA Heads
            axes[0,0].bar(df['spa_heads'], df['f1_score'], color='steelblue', alpha=0.7, width=0.6)
            axes[0,0].set_xlabel('Number of SPA Heads')
            axes[0,0].set_ylabel('F1 Score')
            axes[0,0].set_title('F1 Score vs SPA Attention Heads')
            axes[0,0].grid(True, alpha=0.3, axis='y')
            
            # Training Time vs SPA Heads
            axes[0,1].bar(df['spa_heads'], df['training_time'], color='coral', alpha=0.7, width=0.6)
            axes[0,1].set_xlabel('Number of SPA Heads')
            axes[0,1].set_ylabel('Training Time (s)')
            axes[0,1].set_title('Training Time vs SPA Heads')
            axes[0,1].grid(True, alpha=0.3, axis='y')
            
            # All metrics comparison
            metrics_cols = ['f1_score', 'accuracy', 'precision', 'recall']
            x = np.arange(len(df['spa_heads']))
            width = 0.2
            colors = ['steelblue', 'lightcoral', 'lightgreen', 'plum']
            
            for i, metric in enumerate(metrics_cols):
                axes[1,0].bar(x + i*width, df[metric], width, 
                            label=metric.replace('_', ' ').title(), 
                            color=colors[i], alpha=0.7)
            
            axes[1,0].set_xlabel('Number of SPA Heads')
            axes[1,0].set_ylabel('Score')
            axes[1,0].set_title('All Metrics vs SPA Heads')
            axes[1,0].set_xticks(x + width * 1.5)
            axes[1,0].set_xticklabels(df['spa_heads'])
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3, axis='y')
            
            # Summary table
            summary_text = df.to_string(index=False, float_format='%.4f')
            axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                          fontsize=9, verticalalignment='top', fontfamily='monospace')
            axes[1,1].set_title('Summary Statistics')
            axes[1,1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'spa_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save data
            df.to_csv(output_dir / 'spa_results.csv', index=False)
            
            # Find optimal
            best_idx = df['f1_score'].idxmax()
            best_heads = df.loc[best_idx, 'spa_heads']
            best_f1 = df.loc[best_idx, 'f1_score']
            
            with open(output_dir / 'spa_analysis_summary.txt', 'w') as f:
                f.write("Setup-Punchline Attention Analysis Summary\n")
                f.write("=" * 45 + "\n\n")
                f.write(f"Optimal SPA Heads: {best_heads}\n")
                f.write(f"Best F1 Score: {best_f1:.4f}\n\n")
                f.write("All Results:\n")
                f.write(df.to_string(index=False, float_format='%.4f'))
    
    def _analyze_setup_ratio_results(self, results: Dict, output_dir: Path):
        """Analyze setup-punchline segmentation results"""
        
        data = []
        for exp_id, result in results.items():
            if 'config' in result and 'metrics' in result:
                config = result['config']
                metrics = result['metrics']
                data.append({
                    'setup_ratio': config['setup_ratio'],
                    'f1_score': metrics.get('f1_score', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'training_time': result.get('training_time', 0)
                })
        
        df = pd.DataFrame(data).sort_values('setup_ratio')
        if not df.empty:
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # F1 Score vs Setup Ratio
            bars = axes[0].bar(df['setup_ratio'], df['f1_score'], color='forestgreen', alpha=0.7, width=0.05)
            axes[0].set_xlabel('Setup Ratio')
            axes[0].set_ylabel('F1 Score')
            axes[0].set_title('F1 Score vs Setup-Punchline Ratio')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Highlight best performing ratio
            best_idx = df['f1_score'].idxmax()
            best_ratio = df.loc[best_idx, 'setup_ratio']
            best_f1 = df.loc[best_idx, 'f1_score']
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('darkgreen')
            bars[best_idx].set_linewidth(2)
            
            axes[0].annotate(f'Best: {best_ratio:.1f} ({best_f1:.4f})',
                           xy=(best_ratio, best_f1), xytext=(10, 10),
                           textcoords='offset points', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Accuracy vs Setup Ratio
            bars_acc = axes[1].bar(df['setup_ratio'], df['accuracy'], color='steelblue', alpha=0.7, width=0.05)
            axes[1].set_xlabel('Setup Ratio')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy vs Setup-Punchline Ratio')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Highlight best accuracy
            best_acc_idx = df['accuracy'].idxmax()
            bars_acc[best_acc_idx].set_color('lightblue')
            bars_acc[best_acc_idx].set_edgecolor('darkblue')
            bars_acc[best_acc_idx].set_linewidth(2)
            axes[1].axvline(x=best_ratio, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'setup_ratio_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save results
            df.to_csv(output_dir / 'setup_ratio_results.csv', index=False)
            
            with open(output_dir / 'setup_ratio_summary.txt', 'w') as f:
                f.write("Setup-Punchline Segmentation Analysis Summary\n")
                f.write("=" * 48 + "\n\n")
                f.write(f"Optimal Setup Ratio: {best_ratio}\n")
                f.write(f"Best F1 Score: {best_f1:.4f}\n\n")
                f.write("All Results:\n")
                f.write(df.to_string(index=False, float_format='%.4f'))
    
    def _analyze_incongruity_results(self, results: Dict, output_dir: Path):
        """Analyze incongruity modeling results"""
        
        data = []
        for exp_id, result in results.items():
            if 'config' in result and 'metrics' in result:
                config = result['config']
                metrics = result['metrics']
                data.append({
                    'experiment': exp_id,
                    'dropout_rate': config['dropout_rate'],
                    'f1_score': metrics.get('f1_score', 0),
                    'accuracy': metrics.get('accuracy', 0)
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            # Simple bar plot
            plt.figure(figsize=(12, 6))
            x = range(len(df))
            plt.bar(x, df['f1_score'], alpha=0.7, color='skyblue', edgecolor='navy')
            plt.xlabel('Experiment Configuration')
            plt.ylabel('F1 Score')
            plt.title('Incongruity Modeling Analysis')
            plt.xticks(x, df['experiment'], rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'incongruity_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            df.to_csv(output_dir / 'incongruity_results.csv', index=False)
    
    def _analyze_backbone_freezing_results(self, results: Dict, output_dir: Path):
        """Analyze backbone freezing results"""
        
        data = []
        for exp_id, result in results.items():
            if 'config' in result and 'metrics' in result:
                config = result['config']
                metrics = result['metrics']
                data.append({
                    'freeze_backbone': config['freeze_backbone'],
                    'learning_rate': config['learning_rate'],
                    'f1_score': metrics.get('f1_score', 0),
                    'training_time': result.get('training_time', 0)
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            # Group by freeze_backbone
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # F1 Score comparison
            frozen_data = df[df['freeze_backbone'] == True]
            unfrozen_data = df[df['freeze_backbone'] == False]
            
            # Create grouped bar chart
            x = np.arange(len(df['learning_rate'].unique()))
            width = 0.35
            
            if not frozen_data.empty:
                axes[0].bar(x - width/2, frozen_data['f1_score'], width, 
                           label='Frozen Backbone', color='coral', alpha=0.7)
            if not unfrozen_data.empty:
                axes[0].bar(x + width/2, unfrozen_data['f1_score'], width, 
                           label='Unfrozen Backbone', color='steelblue', alpha=0.7)
            
            axes[0].set_xlabel('Learning Rate')
            axes[0].set_ylabel('F1 Score')
            axes[0].set_title('F1 Score vs Learning Rate')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f'{lr:.0e}' for lr in df['learning_rate'].unique()])
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Training time comparison
            if not frozen_data.empty:
                axes[1].bar(x - width/2, frozen_data['training_time'], width, 
                           label='Frozen Backbone', color='coral', alpha=0.7)
            if not unfrozen_data.empty:
                axes[1].bar(x + width/2, unfrozen_data['training_time'], width, 
                           label='Unfrozen Backbone', color='steelblue', alpha=0.7)
            
            axes[1].set_xlabel('Learning Rate')
            axes[1].set_ylabel('Training Time (s)')
            axes[1].set_title('Training Time vs Learning Rate')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f'{lr:.0e}' for lr in df['learning_rate'].unique()])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'backbone_freezing_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            df.to_csv(output_dir / 'backbone_freezing_results.csv', index=False)
    
    def _analyze_architecture_comparison(self, results: Dict, output_dir: Path):
        """Analyze architecture comparison results"""
        
        data = []
        for exp_id, result in results.items():
            if result.get('success') and 'metrics' in result:
                parts = exp_id.split('_')
                model_type = parts[0]
                freeze_status = parts[2] == 'True'
                
                metrics = result['metrics']
                data.append({
                    'model_type': model_type,
                    'freeze_backbone': freeze_status,
                    'f1_score': metrics.get('f1_score', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'training_time': result.get('training_time', 0)
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            # Create comparison visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # F1 Score comparison
            df_pivot = df.pivot(index='model_type', columns='freeze_backbone', values='f1_score')
            df_pivot.plot(kind='bar', ax=axes[0], width=0.8)
            axes[0].set_title('F1 Score Comparison')
            axes[0].set_ylabel('F1 Score')
            axes[0].legend(['Unfrozen', 'Frozen'])
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
            
            # Accuracy comparison
            df_pivot_acc = df.pivot(index='model_type', columns='freeze_backbone', values='accuracy')
            df_pivot_acc.plot(kind='bar', ax=axes[1], width=0.8)
            axes[1].set_title('Accuracy Comparison')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend(['Unfrozen', 'Frozen'])
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
            
            # Training time comparison
            df_pivot_time = df.pivot(index='model_type', columns='freeze_backbone', values='training_time')
            df_pivot_time.plot(kind='bar', ax=axes[2], width=0.8)
            axes[2].set_title('Training Time Comparison')
            axes[2].set_ylabel('Training Time (s)')
            axes[2].legend(['Unfrozen', 'Frozen'])
            axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            df.to_csv(output_dir / 'architecture_comparison.csv', index=False)
            
            # Find best configuration
            best_idx = df['f1_score'].idxmax()
            best_config = df.loc[best_idx]
            
            with open(output_dir / 'comparison_summary.txt', 'w') as f:
                f.write("Architecture Comparison Summary\n")
                f.write("=" * 35 + "\n\n")
                f.write(f"Best Configuration:\n")
                f.write(f"  Model: {best_config['model_type']}\n")
                f.write(f"  Freeze Backbone: {best_config['freeze_backbone']}\n")
                f.write(f"  F1 Score: {best_config['f1_score']:.4f}\n")
                f.write(f"  Accuracy: {best_config['accuracy']:.4f}\n\n")
                f.write("All Results:\n")
                f.write(df.to_string(index=False, float_format='%.4f'))


def run_focused_experiments():
    """Run all focused experiments for the proposed humor model"""
    
    print("🎭 Focused Proposed Humor Model Experimental Suite")
    print("=" * 65)
    
    runner = ProposedModelExperimentRunner()
    
    all_results = {}
    
    # Run experiments in sequence
    print("\n📋 Running 5 Focused Experiments...")
    
    # Experiment 1: SPA Analysis
    try:
        spa_results = runner.experiment_1_spa_attention_analysis()
        all_results['spa_analysis'] = spa_results
        print("✅ Experiment 1 (SPA Analysis) completed")
    except Exception as e:
        print(f"❌ Experiment 1 failed: {e}")
    
    # Experiment 2: Setup Ratio Optimization  
    try:
        ratio_results = runner.experiment_2_setup_ratio_optimization()
        all_results['setup_ratio'] = ratio_results
        print("✅ Experiment 2 (Setup Ratio) completed")
    except Exception as e:
        print(f"❌ Experiment 2 failed: {e}")
    
    
    # Experiment 4: Backbone Freezing
    try:
        freezing_results = runner.experiment_4_backbone_freezing_study()
        all_results['backbone_freezing'] = freezing_results
        print("✅ Experiment 4 (Backbone Freezing) completed")
    except Exception as e:
        print(f"❌ Experiment 4 failed: {e}")
    
    # Experiment 5: Architecture Comparison
    try:
        comparison_results = runner.experiment_5_architecture_comparison()
        all_results['architecture_comparison'] = comparison_results
        print("✅ Experiment 5 (Architecture Comparison) completed")
    except Exception as e:
        print(f"❌ Experiment 5 failed: {e}")
    
    # Generate final report
    final_dir = runner.base_dir / "final_summary" / datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir.mkdir(parents=True, exist_ok=True)
    
    generate_final_report(all_results, final_dir)
    
    print(f"\n🎉 All experiments completed!")
    print(f"📁 Results saved in: {runner.base_dir}")
    print(f"📋 Final summary: {final_dir}")
    
    return all_results


def generate_final_report(all_results: Dict, output_dir: Path):
    """Generate comprehensive final report"""
    
    report_file = output_dir / 'final_experimental_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Focused Proposed Humor Model Experimental Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of systematic experiments designed to evaluate ")
        f.write("the key architectural components of the proposed humor detection model.\n\n")
        
        f.write("## Model Architecture Overview\n\n")
        f.write("The proposed model consists of:\n")
        f.write("1. **BERT Backbone**: Pre-trained transformer for semantic understanding\n")
        f.write("2. **Setup-Punchline Attention (SPA)**: Cross-attention mechanism (Q=punchline, K/V=setup)\n")
        f.write("3. **Incongruity Modeling**: Feature comparison between setup and punchline segments\n")
        f.write("4. **Multi-component Fusion**: Combines pooled features, SPA output, and incongruity features\n\n")
        
        f.write("## Experimental Results\n\n")
        
        experiment_names = {
            'spa_analysis': 'Setup-Punchline Attention Analysis',
            'setup_ratio': 'Setup-Punchline Segmentation Optimization',
            'incongruity': 'Incongruity Modeling Ablation',
            'backbone_freezing': 'Backbone Freezing Strategy',
            'architecture_comparison': 'Architecture Comparison Study'
        }
        
        for exp_key, exp_name in experiment_names.items():
            f.write(f"### {exp_name}\n\n")
            if exp_key in all_results:
                results = all_results[exp_key]
                if results:
                    # Extract best result
                    best_result = None
                    best_f1 = 0
                    for result in results.values():
                        if result.get('success') and 'metrics' in result:
                            f1 = result['metrics'].get('f1_score', 0)
                            if f1 > best_f1:
                                best_f1 = f1
                                best_result = result
                    
                    if best_result:
                        f.write(f"- **Best F1 Score**: {best_f1:.4f}\n")
                        f.write(f"- **Experiments Conducted**: {len(results)}\n")
                    else:
                        f.write("- No successful experiments\n")
                else:
                    f.write("- No results available\n")
            else:
                f.write("- Experiment not completed\n")
            f.write("\n")
        
        f.write("## Key Findings\n\n")
        f.write("1. **SPA Mechanism**: [Analysis of attention head configurations]\n")
        f.write("2. **Optimal Setup Ratio**: [Best segmentation ratio for setup-punchline split]\n")
        f.write("3. **Incongruity Features**: [Contribution of incongruity modeling]\n")
        f.write("4. **Transfer Learning**: [Backbone freezing strategy effectiveness]\n")
        f.write("5. **Architecture Benefits**: [Proposed model vs transformer baseline]\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the experimental results:\n\n")
        f.write("1. **Model Configuration**: Use optimal parameters identified in experiments\n")
        f.write("2. **Training Strategy**: Apply best backbone freezing approach\n")
        f.write("3. **Architecture**: Proposed model shows [improvement/similarity] over baseline\n\n")
        
        f.write("## Future Work\n\n")
        f.write("1. **Attention Visualization**: Analyze SPA attention patterns qualitatively\n")
        f.write("2. **Error Analysis**: Investigate failure cases and model limitations\n")
        f.write("3. **Cross-Dataset Evaluation**: Test generalization across different humor datasets\n")
    
    # Save summary statistics
    with open(output_dir / 'experiment_summary.json', 'w') as f:
        summary = {}
        for exp_key, results in all_results.items():
            if results:
                summary[exp_key] = {
                    'total_experiments': len(results),
                    'successful_experiments': sum(1 for r in results.values() if r.get('success')),
                    'best_f1': max((r['metrics'].get('f1_score', 0) for r in results.values() 
                                  if r.get('success') and 'metrics' in r), default=0)
                }
        json.dump(summary, f, indent=2)
    
    print(f"📄 Final report generated: {report_file}")


if __name__ == "__main__":
    # Run focused experiments
    results = run_focused_experiments()
