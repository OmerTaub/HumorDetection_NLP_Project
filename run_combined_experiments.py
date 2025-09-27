#!/usr/bin/env python3
"""
Combined Dataset Experiment Runner
=================================

A YAML-configuration-driven experiment runner designed for 
running humor experiments on the combined datasets (200k_jokes + puns).

Usage:
    python run_combined_experiments.py --experiment spa_analysis
    python run_combined_experiments.py --experiment all
    python run_combined_experiments.py --experiment quick_test
    python run_combined_experiments.py --experiment setup_ratio_optimization
    python run_combined_experiments.py --dataset combined_humor_dataset_clean --experiment baseline_comparison
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.utils.config_loader import ConfigLoader
from src.utils.focused_humor_experiments import (
    ProposedModelExperimentRunner, 
    FocusedExperimentLogger
)


class YAMLExperimentRunner:
    """YAML-configuration-driven experiment runner."""
    
    def __init__(self, config_file: str = "experiment_configs_combined.yaml"):
        # Ensure config file path is relative to the script location
        if not os.path.isabs(config_file) and not os.path.exists(config_file):
            script_dir = os.path.dirname(__file__)
            config_file = os.path.join(script_dir, config_file)
        self.config_file = config_file
        self.config_loader = ConfigLoader()
        self.experiment_runner = ProposedModelExperimentRunner()
    
    def run_single_experiment_config(self, 
                                   config, 
                                   experiment_name: str,
                                   experiment_dir: Path) -> dict:
        """Run a single experiment configuration."""
        
        # Create logger for this configuration
        config_dir = experiment_dir / config.experiment_id
        logger = FocusedExperimentLogger(config_dir)
        
        print(f"  🔄 Running: {config.experiment_id}")
        print(f"      Dataset: {config.dataset}")
        print(f"      SPA heads: {config.spa_heads}, Setup ratio: {config.setup_ratio}")
        print(f"      Epochs: {config.epochs}, Batch size: {config.batch_size}")
        
        start_time = time.time()
        
        try:
            # Run the experiment
            result = self.experiment_runner.run_single_experiment(config, logger)
            
            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {
                    'success': bool(result),
                    'config': config.to_dict(),
                    'error': None if result else 'Unknown error'
                }
            
            result['runtime'] = time.time() - start_time
            
            # Save results
            logger.save_all_results()
            
            print(f"      ✅ Completed in {result['runtime']:.1f}s")
            if 'metrics' in result:
                f1 = result['metrics'].get('f1_score', 0)
                acc = result['metrics'].get('accuracy', 0)
                print(f"      📊 F1: {f1:.4f}, Accuracy: {acc:.4f}")
            
            return result
            
        except Exception as e:
            runtime = time.time() - start_time
            print(f"      ❌ Failed after {runtime:.1f}s: {e}")
            
            result = {
                'success': False,
                'config': config.to_dict(),
                'error': str(e),
                'runtime': runtime
            }
            
            return result
    
    def run_experiment_group(self, 
                           experiment_name: str, 
                           configs: list,
                           base_experiment_dir: Path) -> dict:
        """Run all configurations for a single experiment group."""
        
        print(f"\n🔬 Running {experiment_name} ({len(configs)} configurations)")
        print("-" * 60)
        
        experiment_dir = base_experiment_dir / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for i, config in enumerate(configs, 1):
            print(f"\n  [{i}/{len(configs)}]", end=" ")
            
            result = self.run_single_experiment_config(
                config, experiment_name, experiment_dir
            )
            
            results[config.experiment_id] = result
        
        # Save experiment group summary (filter out non-serializable objects)
        summary_file = experiment_dir / "experiment_summary.json"
        serializable_results = {}
        
        for config_id, result in results.items():
            # Create a JSON-safe copy of the result
            safe_result = {}
            for key, value in result.items():
                if key == 'model':
                    # Skip the model object - it's saved separately
                    continue
                elif key == 'history' and isinstance(value, dict):
                    # Keep only serializable parts of history
                    safe_history = {}
                    for hist_key, hist_value in value.items():
                        if isinstance(hist_value, (list, int, float, str, bool)):
                            safe_history[hist_key] = hist_value
                        elif hasattr(hist_value, 'tolist'):  # numpy arrays
                            safe_history[hist_key] = hist_value.tolist()
                    safe_result[key] = safe_history
                else:
                    safe_result[key] = value
            serializable_results[config_id] = safe_result
        
        with open(summary_file, 'w') as f:
            json.dump({
                'experiment_name': experiment_name,
                'num_configs': len(configs),
                'results': serializable_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n  📁 Results saved to: {experiment_dir}")
        
        return results
    
    def run_experiment(self, 
                      experiment_name: str, 
                      dataset: str = "humor") -> dict:
        """Run a single experiment by name."""
        
        print(f"🎯 Running Experiment: {experiment_name}")
        print(f"📊 Dataset: {dataset}")
        print("=" * 60)
        
        # Load configurations
        try:
            experiment_configs = self.config_loader.load_experiment_suite(
                self.config_file,
                experiment_names=[experiment_name],
                dataset=dataset
            )
        except Exception as e:
            print(f"❌ Failed to load configurations: {e}")
            return {'error': str(e)}
        
        if experiment_name not in experiment_configs:
            print(f"❌ Experiment '{experiment_name}' not found in {self.config_file}")
            return {'error': f'Experiment not found: {experiment_name}'}
        
        configs = experiment_configs[experiment_name]
        
        # Create base experiment directory
        base_dir = Path("experiments") / "yaml_driven" 
        
        # Run experiment
        results = self.run_experiment_group(experiment_name, configs, base_dir)
        
        return {experiment_name: results}
    
    def run_all_experiments(self, dataset: str = "humor") -> dict:
        """Run all experiments defined in the config file."""
        
        print("🚀 Running All Experiments")
        print(f"📊 Dataset: {dataset}")
        print("=" * 60)
        
        # Load all configurations
        try:
            experiment_configs = self.config_loader.load_experiment_suite(
                self.config_file,
                dataset=dataset
            )
        except Exception as e:
            print(f"❌ Failed to load configurations: {e}")
            return {'error': str(e)}
        
        # Print summary
        self.config_loader.print_config_summary(experiment_configs)
        
        # Create base experiment directory
        base_dir = Path("experiments") / "yaml_driven"
        
        all_results = {}
        
        # Run each experiment group
        for exp_name, configs in experiment_configs.items():
            try:
                results = self.run_experiment_group(exp_name, configs, base_dir)
                all_results[exp_name] = results
            except Exception as e:
                print(f"❌ Experiment group '{exp_name}' failed: {e}")
                all_results[exp_name] = {'error': str(e)}
        
        # Save comprehensive results (JSON-safe)
        summary_dir = base_dir / "comprehensive_summary" / datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Create JSON-safe version of all_results
        safe_all_results = {}
        for exp_name, exp_results in all_results.items():
            if isinstance(exp_results, dict) and 'error' not in exp_results:
                safe_exp_results = {}
                for config_id, result in exp_results.items():
                    # Keep only essential metrics and avoid model objects
                    safe_result = {
                        'success': result.get('success', False),
                        'metrics': result.get('metrics', {}),
                        'runtime': result.get('runtime', 0),
                        'error': result.get('error'),
                        'config': result.get('config', {})
                    }
                    # Remove model-related keys from config
                    if 'config' in safe_result and isinstance(safe_result['config'], dict):
                        safe_result['config'] = {k: v for k, v in safe_result['config'].items() 
                                               if not k.startswith('model') or k in ['model_name']}
                    safe_exp_results[config_id] = safe_result
                safe_all_results[exp_name] = safe_exp_results
            else:
                safe_all_results[exp_name] = exp_results
        
        with open(summary_dir / "all_results.json", 'w') as f:
            json.dump(safe_all_results, f, indent=2)
        
        print(f"\n🎉 All experiments completed!")
        print(f"📁 Comprehensive results: {summary_dir}")
        
        return all_results
    
    def print_results_summary(self, results: dict):
        """Print a summary of experimental results."""
        
        print("\n📊 EXPERIMENT RESULTS SUMMARY")
        print("=" * 50)
        
        total_configs = 0
        successful_configs = 0
        best_f1 = 0
        best_config = None
        
        for exp_name, exp_results in results.items():
            if isinstance(exp_results, dict) and 'error' not in exp_results:
                print(f"\n🔬 {exp_name}:")
                
                for config_id, result in exp_results.items():
                    total_configs += 1
                    
                    if result.get('success', False):
                        successful_configs += 1
                        
                        if 'metrics' in result:
                            f1 = result['metrics'].get('f1_score', 0)
                            acc = result['metrics'].get('accuracy', 0)
                            runtime = result.get('runtime', 0)
                            
                            print(f"  ✅ {config_id}: F1={f1:.4f}, Acc={acc:.4f}, Time={runtime:.1f}s")
                            
                            if f1 > best_f1:
                                best_f1 = f1
                                best_config = config_id
                        else:
                            print(f"  ✅ {config_id}: Success (no metrics)")
                    else:
                        error = result.get('error', 'Unknown error')
                        print(f"  ❌ {config_id}: {error}")
            else:
                print(f"\n❌ {exp_name}: Failed to run")
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total configurations: {total_configs}")
        print(f"   Successful: {successful_configs}")
        print(f"   Success rate: {successful_configs/total_configs*100:.1f}%" if total_configs > 0 else "   Success rate: 0%")
        if best_config:
            print(f"   Best F1 score: {best_f1:.4f} ({best_config})")


def main():
    parser = argparse.ArgumentParser(description='Run YAML-configured humor experiments on combined dataset')
    
    parser.add_argument('--experiment', 
                       choices=['all', 'architecture_comparison', 'spa_analysis', 'setup_ratio_optimization', 
                               'backbone_freezing', 'regularization_study', 
                               'learning_rate_study', 'baseline_comparison',
                               'quick_test', 'sample_test'],
                       default='quick_test',
                       help='Which experiment to run')
    
    parser.add_argument('--config', 
                       default='experiment_configs_combined.yaml',
                       help='Configuration file to use')
    
    parser.add_argument('--dataset',
                       default='humor',
                       help='Dataset to use for experiments')
    
    args = parser.parse_args()
    
    print("🎭 YAML-Driven Combined Dataset Experiment Runner")
    print("=" * 50)
    print(f"Config file: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Experiment: {args.experiment}")
    print()
    
    # Initialize runner
    runner = YAMLExperimentRunner(args.config)
    
    try:
        # Run experiments
        if args.experiment == 'all':
            results = runner.run_all_experiments(args.dataset)
        else:
            results = runner.run_experiment(args.experiment, args.dataset)
        
        # Print summary
        runner.print_results_summary(results)
        
    except Exception as e:
        print(f"❌ Experiment runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
