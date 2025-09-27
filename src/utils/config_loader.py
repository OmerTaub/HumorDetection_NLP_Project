#!/usr/bin/env python3
"""
Configuration Loader for Humor Experiments
==========================================

This module provides functionality to load and parse YAML configuration files
for humor detection experiments, converting them to ProposedModelConfig objects.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, replace
import copy

try:
    from .focused_humor_experiments import ProposedModelConfig
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.utils.focused_humor_experiments import ProposedModelConfig


class ConfigLoader:
    """Loads and manages experiment configurations from YAML files."""
    
    def __init__(self, config_dir: str = "."):
        self.config_dir = Path(config_dir)
    
    def load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _convert_numeric_types(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string representations of numbers to proper numeric types."""
        converted = config_dict.copy()
        
        # Define numeric fields and their expected types
        numeric_fields = {
            'learning_rate': float,
            'dropout_rate': float, 
            'setup_ratio': float,
            'epochs': int,
            'batch_size': int,
            'spa_heads': int,
            'max_seq_length': int,
            'random_seed': int
        }
        
        for field, expected_type in numeric_fields.items():
            if field in converted:
                try:
                    if isinstance(converted[field], str):
                        converted[field] = expected_type(converted[field])
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert {field}={converted[field]} to {expected_type.__name__}: {e}")
        
        return converted
    
    def create_base_config(self, base_config_dict: Dict[str, Any], dataset: Optional[str] = None) -> ProposedModelConfig:
        """Create a base ProposedModelConfig from a dictionary."""
        # Override dataset if provided
        if dataset:
            base_config_dict['dataset'] = dataset
        
        # Ensure proper type conversion for numeric values
        base_config_dict = self._convert_numeric_types(base_config_dict)
        
        # Create config with defaults, overriding with provided values
        config = ProposedModelConfig(
            experiment_id="base_config",
            experiment_category="base",
            **base_config_dict
        )
        
        return config
    
    def generate_experiment_configs(self, 
                                   experiment_config: Dict[str, Any], 
                                   base_config: ProposedModelConfig) -> List[ProposedModelConfig]:
        """Generate all configuration variations for an experiment."""
        configs = []
        
        base_params = experiment_config.get('base_params', {})
        variations = experiment_config.get('variations', [])
        
        # Override dataset if specified in experiment config
        if 'dataset' in experiment_config:
            base_config = replace(base_config, dataset=experiment_config['dataset'])
        
        # Convert numeric types in base_params
        base_params = self._convert_numeric_types(base_params)
        
        # Create base configuration with experiment-specific parameters
        exp_base_config = replace(base_config, **base_params)
        
        # Generate variations
        for variation in variations:
            experiment_id = variation.get('experiment_id', 'unnamed_experiment')
            
            # Create variation config
            var_params = {k: v for k, v in variation.items() if k != 'experiment_id'}
            var_params = self._convert_numeric_types(var_params)
            var_config = replace(exp_base_config, 
                               experiment_id=experiment_id,
                               **var_params)
            
            configs.append(var_config)
        
        return configs
    
    def load_experiment_suite(self, 
                            config_file: str, 
                            experiment_names: Optional[List[str]] = None,
                            dataset: Optional[str] = None,
                            base_config_name: str = "base_config") -> Dict[str, List[ProposedModelConfig]]:
        """Load a complete experiment suite from a YAML file."""
        
        # Load YAML config
        yaml_config = self.load_yaml_config(config_file)
        
        # Create base configuration
        base_config_dict = yaml_config.get(base_config_name, {})
        base_config = self.create_base_config(base_config_dict, dataset)
        
        # Load experiments
        experiment_configs = {}
        
        # If no specific experiments requested, load all
        if experiment_names is None:
            experiment_names = [key for key in yaml_config.keys() 
                              if key not in [base_config_name, 'puns_base_config'] 
                              and not key.startswith('Experiment')]
        
        for exp_name in experiment_names:
            if exp_name in yaml_config:
                exp_config = yaml_config[exp_name]
                configs = self.generate_experiment_configs(exp_config, base_config)
                experiment_configs[exp_name] = configs
            else:
                print(f"Warning: Experiment '{exp_name}' not found in config file")
        
        return experiment_configs
    
    def load_single_experiment(self, 
                             config_file: str, 
                             experiment_name: str,
                             dataset: Optional[str] = None,
                             base_config_name: str = "base_config") -> List[ProposedModelConfig]:
        """Load configurations for a single experiment."""
        
        suite = self.load_experiment_suite(config_file, [experiment_name], dataset, base_config_name)
        return suite.get(experiment_name, [])
    
    def print_config_summary(self, experiment_configs: Dict[str, List[ProposedModelConfig]]):
        """Print a summary of loaded configurations."""
        
        print("📋 Loaded Experiment Configurations:")
        print("=" * 50)
        
        total_configs = 0
        for exp_name, configs in experiment_configs.items():
            print(f"\n🔬 {exp_name}: {len(configs)} configurations")
            for config in configs:
                print(f"  - {config.experiment_id}")
                print(f"    Dataset: {config.dataset}")
                print(f"    Epochs: {config.epochs}, Batch: {config.batch_size}")
                print(f"    SPA heads: {config.spa_heads}, Setup ratio: {config.setup_ratio}")
            total_configs += len(configs)
        
        print(f"\n📊 Total configurations: {total_configs}")


def test_config_loader():
    """Test the configuration loader with example files."""
    
    print("Testing Configuration Loader")
    print("=" * 40)
    
    loader = ConfigLoader()
    
    # Test loading puns config
    try:
        configs = loader.load_experiment_suite(
            "experiment_configs_puns.yaml",
            experiment_names=["spa_analysis", "quick_test"],
            dataset="puns_pos_neg"
        )
        
        loader.print_config_summary(configs)
        
        print("\n✅ Configuration loader test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_config_loader()
