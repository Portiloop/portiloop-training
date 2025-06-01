# Portiloop Python Documentation

## Project Overview

The **portiloop_python** project is a comprehensive machine learning system designed for automated sleep analysis, specifically focusing on:

- **Sleep spindle detection** - Identifying characteristic EEG patterns during sleep
- **Sleep stage classification** - Categorizing different phases of sleep
- **Pareto Multi-objective Bayesian Optimization (PMBO)** - Automated hyperparameter optimization
- **Multi-task learning** - Joint training for multiple sleep analysis objectives

The system employs a hybrid CNN-RNN architecture with automated hyperparameter search to achieve optimal performance across multiple objectives (accuracy vs. computational efficiency).

## Architecture Overview

```
portiloop_python/
├── Utils/              # Core utilities and configuration management
├── PMBO/              # Pareto Multi-objective Bayesian Optimization system
├── ANN/               # Artificial Neural Networks (main ML pipeline)
│   ├── models/        # Neural network architectures
│   └── data/          # Data loading and preprocessing
├── Dataset generation/ # Dataset creation and preprocessing tools
└── Old files/         # Legacy implementations and experiments
```

## Directory Structure and File Interactions

### Utils/ - Core Utilities

#### `utils.py` (228 lines)
**Purpose**: Central configuration and utility functions for the entire system.

**Key Functions**:
- `sample_config_dict()`: Generates random or Gaussian-sampled neural network configurations
- `sample_from_range()`: Samples hyperparameters from defined ranges
- `same_config_dict()`: Compares two configurations for equality
- `clip()`: Value clipping utility

**Hyperparameter Ranges Defined**:
- Sequence length, kernel sizes, channel counts
- Learning rates, batch sizes, dropout rates
- Architecture parameters (CNN layers, RNN layers)

**Interactions**:
- Used by `PMBO/pareto_search.py` for configuration sampling
- Referenced by all training scripts for consistent parameter definitions
- Feeds into `ANN/portiloop_detector_training.py` for model instantiation

#### `merge_results.py` (35 lines)
**Purpose**: Aggregates experimental results from distributed runs.

**Functionality**:
- Merges JSON files from multiple experiment runs
- Consolidates results into single `merged_results.json`

**Interactions**:
- Processes outputs from PMBO experiments
- Used after distributed training campaigns

### PMBO/ - Pareto Multi-objective Bayesian Optimization

This directory implements a sophisticated distributed hyperparameter optimization system that balances multiple objectives (e.g., accuracy vs. model size).

#### `pareto_search.py` (581 lines)
**Purpose**: Core meta-learning engine for Pareto-optimal hyperparameter search.

**Key Classes**:
- `SurrogateModel`: 3-layer MLP that predicts model performance from hyperparameters
- `MetaDataset`: Dataset wrapper for training the surrogate model
- `LoggerWandbPareto`: Weights & Biases integration for experiment tracking

**Key Functions**:
- `update_pareto()`: Maintains Pareto front of non-dominated solutions
- `train_surrogate()`: Trains surrogate model on historical experiments
- `transform_config_dict_to_input()`: Converts config to neural network input
- `exp_max_pareto_efficiency()`: Selects most promising experiments

**Data Flow**:
1. Receives completed experiments from workers
2. Updates Pareto front with non-dominated solutions
3. Trains surrogate model on historical data
4. Generates new promising configurations
5. Sends configurations to workers for evaluation

**Interactions**:
- Uses `Utils/utils.py` for configuration sampling
- Sends configs to `ANN/portiloop_detector_training.py` via workers
- Receives results from distributed training runs
- Logs to Weights & Biases for visualization

#### `pareto_network.py` (463 lines)
**Purpose**: Distributed system coordinator implementing server-worker architecture.

**Key Classes**:
- `MetaLearner`: Coordinates optimization process, communicates with server
- `Worker`: Executes training jobs, reports results back

**Communication Flow**:
```
MetaLearner ←→ Server ←→ Workers
     ↓           ↓         ↓
  Config     Queuing   Training
Generation  Management Execution
```

**Interactions**:
- Uses `pareto_search.py` for optimization logic
- Communicates via `pareto_network_server_utils.py`
- Launches `ANN/portiloop_detector_training.py` on workers
- Manages experiment lifecycle and result collection

#### `pareto_network_server_utils.py` (394 lines)
**Purpose**: Network communication infrastructure for distributed PMBO.

**Key Classes**:
- `Server`: Central coordinator managing worker queues and meta-learner communication

**Features**:
- Socket-based communication with timeout handling
- Experiment queuing and load balancing
- Fault tolerance and connection management

#### `pareto_trainer.py` (78 lines)
**Purpose**: Entry point for launching PMBO campaigns.

**Functionality**:
- Initializes and starts meta-learning process
- Handles command-line arguments and configuration

#### `pareto_viewer.py` (96 lines)
**Purpose**: Visualization and analysis of Pareto optimization results.

**Features**:
- Plots Pareto fronts showing trade-offs
- Analyzes optimization convergence
- Exports results for further analysis

### ANN/ - Artificial Neural Networks

This directory contains the core machine learning pipeline, including model architectures, training procedures, and data handling.

#### `portiloop_detector_training.py` (792 lines)
**Purpose**: Main training orchestrator for portiloop neural networks.

**Key Functions**:
- `train()`: Complete training pipeline with validation and early stopping
- `run_inference()`: Model evaluation on validation/test sets
- `run()`: High-level training coordinator called by PMBO workers

**Features**:
- Multi-task learning support (spindles + sleep staging)
- Flexible loss functions (BCE, CrossEntropy)
- Early stopping and learning rate scheduling
- Comprehensive metric calculation
- Weights & Biases integration

**Training Pipeline**:
1. Load configuration from PMBO or manual setup
2. Initialize model using `models/lstm.py`
3. Load data using `data/mass_data.py`
4. Execute training with validation monitoring
5. Return performance metrics to PMBO system

**Interactions**:
- Receives configurations from `PMBO/pareto_search.py`
- Uses `models/lstm.py` for network architecture
- Loads data via `data/mass_data.py` or `data/moda_data.py`
- Returns results to PMBO for optimization

#### `lightning_mass.py` (888 lines)
**Purpose**: PyTorch Lightning implementation for streamlined training.

**Key Classes**:
- `MassLightning`: Lightning module for standard CNN-RNN architecture
- `MassLightningViT`: Vision Transformer variant for EEG analysis

**Features**:
- Automatic GPU/distributed training handling
- Built-in validation and testing loops
- Comprehensive metric tracking
- Multi-task loss balancing

**Advantages over standard training**:
- Simplified distributed training
- Automatic mixed precision
- Built-in checkpointing and logging
- Cleaner separation of training logic

#### `adaptation_training.py` (1735 lines)
**Purpose**: Advanced training with domain adaptation capabilities.

**Features**:
- Transfer learning between datasets
- Domain adaptation techniques
- Advanced regularization strategies
- Cross-domain validation

#### `utils.py` (479 lines)
**Purpose**: ANN-specific utilities and helper functions.

**Key Functions**:
- `get_configs()`: Load training configurations
- `set_seeds()`: Reproducible random number generation
- `get_metrics()`: Calculate evaluation metrics
- Various preprocessing and postprocessing utilities

#### `wamsley_utils.py` (516 lines)
**Purpose**: Traditional signal processing baselines for spindle detection.

**Features**:
- Wamsley algorithm implementation
- Classical spindle detection methods
- Baseline performance comparison
- Signal processing utilities

### ANN/models/ - Neural Network Architectures

#### `lstm.py` (283 lines)
**Purpose**: Core neural network architecture implementation.

**Key Classes**:
- `PortiloopNetwork`: Main hybrid CNN-RNN architecture
- `ConvPoolModule`: Convolutional building blocks
- `FcModule`: Fully connected building blocks

**Architecture Details**:
```
Input EEG Signal
       ↓
   CNN Layers (feature extraction)
       ↓
   GRU/LSTM (temporal modeling)
       ↓
   Embedding Layer
       ↓
  ┌─────────────────┐
  ↓                 ↓
Spindle           Sleep Stage
Classifier        Classifier
```

**Multi-task Learning**:
- Shared feature extraction (CNN + RNN)
- Separate classifiers for different tasks
- Joint training with weighted loss combination

**Interactions**:
- Instantiated by `portiloop_detector_training.py`
- Configuration from `Utils/utils.py`
- Used in both standard and Lightning training

#### `model_blocks.py` (705 lines)
**Purpose**: Reusable neural network components and advanced architectures.

**Components**:
- Attention mechanisms
- Transformer layers
- Custom activation functions
- Advanced pooling operations

#### Other Model Files:
- `sleep_staging_models.py`: Specialized architectures for sleep staging
- `encoding.py`: Embedding and encoding layers
- `masking.py`: Sequence masking for variable-length inputs
- `old_lstm.py`: Legacy implementations for comparison

### ANN/data/ - Data Loading and Preprocessing

#### `mass_data.py` (1177 lines)
**Purpose**: Primary data loader for MASS (Montreal Archive of Sleep Studies) dataset.

**Key Classes**:
- `MassDataset`: Main dataset class for MASS data
- `SpindleTrainDataset`: Specialized for spindle detection training
- `SleepStageDataset`: Sleep stage classification data
- `MassRandomSampler`: Efficient random sampling
- `MassValidationSampler`: Sequential validation sampling

**Data Pipeline**:
1. Read EDF files using `pyedflib`
2. Load sleep stage and spindle annotations
3. Apply preprocessing (filtering, normalization)
4. Create sliding windows for temporal modeling
5. Generate batches for training

**Features**:
- Multi-channel EEG support
- Flexible windowing strategies
- Memory-efficient loading for large datasets
- Support for both classification and regression

#### `mass_data_new.py` (805 lines)
**Purpose**: Enhanced MASS data loader with improved functionality.

**Improvements**:
- Better memory management
- Enhanced preprocessing options
- Improved sampling strategies
- Better integration with Lightning

#### Other Data Files:
- `moda_data.py`: MODA dataset loading
- `sleepedf_data.py`: Sleep-EDF dataset support
- `reg_balancing.py`: Class balancing for imbalanced datasets
- `adaptation_data.py`: Cross-domain data utilities

### Dataset generation/ - Data Creation Tools

#### `dataset_generator_classification.py` (125 lines)
**Purpose**: Creates classification datasets from raw EEG recordings.

**Process**:
1. Read raw EDF files and annotations
2. Extract segments around sleep events
3. Create binary labels for spindle presence
4. Export in format suitable for training

#### `dataset_generator_regression.py` (120 lines)
**Purpose**: Creates regression datasets with continuous targets.

**Features**:
- Continuous spindle density estimation
- Regression-based approach to detection
- Flexible target generation strategies

### Old files/ - Legacy and Experimental Code

Contains historical implementations and experimental features:
- `portiloop_detector_test.py`: Legacy testing procedures
- `simulatePortiloop.py`: Simulation utilities
- `experiments.py`: Experimental configurations
- Various prototype and deprecated implementations

## System Interactions and Data Flow

### 1. Hyperparameter Optimization Flow

```
Utils/utils.py (config ranges)
        ↓
PMBO/pareto_search.py (meta-learning)
        ↓
PMBO/pareto_network.py (distribution)
        ↓
ANN/portiloop_detector_training.py (training)
        ↓
Results back to PMBO system
```

### 2. Training Pipeline Flow

```
ANN/data/mass_data.py (data loading)
        ↓
ANN/models/lstm.py (model architecture)
        ↓
ANN/portiloop_detector_training.py (training loop)
        ↓
Metrics and checkpoints
```

### 3. Multi-objective Optimization

The system balances multiple objectives:
- **Performance**: Accuracy, F1-score, precision, recall
- **Efficiency**: Model size, inference time, memory usage
- **Robustness**: Generalization across subjects and datasets

### 4. Distributed Architecture

```
Meta-Learner (optimization logic)
        ↓
Server (coordination)
        ↓
Workers (parallel training)
        ↓
Results aggregation
```

## Key Technologies and Dependencies

- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: High-level training framework
- **NumPy/SciPy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **pyedflib**: EDF file reading for EEG data
- **Weights & Biases**: Experiment tracking
- **Socket programming**: Distributed communication
- **JSON**: Configuration and result storage

## Usage Patterns

### 1. Single Experiment
```python
from ANN.portiloop_detector_training import run
from Utils.utils import sample_config_dict

config = sample_config_dict("experiment_name", {}, [])
results = run(config, "project_name", save_model=True, unique_name=True)
```

### 2. PMBO Campaign
```bash
python PMBO/pareto_network.py --meta  # Start meta-learner
python PMBO/pareto_network.py --server  # Start server
python PMBO/pareto_network.py --worker  # Start workers
```

### 3. Lightning Training
```python
from ANN.lightning_mass import MassLightning
from pytorch_lightning import Trainer

model = MassLightning(config)
trainer = Trainer(gpus=1, max_epochs=100)
trainer.fit(model, train_dataloader, val_dataloader)
```

## Performance Monitoring and Logging

The system provides comprehensive monitoring through:
- **Weights & Biases**: Real-time experiment tracking
- **Pareto front visualization**: Multi-objective trade-off analysis
- **Model checkpointing**: Best model preservation
- **Metric logging**: Detailed performance tracking
