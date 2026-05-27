# SAT: Structured Action Transformer (Simulation)

This directory contains the official simulation implementation of **SAT (Structured Action Transformer)**. SAT is a policy architecture designed for complex robotic manipulation tasks, leveraging structured representations and flow matching for high-performance action generation.

## 🚀 Installation

### 1. Environment Setup
We recommend using Conda to manage your environment. Ensure you have PyTorch and CUDA installed correctly.
```bash
conda create -n sat python=3.9
conda activate sat
# Install PyTorch according to your CUDA version
pip install torch torchvision
```

### 2. Install SAT Package
Install the core `sat` package in editable mode:

```bash
cd sim
pip install -e .
```

### 3. Dependencies
Install other required dependencies:
```bash
pip install -r requirements.txt
```
*(Note: Ensure you have simulation environments like Adroit, MetaWorld, or DexArt installed as required by your tasks.)*

## 📁 Dataset Preparation
The training scripts assume that the datasets are organized in the following structure relative to the project root:

```text
sim/
├── data/
│   ├── adroit/
│   ├── metaworld/
│   └── dexart/
```

## 🛠️ Usage

### 1. Training from Scratch
To train a SAT policy for a specific task:

```bash
python train.py --config-name=sat.yaml task=adroit_hammer training.seed=42
```

### 2. Multi-GPU Training (Batch Tasks)
Use the provided bash script to run multiple tasks across different GPUs:

```bash
bash train.bash
```
You can modify the `tasks` and `gpu_device_ids` in `train.bash` to customize your training queue.

### 3. Pre-training
To run large-scale pre-training on mixed datasets:

```bash
python pretrain.py --config-name=sat.yaml task=mix_pretrain
```

### 4. Transfer Learning
To fine-tune a pre-trained model on a target task:

```bash
python train_transfer.py --config-name=sat_transfer.yaml task=dexart_laptop
```

### 5. Evaluation
To evaluate a trained checkpoint:

```bash
python eval_policy.py --config-name=sat.yaml task=adroit_hammer hydra.run.dir=outputs/your_experiment_path
```

## ⚙️ Configuration
We use **Hydra** for configuration management. All config files are located in `sat/config/`.

- `sat/config/sat.yaml`: Main configuration for SAT.
- `sat/config/task/`: Task-specific parameters (observation shapes, dataset paths, etc.).
- `sat/config/sat_transfer.yaml`: Configuration for transfer learning.

You can override any parameter via command line:
```bash
python train.py policy.num_inference_steps=10 training.batch_size=256
```

## 📂 Directory Structure

- `sat/`: Core Python package containing models, policies, and datasets.
  - `model/`: SAT architecture and vision encoders.
  - `policy/`: SAT policy wrapper and flow matching logic.
  - `config/`: Hydra YAML configuration files.
- `scripts/`: Helper shell scripts for training and evaluation.
- `train.py`: Main entry point for standard training.
- `pretrain.py`: Entry point for pre-training.
- `train_transfer.py`: Entry point for fine-tuning/transfer learning.
- `eval_policy.py`: Script for evaluating trained models.
