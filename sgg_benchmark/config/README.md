# SGG-Benchmark Configuration System

## Overview

SGG-Benchmark uses **Hydra + OmegaConf** for configuration management, providing a modern, flexible, and type-safe configuration system with advanced features like config composition and variable interpolation.

## Quick Start

### Install Dependencies

```bash
pip install hydra-core omegaconf
```

### Basic Usage

```python
from omegaconf import DictConfig
from sgg_benchmark.config import load_config_from_file

# Load config from file
cfg = load_config_from_file("configs/hydra/default.yaml")
print(cfg.model.relation_on)

# Or use Hydra decorator (recommended)
import hydra

@hydra.main(version_base=None, config_path="configs/hydra", config_name="default")
def main(cfg: DictConfig):
    print(cfg.model.relation_on)
    # Full IDE autocomplete support!
    return cfg

if __name__ == "__main__":
    main()
```

## Configuration Methods

### Method 1: Hydra Decorator (Recommended ⭐)

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs/hydra", config_name="default")
def main(cfg: DictConfig):
    print(cfg.model.relation_on)
    # Full IDE autocomplete!

if __name__ == "__main__":
    main()
```

**Pros**: 
- Type hints and IDE support
- Built-in CLI argument handling
- Config composition
- Variable interpolation

**Cons**: Requires Hydra knowledge

### Method 2: Direct OmegaConf Loading

```python
from sgg_benchmark.config import load_config_from_file

cfg = load_config_from_file("configs/hydra/my_config.yaml")
print(cfg.model.relation_on)
```

**Pros**: Simple, clean syntax  
**Cons**: Manual CLI parsing needed

### Method 3: Config Manager

```python
from sgg_benchmark.config import Config

cfg = Config()
cfg.merge_from_file("configs/hydra/my_config.yaml")
print(cfg.model.relation_on)
```

**Pros**: Easy to use  
**Cons**: Slightly more verbose

## Command Line Usage

### Basic Arguments

```bash
python train.py \
    model.relation_on=true \
    solver.base_lr=0.001 \
    +model.new_param=value
```

### Override Syntax

```bash
# Set a value
python train.py model.relation_on=true

# Add a new parameter
python train.py +model.new_param=value

# Append to a list
python train.py +dataset.classes=[class1,class2]

# Multirun (sweep)
python train.py -m solver.base_lr=0.001,0.01,0.1
```

## Configuration Format

Configurations are written in YAML with lowercase keys:

```yaml
# configs/hydra/my_config.yaml
seed: 42
metric_to_track: "mR"

model:
  relation_on: true
  roi_relation_head:
    num_classes: 51
    use_gt_box: true

solver:
  base_lr: 0.001
  momentum: 0.9

global_setting:
  basic_encoder: "Hybrid-Attention"
  use_bias: true
  gcl_setting:
    group_split_mode: "divide4"
    knowledge_transfer_mode: "None"
    knowledge_loss_coefficient: 1.0
    no_relation_restrain: false
    zero_label_padding_mode: "rand_insert"
    no_relation_penalty: 0.1
```

### Accessing Configuration Values

```python
# Using dot notation
value = cfg.model.relation_on
num_classes = cfg.model.roi_relation_head.num_classes

# Using subscript notation
value = cfg["model"]["relation_on"]

# Type conversions happen automatically!
bool_value = cfg.model.relation_on  # Already proper type!
```

### Updating Configuration Values

```python
from omegaconf import OmegaConf

# Update existing values
OmegaConf.update(cfg, "model.relation_on", False)
OmegaConf.update(cfg, "solver.base_lr", 0.01)

# Add new values (requires struct mode to be off)
OmegaConf.set_struct(cfg, False)
cfg.model.new_param = "value"
OmegaConf.set_struct(cfg, True)
```

## Advanced Features

### Config Composition

Compose multiple configurations together:

```yaml
# configs/hydra/experiment.yaml
defaults:
  - base
  - model: yolo
  - solver: adam
  - override dataset: psg
  - _self_

# Automatically composes configs in order!
```

### Variable Interpolation

Reference other config values:

```yaml
output_dir: "/experiments/runs"
checkpoint_dir: "${output_dir}/checkpoints"
log_dir: "${output_dir}/logs"
dataset_path: "${oc.env:DATASET_ROOT,/data/vg}"
```

### Defaults List

```yaml
defaults:
  - base                          # Include base.yaml
  - model: yolo                   # Include model/yolo.yaml
  - override solver: adam         # Override solver from base
  - _self_                        # Current file overrides all above

# Load order: base -> model/yolo -> override solver -> current file
```

### Multirun and Hyperparameter Sweeps

```bash
# Sweep over learning rates
python train.py -m solver.base_lr=0.001,0.01,0.1

# Grid search
python train.py -m \
    solver.base_lr=0.001,0.01 \
    model.roi_relation_head.num_classes=51,134

# Range
python train.py -m 'solver.base_lr=range(0.001,0.1,0.01)'
```

## GCL Settings

GCL (Group Collaborative Learning) settings are configured in the `global_setting.gcl_setting` section:

```yaml
global_setting:
  gcl_setting:
    group_split_mode: "divide4"           # divide3, divide4, divide5, average
    knowledge_transfer_mode: "None"        # KL_logit_TopDown, KL_logit_BottomUp, etc.
    knowledge_loss_coefficient: 1.0
    no_relation_restrain: false
    zero_label_padding_mode: "rand_insert" # rand_insert, rand_choose, all_include
    no_relation_penalty: 0.1
```

Access in code:

```python
if cfg.global_setting.gcl_setting.group_split_mode == "divide4":
    # Use divide4 grouping
    pass
```

## Documentation

- **Implementation Details**: See `hydra_config.py` and `config_manager.py`
- **Structured Configs**: See `structured_configs.py` for type-safe config definitions
- **Examples**: Check `tools/train_example_hydra.py`

## Best Practices

1. **Use the Hydra decorator** for new training scripts - it provides the best experience
2. **Keep configs modular** - use composition to avoid duplication
3. **Use variable interpolation** - reduces hardcoded paths
4. **Type check configs** - use structured configs for IDE support
5. **Document config options** - add comments explaining non-obvious settings

## Example Configuration Structure

```
configs/
├── hydra/
│   ├── default.yaml              # Base
│   ├── PSG/
│   │   ├── react_yolov8m.yaml
│   │   └── react_yoloe_v11l.yaml
│   ├── VG150/
│   │   └── react_yolov8m.yaml
│   └── model/
│       ├── yolo.yaml
│       └── rcnn.yaml
```

