# Parameters Documentation

This document explores all parameters used in the CTR-GCN training and inference pipeline. Parameters are defined in `main.py` using `argparse.ArgumentParser` and can be set via command-line arguments or configuration files.

## Parameter Priority

**Priority order:** Command line > Config file > Default values

This means:
1. Command-line arguments override everything
2. Config file values override defaults
3. Default values are used if not specified elsewhere

## Parameter Categories

### 1. General Configuration

#### `--work-dir`
- **Type:** String
- **Default:** `'./work_dir/temp'`
- **Description:** The working directory for storing results, logs, and model checkpoints
- **Usage:** All outputs (models, logs, tensorboard files) are saved here
- **Example:** `--work-dir ./work_dir/ntu60/xsub/ctrgcn_joint`

#### `-model_saved_name`
- **Type:** String
- **Default:** `''` (empty string)
- **Description:** Custom name/path for saving the model
- **Usage:** If not set, defaults to `{work_dir}/runs` during training
- **Note:** Used for organizing different model runs

#### `--config`
- **Type:** String (file path)
- **Default:** `'./config/nturgbd-cross-view/test_bone.yaml'`
- **Description:** Path to the YAML configuration file
- **Usage:** Loads default parameter values from YAML file
- **Example:** `--config ./config/nturgbd-cross-subject/default.yaml`
- **Note:** Config file parameters are validated against command-line arguments

---

### 2. Processor Parameters

#### `--phase`
- **Type:** String
- **Default:** `'train'`
- **Description:** Execution phase/mode
- **Valid values:** `'train'` or `'test'`
- **Usage:** 
  - `train`: Training mode - trains the model
  - `test`: Testing mode - evaluates on test set
- **Example:** `--phase test`

#### `--save-score`
- **Type:** Boolean (via `str2bool`)
- **Default:** `False`
- **Description:** Whether to save classification scores
- **Usage:** If `True`, prediction scores are stored (useful for ensemble methods or detailed analysis)
- **Example:** `--save-score True`

---

### 3. Visualization and Debugging

#### `--seed`
- **Type:** Integer
- **Default:** `1`
- **Description:** Random seed for reproducibility
- **Usage:** Sets seed for PyTorch, NumPy, and Python's random module
- **Effect:** Ensures deterministic behavior (with `cudnn.deterministic = True`)
- **Example:** `--seed 42`

#### `--log-interval`
- **Type:** Integer
- **Default:** `100`
- **Description:** Interval (in iterations) for printing training progress messages
- **Usage:** Controls how frequently training metrics are printed to console
- **Example:** `--log-interval 50` (prints every 50 iterations)

#### `--save-interval`
- **Type:** Integer
- **Default:** `1`
- **Description:** Interval (in epochs) for saving model checkpoints
- **Usage:** How often to save model state during training
- **Example:** `--save-interval 5` (saves every 5 epochs)

#### `--save-epoch`
- **Type:** Integer
- **Default:** `30`
- **Description:** Starting epoch number for saving models
- **Usage:** Models are only saved after this epoch number
- **Example:** `--save-epoch 20` (start saving from epoch 20)

#### `--eval-interval`
- **Type:** Integer
- **Default:** `5`
- **Description:** Interval (in epochs) for evaluating the model on validation set
- **Usage:** How often to run validation during training
- **Example:** `--eval-interval 10` (evaluate every 10 epochs)

#### `--print-log`
- **Type:** Boolean (via `str2bool`)
- **Default:** `True`
- **Description:** Whether to print logging messages
- **Usage:** Set to `False` to suppress console output
- **Example:** `--print-log False`

#### `--show-topk`
- **Type:** List of integers
- **Default:** `[1, 5]`
- **Description:** Which Top-K accuracy metrics to display
- **Usage:** Shows Top-1, Top-5 accuracy, etc.
- **Example:** `--show-topk 1 5 10` (shows Top-1, Top-5, Top-10)
- **Note:** Uses `nargs='+'` to accept multiple values

---

### 4. Data Feeder Parameters

#### `--feeder`
- **Type:** String (module path)
- **Default:** `'feeder.feeder'`
- **Description:** Data loader class to use (dynamically imported)
- **Usage:** Specifies which feeder class loads and preprocesses the data
- **Examples:**
  - `feeders.feeder_ntu.Feeder` (for NTU-RGB+D dataset)
  - `feeders.feeder_ucla.Feeder` (for UCLA dataset)
- **Note:** Uses `import_class()` function to dynamically import the class

#### `--num-worker`
- **Type:** Integer
- **Default:** `32`
- **Description:** Number of worker processes for data loading
- **Usage:** Parallel data loading for faster training
- **Note:** Higher values = faster loading but more memory usage
- **Example:** `--num-worker 16`

#### `--train-feeder-args`
- **Type:** Dictionary (via `DictAction`)
- **Default:** `dict()` (empty dictionary)
- **Description:** Arguments passed to the training data feeder
- **Usage:** Configures data augmentation and preprocessing for training
- **Common arguments** (from config files):
  - `data_path`: Path to training data file
  - `split`: `'train'` or `'test'`
  - `debug`: Boolean for debug mode
  - `random_choose`: Randomly sample frames
  - `random_shift`: Random temporal shift
  - `random_move`: Random spatial movement
  - `window_size`: Number of frames per sample
  - `normalization`: Normalize skeleton data
  - `random_rot`: Random rotation augmentation
  - `p_interval`: Probability interval for sampling
  - `vel`: Use velocity features
  - `bone`: Use bone features
- **Example:** `--train-feeder-args "data_path='data/ntu/NTU60_CS.npz', split='train', window_size=64"`

#### `--test-feeder-args`
- **Type:** Dictionary (via `DictAction`)
- **Default:** `dict()` (empty dictionary)
- **Description:** Arguments passed to the test/validation data feeder
- **Usage:** Configures data preprocessing for testing (usually no augmentation)
- **Common arguments:** Similar to `train-feeder-args` but typically with:
  - `split: 'test'`
  - `debug: False`
  - No random augmentations
- **Example:** `--test-feeder-args "data_path='data/ntu/NTU60_CS.npz', split='test', window_size=64"`

---

### 5. Model Parameters

#### `--model`
- **Type:** String (module path)
- **Default:** `None`
- **Description:** Model class to use (dynamically imported)
- **Usage:** Specifies which model architecture to use
- **Examples:**
  - `model.ctrgcn.Model` (CTR-GCN model)
  - `model.baseline.Model` (baseline model)
- **Note:** Uses `import_class()` function to dynamically import the class

#### `--model-args`
- **Type:** Dictionary (via `DictAction`)
- **Default:** `dict()` (empty dictionary)
- **Description:** Arguments passed to the model constructor
- **Usage:** Configures model architecture parameters
- **Common arguments** (from config files):
  - `num_class`: Number of action classes (e.g., 60 for NTU-60, 120 for NTU-120)
  - `num_point`: Number of skeleton joints (e.g., 25 for NTU-RGB+D)
  - `num_person`: Maximum number of persons (e.g., 2)
  - `graph`: Graph class for skeleton structure
  - `graph_args`: Arguments for graph construction
    - `labeling_mode`: `'spatial'` or other modes
- **Example:** `--model-args "num_class=60, num_point=25, num_person=2"`

#### `--weights`
- **Type:** String (file path)
- **Default:** `None`
- **Description:** Path to pre-trained model weights file
- **Usage:** Load weights for fine-tuning or inference
- **Example:** `--weights ./work_dir/ntu60/xsub/ctrgcn_joint/runs/model_best.pth`

#### `--ignore-weights`
- **Type:** List of strings
- **Default:** `[]` (empty list)
- **Description:** Names of weight layers to ignore during initialization
- **Usage:** Skip loading specific layers (useful for transfer learning)
- **Example:** `--ignore-weights classifier.fc1 classifier.fc2`
- **Note:** Uses `nargs='+'` to accept multiple layer names

---

### 6. Optimizer Parameters

#### `--base-lr`
- **Type:** Float
- **Default:** `0.01`
- **Description:** Initial/base learning rate
- **Usage:** Starting learning rate for the optimizer
- **Note:** Learning rate scheduling multiplies this value
- **Example:** `--base-lr 0.1`

#### `--step`
- **Type:** List of integers
- **Default:** `[20, 40, 60]`
- **Description:** Epoch numbers where learning rate is reduced
- **Usage:** Learning rate decay milestones (multiplied by `lr-decay-rate`)
- **Example:** `--step 35 55` (decay at epochs 35 and 55)
- **Note:** Uses `nargs='+'` to accept multiple epoch numbers

#### `--device`
- **Type:** List of integers
- **Default:** `[0]`
- **Description:** GPU device indices for training/testing
- **Usage:** 
  - Single GPU: `[0]`
  - Multiple GPUs: `[0, 1, 2]` (enables DataParallel)
- **Example:** `--device 0 1` (use GPUs 0 and 1)
- **Note:** Uses `nargs='+'` to accept multiple device IDs

#### `--optimizer`
- **Type:** String
- **Default:** `'SGD'`
- **Description:** Optimizer type
- **Usage:** Specifies which optimizer to use (SGD, Adam, etc.)
- **Example:** `--optimizer Adam`

#### `--nesterov`
- **Type:** Boolean (via `str2bool`)
- **Default:** `False`
- **Description:** Whether to use Nesterov momentum (for SGD)
- **Usage:** Enables Nesterov accelerated gradient
- **Example:** `--nesterov True`

#### `--batch-size`
- **Type:** Integer
- **Default:** `256`
- **Description:** Training batch size
- **Usage:** Number of samples per training batch
- **Note:** Larger batch size = more memory, potentially faster training
- **Example:** `--batch-size 64`

#### `--test-batch-size`
- **Type:** Integer
- **Default:** `256`
- **Description:** Testing/validation batch size
- **Usage:** Number of samples per test batch
- **Note:** Can be different from training batch size
- **Example:** `--test-batch-size 64`

#### `--start-epoch`
- **Type:** Integer
- **Default:** `0`
- **Description:** Starting epoch number for training
- **Usage:** Resume training from a specific epoch
- **Example:** `--start-epoch 20` (resume from epoch 20)

#### `--num-epoch`
- **Type:** Integer
- **Default:** `80`
- **Description:** Total number of training epochs
- **Usage:** When to stop training
- **Example:** `--num-epoch 65`

#### `--weight-decay`
- **Type:** Float
- **Default:** `0.0005`
- **Description:** L2 regularization coefficient (weight decay)
- **Usage:** Prevents overfitting by penalizing large weights
- **Example:** `--weight-decay 0.0004`

#### `--lr-decay-rate`
- **Type:** Float
- **Default:** `0.1`
- **Description:** Learning rate decay multiplier
- **Usage:** Multiplies learning rate at each decay step
- **Example:** `--lr-decay-rate 0.1` (reduces LR by 10x at each step)

#### `--warm_up_epoch`
- **Type:** Integer
- **Default:** `0`
- **Description:** Number of warm-up epochs
- **Usage:** Gradually increase learning rate from 0 to base-lr over these epochs
- **Example:** `--warm_up_epoch 5`

---

## Parameter Parsing Flow

1. **Initial Parse** (`parser.parse_args()`):
   - Parses command-line arguments
   - Uses defaults for unspecified arguments

2. **Config File Loading**:
   - If `--config` is provided, loads YAML file
   - Validates all config keys exist in parser
   - Updates parser defaults with config values

3. **Final Parse** (`parser.parse_args()` again):
   - Re-parses with updated defaults
   - Command-line arguments still take priority

4. **Usage**:
   - Parsed arguments passed to `Processor` class
   - Accessed via `self.arg.parameter_name` throughout the code

## Helper Functions

### `str2bool(v)`
- **Purpose:** Converts string to boolean
- **True values:** `'yes'`, `'true'`, `'t'`, `'y'`, `'1'`
- **False values:** `'no'`, `'false'`, `'f'`, `'n'`, `'0'`
- **Usage:** Used for boolean parameters like `--save-score`, `--print-log`, `--nesterov`

### `DictAction`
- **Purpose:** Custom argparse action for dictionary arguments
- **Usage:** Allows passing dictionary arguments via command line
- **Example:** `--train-feeder-args "key1=value1, key2=value2"`
- **Note:** Uses `eval()` internally (be careful with untrusted input)

## Example Usage

### Command Line
```bash
python main.py \
  --work-dir ./work_dir/ntu60/xsub/ctrgcn_joint \
  --config ./config/nturgbd-cross-subject/default.yaml \
  --phase train \
  --batch-size 64 \
  --num-epoch 65 \
  --device 0 \
  --seed 1
```

### Config File (YAML)
```yaml
work_dir: ./work_dir/ntu60/xsub/ctrgcn_joint
feeder: feeders.feeder_ntu.Feeder
train_feeder_args:
  data_path: data/ntu/NTU60_CS.npz
  split: train
  window_size: 64
model: model.ctrgcn.Model
model_args:
  num_class: 60
  num_point: 25
  num_person: 2
batch_size: 64
num_epoch: 65
base_lr: 0.1
```

## Notes

- All parameters can be overridden via command line
- Config files provide a convenient way to manage parameter sets
- Dictionary parameters (`--train-feeder-args`, `--model-args`) are typically set in config files
- Boolean parameters accept various string formats via `str2bool`
- List parameters (`--device`, `--step`, `--show-topk`) accept multiple space-separated values
