# 3D-SAT-OLED:  Geometry-Driven Attention Model with 3D Molecular Features for Multi-Property Prediction of OLED Materials 

The **3D-SAT-OLED** model is a deep learning architecture based on a three-dimensional spatial self-attention mechanism, designed for accurate prediction of key performance parameters of OLED materials. The model integrates molecular 3D structural information by incorporating a **Gaussian-kernel-based Geometric Property Extractor (GPE)** and embedding spatial positional encodings within the self-attention mechanism, thereby enhancing its ability to represent complex molecular conformations of OLED systems. Furthermore, the model employs a **Dynamic Tanh (DyT)** normalization mechanism and the **Swish-Gated Linear Unit (SwiGLU)** activation function to improve numerical stability and nonlinear modeling capacity. Experimental results demonstrate that 3D-SAT-OLED achieves **state-of-the-art (SOTA)** performance on both the QM9 and OLED-DFT datasets, with R², MAE, and other target property metrics significantly surpassing those of existing baseline models, showcasing strong generalization and predictive accuracy.

## **Hardware and Environment Requirements**

### CPU Node
- **Processor (CPU):** 4 cores (vCPUs), Intel(R) Xeon(R) Platinum 8200 series
- **Memory (RAM):** 16 GB  
- **Storage (Disk):** 40 GB  
- **GPU:** None  
- Suitable for: data preprocessing, feature extraction, small-scale testing

### GPU Node

- **Processor (CPU):** 8 cores (vCPUs) , Intel(R) Xeon(R) Platinum 8200 series
- **Memory (RAM):** 31 GB  
- **GPU:** 1 × NVIDIA T4 (16 GB GDDR6 VRAM, Turing architecture, 2560 CUDA cores, supports Tensor Cores/RT Cores)
- **Storage (Disk):** 40 GB  
- **Use cases:** Training the 3D-SAT-OLED model and performing large-scale inference

### Dependencies

> **Note:** This project requires **Python 3.9**. Please ensure you are using a Python 3.9 environment before installing dependencies.

| Package    | Version |
| ---------- | ------- |
| joblib     | 1.1.0   |
| numpy      | 1.20.3  |
| pandas     | 1.3.5   |
| addict     | 2.4.0   |
| tqdm       | 4.66.0  |
| sklearn    | 0.24.2  |
| torch      | 2.2.0   |
| pyyaml     | 6.0     |
| scipy      | 1.6.3   |
| matplotlib | 3.5.1   |

## Directory Structure

```
3D-SAT-OLED/
├── main.py                       # Batch test main entry
├── test_oled.py                  # OLED dataset test script
├── test_qm9.py                   # QM9 dataset test script
├── draw.py                       # Visualization and plotting tools
├── check_versions.py             # Dependency version checker
├── model_reproduction_videos/    # Videos for model reproduction
├── SAT_OLED_Model/               # Core model implementation
│   ├── config/                   # Default model configurations
│   ├── data/                     # Data processing and molecular structure generation
│   ├── models/                   # Model architectures
│   ├── transformers/             # Modified BERT 
│   ├── tasks/                    # Training and data partitioning tasks
│   ├── utils/                    # Utility functions
│   ├── weights/                  # Pretrained models and dictionaries
│   ├── oled/                     # OLED dataset
│   └── qm9/                      # QM9 dataset
```

## Datasets

### OLED QM Properties Dataset

- **Properties**: PLQY (Photoluminescence Quantum Yield), E_ad (Adiabatic Energy), HOMO（Highest Occupied Molecular Orbital）, LUMO(Lowest Unoccupied Molecular Orbita)
- **Format**: `.npz ` files containing molecular coordinates, atomic types, and target properties
- **Location**: `SAT_OLED_Model/oled/`

### QM9 Dataset

- **Properties**: HOMO（Highest Occupied Molecular Orbital）, LUMO(Lowest Unoccupied Molecular Orbita)、 Gap (HOMO-LUMO gap)
- **Format**: `.npz`files containing molecular coordinates, atomic types, and target properties
- **Location**:`SAT_OLED_Model/qm9/`

## Model 

- The  model is `3D-SAT-OLED`, which is based on the BERT structure. It develops a molecular structure encoder  to support multi-label regression tasks and adapt to OLED material structure data.

- Weights can be found in the `SAT_OLED_Model/weights/` directory

## Configuration

The model can be configured via the `default.yaml` file or programmatically:

```python
config = {
    'target_columns': ['plqy', 'e_ad', 'homo', 'lumo'],  # OLED properties
    'model_name': 'SAT_OLED_Model',
    'data_type': 'molecule',
    'task': 'multilabel_regression',
    'epochs': 40,
    'learning_rate': 0.0002,
    'batch_size': 8,
    'patience': 10,
    'metrics': 'r2',
    'drop_out': 0.5,
    'pre_norm': True,
    'optim_type': 'AdamW',
    'seed': 42
}
```


## Quick Start

>**Note:** This project requires **Python 3.9**. Please ensure you are using a Python 3.9 environment before installing dependencies.

1. **Installation**

   ```bash
   pip install joblib==1.1.0 numpy==1.24.3 pandas==1.5.3 addict==2.4.0 tqdm==4.66.4 scikit-learn==0.24.2 torch==2.2.0 pyyaml==6.0.1 scipy==1.13.1
   ```

2. **Run Tests**

   ```bash
   # Check dependency versions
   python check_versions.py
   
   # Run all tests (OLED + QM9)
   python main.py
   
   # Run individual dataset tests
   python test_oled.py
   python test_qm9.py
   ```

   > Logs and results are saved in `exp_*` directories, with performance metrics and visualization results saved in `*_metrics` directories

## Performance Metrics

The model uses multiple metrics to evaluate performance:

- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MedianAE**: Median Absolute Error
- **Correlation**: Pearson correlation coefficient

> This work builds upon DP Technology’s UniMol framework, introducing architectural innovations specifically designed for OLED molecular modeling.
>  In particular, the `SAT_OLED_Model` enhances the BERT self-attention mechanism by embedding geometric positional encodings directly into the query-key computation.Combined with MaterialStructureEncoder, DyT normalization, and AngleExtractor, the model supports accurate prediction of multiple OLED properties (PLQY, E_ad, HOMO, LUMO,GAP).
