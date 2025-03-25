# Machine-Learning-Accelerated-Prediction-of-Amorphization-Enthalpy-in-Ionic-Compounds

This repository provides code for training and fine-tuning Graph Neural Networks (GNNs) to predict the amorphization enthalpy of ionic compounds. The project integrates density functional theory (DFT) data and e3nn models to analyze and predict the amorphization enthalpy of these materials.


## Requirements

To install the required Python packages, use the following command:

```bash
pip install -r requirements.txt
```


### Directory Structure
The repository is organized as follows:

#### ./checkpoints/
**finetune**: Contains the fine-tuned models used in this study.  
**pretrain**: Contains the pretrained models used in this study.  

#### ./dataset/
**feature.csv**: The features used during the pretraining stage.  
**finetune_input.csv**: The features used during the fine-tuning stage.  
**dft_enthalpy.csv**: The amorphization enthalpy data evaluated by DFT method.  
**GNN_enthalpy.csv**: The amorphization enthalpy data evaluated by GNN method.  
**cifs.tar.xz**/: A compassed folder containing CIF files for each material used in both pretraining and fine-tuning. Each CIF file is named according to the material ID.  

#### ./scripts/
Scripts for pretraining and fine-tuning the models.

#### ./utils/
Utility functions for pretraining and fine-tuning.


## Usage
### 1. Pretraining the Model for Shear Modulus Prediction
To pretrain the model for shear modulus prediction, run the following script:
```bash
python scripts/training.py
```

### 2. Fine-tuning vs. Scratch Comparison
To perform a 5-fold cross-validation comparison between the fine-tuned GNN model and a model trained from scratch, run:
```bash
python scripts/finetune.py
```

## Usage
If you use this code in your research, please cite the corresponding publication.
```Yu, Q., Sun, G., & Luo, W. (2025). Machine Learning-Accelerated Prediction of Amorphization Enthalpy in Ionic Compounds. ACS Materials Letters, 1496–1502. https://doi.org/10.1021/acsmaterialslett.5c00066
