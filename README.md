# MNIST — Linear and Deep Models

This repository contains a Jupyter notebook (ML_Assignment_2.ipynb) implementing and evaluating linear classifiers and a simple multi-layer perceptron on the MNIST dataset. The project was developed by Mohammed Hatem, Ahmed Ayman, and Joseph Yousery.

Contents
- ML_Assignment_2.ipynb — end-to-end notebook with data preparation, implementations of: 
  - Binary logistic regression (from-scratch)
  - Softmax (multiclass) regression (from-scratch and PyTorch-builtins)
  - Flexible multi-layer perceptron (MLP) with training, evaluation and visualizations
- data/ — (created automatically by torchvision when running the notebook)

Highlights
- Implements logistic and softmax regression from first principles (manual parameter updates) to clarify the algorithms.
- Provides a flexible MLP class with configurable hidden layers and activation functions (ReLU, LeakyReLU).
- Includes training utilities: early stopping, per-epoch metrics, learning curves with error bars, confusion matrices, and per-class accuracy.

How to run
1. Clone the repo:
   ```bash
   git clone https://github.com/Mohammed-Hatem/mnist-linear-and-deep.git
   cd mnist-linear-and-deep
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install torch torchvision matplotlib scikit-learn pandas seaborn
   ```
   If you want GPU support, install a PyTorch build that matches your CUDA version from https://pytorch.org/get-started/locally/.

3. Open the notebook with Jupyter or Colab and run the cells in order:
   ```bash
   jupyter notebook ML_Assignment_2.ipynb
   ```

Reproducibility notes
- Random seed: torch.manual_seed(42) is set in the notebook; for full reproducibility also set numpy and Python seeds if desired.
- Data splits: the notebook uses stratified splits for the linear baselines and a train/validation split for the MLP experiments; see cells in the notebook for details.

Suggested improvements
- Use torch.optim for all training loops (replacing manual parameter updates) and explore optimizers like Adam and learning-rate schedules.
- Add regularization (dropout, weight decay), batch normalization, and data augmentation to improve performance.
- Provide a script to run experiments non-interactively and to sweep hyperparameters.

License
This repository is provided for educational purposes. Add a license file if you want to permit wider reuse.
