# Wellness Dimension Benchmarking (__WellDunn__)

Welcome to the **Wellness Dimension** Benchmarking repository! This repository hosts code and datasets related to the Wellness Dimension Benchmarking.

## Getting Started

Follow these steps to get started with the project:

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. `curl -L https://tinyurl.com/4wucdvsw --output ExplainWD_model_RoBERTa_GL_4_Dimension.pt` execute this command in the terminal to download the file "ExplainWD_model_RoBERTa_GL_4_Dimension.pt" using curl.
4. Run `python3 example_welldunn_GL.py` to execute the example using Gamblers Loss on the EXPLAINWD dataset.
5. `curl -L https://tinyurl.com/3c9mjpvp --output ExplainWD_model_RoBERTa_SCE_4_Dimension.pt` execute this command in the terminal to download the file "ExplainWD_model_RoBERTa_SCE_4_Dimension.pt" using curl.
6. Run `python3 example_welldunn_SCE.py` to execute the example using Sigmoid Cross Entropy on the EXPLAINWD dataset.
7. For more detailed information on the training process, navigate to the `explainWD` and `multiWD` subfolders.


## Folders and Datasets

### 1. `explainWD` Folder

This folder contains code and datasets related to the Explain Wellness Dimension.

- `GamblersLoss_ExplainWD.ipynb`: Jupyter Notebook for the GamblersLoss explainability in Wellness Dimension.
- `SigmoidCrossEntropy_ExplainWD.py`: Python script for implementing Sigmoid Cross Entropy explainability in Wellness Dimension.
- `SigmoidCrossEntropy_ExplainWD.ipynb`: Jupyter Notebook demonstrating Sigmoid Cross Entropy explainability.
- `gamblersloss_explainwd.py`: Python script for GamblersLoss explainability.
- `ExplainWD.csv`: Dataset associated with the Explain Wellness Dimension.

### 2. `multiWD` Folder

This folder contains code and datasets related to the Multi Wellness Dimension.

- `GamblersLoss_MultiWD.ipynb`: Jupyter Notebook related to the GamblersLoss within the context of Multi-label Wellness Aspects.
- `MultiWD_CSE.py`: Python script for the application of Sigmoid Cross Entropy in the context of Multi-label Wellness Aspects.
- `MultiWD_CSE.ipynb`: Jupyter Notebook for interface friendly understanding of `MultiWD_CSE.py`
- `gamblersloss_multiwd.py`: Python script for GamblersLoss in Multi-label Wellness Dimension.
- `MultiWD.csv`: Dataset associated with the Multi-label Wellness Dimension.

## Contributing

We encourage contributions from the community to enhance the Wellness Dimension project.

Wishing you wellness and happiness!
