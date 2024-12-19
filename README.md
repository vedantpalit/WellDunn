## WellDunn: On the Robustness and Explainability of Language Models and Large Language Models in Identifying Wellness Dimensions


This is the official repository for our work **"WellDunn: On the Robustness and Explainability of Language Models and Large Language Models in Identifying Wellness Dimensions"**, accepted in the BlackBox NLP Workshop at EMNLP 2024. 

You can read the full paper at [Link](https://aclanthology.org/2024.blackboxnlp-1.23/).


<p align="center">
    <img src="https://github.com/user-attachments/assets/2185dd03-8fc0-4959-bbba-991a2b0282b0" style="width: 100%; max-width: 639px;"><br>
    <strong>WellDunn workflow</strong><em>: MULTIWD task (L) and WELLXPLAIN task (R). The architecture includes shared steps: (1)
Fine-tuning of general purpose and domain-specific LMs for extracting data representations, followed by (2) feeding them into a
feed-forward neural network classifier (FFNN). Two loss functions assess LMs’ robustness: Sigmoid Cross-Entropy(SCE) and
Gambler’s Loss(GL). Singular Value Decomposition (SVD) and Attention-Overlap (AO) Score assess the explainability. In:
Input, and Out: Output. WellDunn Benchmarking Box: This middle rectangle highlights the components of the benchmark
system, which includes steps of (1) Fine-tuning and (2) FFNN classifier, as well as Robustness and Explainability components. </em>
</p>

There are five folders in our code:
1) LMs_MultiWD
This folder contains code on MultiWD task with Gambler's loss (GL) and Sigmoid Cross Entropy loss (SCE). It also includes code to calculate SVD.
2) LMs_WellXplain
This folder contains code on WellXplain task with Gambler's loss (GL) and Sigmoid Cross Entropy loss (SCE). It also includes code to calculate SVD and AO.
3) LLMs_WellXplain
This includes code for Large Language Models (LLMs). It has GPT-4 and GPT 3.5 (for both zero-shot and few-shot settings), Llama and MedAlpaca on WellXplain task with SCE.
4) environment.txt
This file contains the requirements for the environment to run the code.
5) README

To train and test models in cases (1) and (2) and for Llama and MedAlpaca in case (3), you would need to upload the datasets MultiWD and WellXplain into the current path of the project on your device or (for Colab use) upload them into your Google Drive (in this path /content/drive/MyDrive/results). To do so, just go to the Data Loading part or search for pd.read_csv and change it according to your preference.

Since we need huggingface login, you would need to create an access token in your account in huggingface at https://huggingface.co/settings/toke link and use it where it requests for such a token or assign it in any variable named "access_token_read."

You may get different results since it runs in different situations; however, the results would not differ significantly.

You can use Google Colab to run the Jupyter Source File. Note that we used Colab Pro, which provides us with A100.

To run a code, you just need to run it in the order in which the codes appear.

We will provide the full datasets after acceptance of the paper.

Each code file includes appropriate comments to help understand the code.

Note that the datasets we have used are available at the following links:

  - MultiWD Dataset at: https://github.com/drmuskangarg/MultiWD

  - WellXplain Dataset at: https://github.com/drmuskangarg/WellnessDimensions/

## Citation

**If you find our repository useful for your work, you can cite by:**

```bibtex
@inproceedings{mohammadi-etal-2024-welldunn,
    title = "{W}ell{D}unn: On the Robustness and Explainability of Language Models and Large Language Models in Identifying Wellness Dimensions",
    author = "Mohammadi, Seyedali  and
      Raff, Edward  and
      Malekar, Jinendra  and
      Palit, Vedant  and
      Ferraro, Francis  and
      Gaur, Manas",
    editor = "Belinkov, Yonatan  and
      Kim, Najoung  and
      Jumelet, Jaap  and
      Mohebbi, Hosein  and
      Mueller, Aaron  and
      Chen, Hanjie",
    booktitle = "Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2024",
    address = "Miami, Florida, US",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.blackboxnlp-1.23",
    doi = "10.18653/v1/2024.blackboxnlp-1.23",
    pages = "364--388",
    abstract = "Language Models (LMs) are being proposed for mental health applications where the heightened risk of adverse outcomes means predictive performance may not be a sufficient litmus test of a model{'}s utility in clinical practice. A model that can be trusted for practice should have a correspondence between explanation and clinical determination, yet no prior research has examined the attention fidelity of these models and their effect on ground truth explanations. We introduce an evaluation design that focuses on the robustness and explainability of LMs in identifying Wellness Dimensions (WDs). We focus on two existing mental health and well-being datasets: (a) Multi-label Classification-based MultiWD, and (b) WellXplain for evaluating attention mechanism veracity against expert-labeled explanations. The labels are based on Halbert Dunn{'}s theory of wellness, which gives grounding to our evaluation. We reveal four surprising results about LMs/LLMs: (1) Despite their human-like capabilities, GPT-3.5/4 lag behind RoBERTa, and MedAlpaca, a fine-tuned LLM on WellXplain fails to deliver any remarkable improvements in performance or explanations. (2) Re-examining LMs{'} predictions based on a confidence-oriented loss function reveals a significant performance drop. (3) Across all LMs/LLMs, the alignment between attention and explanations remains low, with LLMs scoring a dismal 0.0. (4) Most mental health-specific LMs/LLMs overlook domain-specific knowledge and undervalue explanations, causing these discrepancies. This study highlights the need for further research into their consistency and explanations in mental health and well-being.",
}
