Welcome to Wellness Dimension.

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
MultiWD Dataset at: https://github.com/drmuskangarg/MultiWD
WellXplain Dataset at: https://github.com/drmuskangarg/WellnessDimensions/


