# Enhanced Saiga-Mod Project

Welcome to the `enhanced-saiga-mod` project, a comprehensive collection of Jupyter notebooks specifically designed for
training large language models on datasets from the [Saiga (rulm)](https://github.com/IlyaGusev/rulm) project. This
repository is an essential resource for anyone looking to leverage the advanced capabilities of the Saiga datasets for
language model training.

Our notebooks are crafted to provide intuitive, step-by-step guidance for training state-of-the-art LoRA adapters for
different language models, ensuring that even those new to the field can successfully navigate the complexities of
language model training.

## Repository Contents

### Jupyter Notebooks

* [yarn_mistral_7b_128k.ipynb](./yarn_mistral_7b_128k.ipynb) - this notebook contains a script for training the
  [NousResearch/Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) model. This model, an
  advancement over the base [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), incorporates
  the [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) algorithm, enabling it to handle a
  context size of up to 128k tokens. The notebook provides a detailed and user-friendly guide for training the LoRA
  adapter specifically for the Yarn-Mistral-7b-128k model. It meticulously outlines the necessary steps and parameters
  required to optimize performance and achieve the best possible results with this enhanced model.
* [rugpt35_13b.ipynb](./rugpt35_13b.ipynb) - This notebook focuses on training
  the [ruGPT-3.5-13B](https://huggingface.co/ai-forever/ruGPT-3.5-13B) model, a powerful
  language model specifically tailored for understanding and generating Russian text. It guides users through creating a
  LoRA layer for model adaptation and subsequently performing a conversion to the GGML format for optimized deployment.
* [llama2_7b_yakovlev.ipynb](./llama2_7b_yakovlev.ipynb) - This notebook provides a detailed guide for training a
  Russian language model based on the [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
  model. The model is trained to imitate a historical figure
  named [Ivan Yakovlevich Yakovlev](https://