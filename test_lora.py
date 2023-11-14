
import torch
import logging
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# MODEL_NAME = "IlyaGusev/gigasaiga_lora"
# MODEL_NAME = "evilfreelancer/ruGPT-3.5-13B-lora"
# MODEL_NAME = "evilfreelancer/saiga_mistral_7b_128k_lora"
# MODEL_NAME = "./yarn_mistral_7b_128k"
MODEL_NAME = "./yarn_mistral_7b_128k_yakovlev"

DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"