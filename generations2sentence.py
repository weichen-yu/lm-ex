import os
import numpy as np
from transformers import GPT2Tokenizer

name = 'baseline_3000'
gen_path = os.path.join('/home2/ywc/extraction/lm-extraction-benchmark/baseline/tmp/' ,name,'generations')
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

for file in os.listdir(gen_path):
    file = os.path.join(gen_path,file)
    tokens = np.load(file)

    