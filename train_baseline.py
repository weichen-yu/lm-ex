# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
from absl import logging
import csv
import os
import tempfile
from typing import Tuple, Union

import numpy as np
import transformers
import torch
import time
import torch.backends.cudnn as cudnn

import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


_ROOT_DIR = flags.DEFINE_string(
    'root-dir', "tmp/",
    "Path to where (even intermediate) results should be saved/loaded."
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment-name',
    'sample',
    "Name of the experiment. This defines the subdir in `root_dir` where "
    "results are saved.")
_DATASET_DIR = flags.DEFINE_string(
    "dataset-dir", "../datasets",
    "Path to where the data lives.")
_DATSET_FILE = flags.DEFINE_string(
    "dataset-file", "train_dataset.npy", "Name of dataset file to load.")
_NUM_TRIALS = flags.DEFINE_integer(
    'num-trials', 100, 'Number of generations per prompt.')
_local_rank = flags.DEFINE_integer(
    'local_rank', 0, 'cuda num')
_iter = flags.DEFINE_integer(
    'iter', 1000, 'iteration for training')

torch.distributed.init_process_group('nccl', init_method='env://')
if torch.distributed.get_world_size() != torch.cuda.device_count():
    raise AssertionError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
        torch.distributed.get_world_size(), torch.cuda.device_count()))


_SUFFIX_LEN = 50
_PREFIX_LEN = 50



class Model(nn.Module):
    def __init__(self, config=None):
        super(Model,self).__init__()
        self._MODEL = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self._MODEL = self._MODEL.half().cuda().eval()
        self.hidden_dim = 10
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim*_SUFFIX_LEN,_SUFFIX_LEN)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, prefix, batch_size):
        """Generates suffixes given `prompts` and scores using their likelihood.

        Args:
        prompts: A np.ndarray of shape [num_prompts, prefix_length]. These
            provide the context for generating each suffix. Each value should be an
            int representing the token_id. These are directly provided by loading the
            saved datasets from extract_dataset.py.
        batch_size: The number of prefixes to generate suffixes for
            sequentially.

        Returns:
            A tuple of generations and their corresponding likelihoods.
            The generations take shape [num_prompts, _SUFFIX_LEN].
            The likelihoods take shape [num_prompts]
        """
        generation_len = _SUFFIX_LEN + _PREFIX_LEN
        for i, off in enumerate(range(0, len(prefix), batch_size)):
            prefix_batch = prefix[off:off+batch_size]
            prefix_batch = np.stack(prefix_batch, axis=0)
            input_ids = torch.tensor(prefix_batch, dtype=torch.int64)
            with torch.no_grad():
                # 1. Generate outputs from the model
                generated_tokens = self._MODEL.generate(
                    input_ids.cuda(),
                    max_length=generation_len,
                    do_sample=True,
                    num_beams=5,
                    top_k=10,
                    top_p=1,
                    pad_token_id=50256  # Silences warning.
                ).cpu().detach()

                # 2. Compute each sequence's probability, excluding EOS and SOS.
                outputs = self._MODEL(
                    generated_tokens.cuda(),
                    labels=generated_tokens.cuda(),
                )
                import pdb;pdb.set_trace()
                logits_ = outputs.logits
                losses_ = outputs.losses
                logits = logits_[:, :-1].reshape((-1, logits_.shape[-1])).float()
                
                p_ = self.bn(logits)
                p_ = self.linear(p_)
                p_ = self.relu(p_)
                # 
                
                # generations.extend(generated_tokens.numpy())
                # losses.extend(likelihood.numpy())
        return p_

    

def write_array(
    file_path: str, array: np.ndarray, unique_id: Union[int, str]):
    """Writes a batch of `generations` and `losses` to a file.

    Formats a `file_path` (e.g., "/tmp/run1/batch_{}.npy") using the `unique_id`
    so that each batch goes to a separate file. This function can be used in
    multiprocessing to speed this up.

    Args:
        file_path: A path that can be formatted with `unique_id`
        array: A numpy array to save.
        unique_id: A str or int to be formatted into `file_path`. If `file_path`
          and `unique_id` are the same, the files will collide and the contents
          will be overwritten.
    """
    file_ = file_path.format(unique_id)
    np.save(file_, array)


def load_prompts(dir_: str, file_name: str) -> np.ndarray:
    """Loads prompts from the file pointed to `dir_` and `file_name`."""
    return np.load(os.path.join(dir_, file_name)).astype(np.int64)

def separate_dataset():
    prompts_all = load_prompts(_DATASET_DIR.value, "train_prefix.npy")
    #if not os.path.exists(os.path.join(_DATASET_DIR.value,'val_prefix.npy')):
    #    os.mk
    prompts_val_prefix = prompts_all[-1000:]
    name = 'val_prefix'
    np.save(os.path.join(_DATASET_DIR.value,'val_prefix.npy'),prompts_val_prefix)
    
def is_memorization(guesses, answers):
    precision = 0
    for guess in guesses:
        precision += min(np.sum(np.all(guess == answers, axis=-1)),1)
    precision = precision/guesses.shape[0]
    return precision
    #return np.all(guesses==answers, axis=-1)

# for a lot of prompts pre prefix, see each prefix respectively
def precision_multiprompts(generations,answers,num_perprompt):
    precision_multi = 0
    generations = generations[:,:num_perprompt,:]
    for generation in generations:
        is_in = 0
        for prompt in generation:
            is_in += min(np.sum(np.all(prompt == answers, axis=-1)),1)
        precision_multi += min(is_in,1)
    precision_multi = precision_multi/generations.shape[0]
    return precision_multi

def error_100(guesses, answers):
    error=0
    i=0
    while error <= 100:
        if np.sum(np.all(guesses[i]==answers,axis=-1)):
            i += 1
        else:
            error += 1
            i += 1
    return i
def np2var(x):
    x = torch.from_numpy(x)
    x = autograd.Variable(x).cuda()
    return x

class loss(nn.Module):
    def __init__(self):
        super(loss,self).__init__()
        self.CE = F.cross_entropy(
                #     logits, generated_tokens[:, 1:].flatten(), reduction='none')
                # loss_per_token = loss_per_token.reshape((-1, generation_len - 1))[:,-_SUFFIX_LEN-1:-1]
                # likelihood = loss_per_token.mean(1)

    def forward(self,predictions,answers)


def main(_):
    #separate_dataset()
    start_t = time.time()
    
    experiment_base = os.path.join(_ROOT_DIR.value, _EXPERIMENT_NAME.value)
    generations_base = os.path.join(experiment_base, "generations")
    os.makedirs(generations_base, exist_ok=True)
    losses_base = os.path.join(experiment_base, "losses")
    os.makedirs(losses_base, exist_ok=True)
    prompts = load_prompts(_DATASET_DIR.value, "train_prefix.npy")[-1000:]
    answers = np.load(os.path.join(_DATASET_DIR.value, "val_dataset.npy"))[:, -50:].astype(np.int64)


    # We by default do not overwrite previous results.
    all_generations, all_losses = [], []
    model = Model().float().cuda()
    prompts = np2var(prompts)
    answers = np2var(answers)
    os.makedirs(experiment_base, exist_ok=True)

    for iter in range(_NUM_TRIALS.value):
        predictions = model(prompts)
        
        generation_string = os.path.join(generations_base, "{}.npy")
        losses_string = os.path.join(losses_base, "{}.npy")

        write_array(generation_string, generations, trial)
        write_array(losses_string, losses, trial)

            all_generations.append(generations)
            all_losses.append(losses)
        generations = np.stack(all_generations, axis=1)
        losses = np.stack(all_losses, axis=1)
   

    for generations_per_prompt in [1, 10, 100]:
        limited_generations = generations[:, :generations_per_prompt, :]
        limited_losses = losses[:, :generations_per_prompt, :]

        print('generations shape:',limited_losses.shape)
        
        axis0 = np.arange(generations.shape[0])
        axis1 = limited_losses.argmin(1).reshape(-1)
        guesses = limited_generations[axis0, axis1, -_SUFFIX_LEN:]
        batch_losses = limited_losses[axis0, axis1]
        
        with open("guess%d.csv"%generations_per_prompt, "w") as file_handle:
            print("Writing out guess with", generations_per_prompt)
            writer = csv.writer(file_handle)
            writer.writerow(["Example ID", "Suffix Guess"])

            order = np.argsort(batch_losses.flatten())
            
            # Write out the guesses
            for example_id, guess in zip(order, guesses[order]):
                row_output = [
                    example_id, str(list(guesses[example_id])).replace(" ", "")
                ]
                writer.writerow(row_output)
        #print([np.sum(np.all(answers[i]==answers,axis=-1)) for i in range(15000)])
        #print(sum([np.sum(np.all(answers[i]==answers,axis=-1)) for i in range(15000)]))
        print('guess and answer shape:', guesses.shape, answers.shape)
        print('precision:',is_memorization(guesses, answers))
        print('precision_multi:',precision_multiprompts(generations[:,:,-50:], answers,generations_per_prompt))
        print('error100 number:',error_100(guesses, answers))
        end_t = time.time()
        print('time cost:',end_t-start_t)
# for test






if __name__ == "__main__":
    app.run(main)
