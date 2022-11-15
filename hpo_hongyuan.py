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

import copy
import gc
import numpy as np
import transformers
import torch
import torch.distributed
import random
import time
import torch.backends.cudnn as cudnn

import argparse
import optuna
import logging
from optuna.trial import TrialState
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

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
    'num-trials', 25, 'Number of generations per prompt.')
_local_rank = flags.DEFINE_integer(
    'local_rank', 0, 'cuda num')

#torch.distributed.init_process_group('nccl', init_method='env://')
#if torch.distributed.get_world_size() != torch.cuda.device_count():
#    raise AssertionError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
#        torch.distributed.get_world_size(), torch.cuda.device_count()))


_SUFFIX_LEN = 50
_PREFIX_LEN = 50

parser = argparse.ArgumentParser(description='Auto Tuning')
# args for hyperparameters search
parser.add_argument('--search_algo', type=str, default='tpe', help='search method tpe or cma')
parser.add_argument('--search_direction', type=str, default='maximize', help='minimize maximize')
parser.add_argument('--num_samples', default=9999999, type=int, help='number of samples')
parser.add_argument('--n_jobs', default=4, type=int, help='number of samples')
parser.add_argument('--log_file', type=str, default="search.log", help='log file')
parser.add_argument('--seed', default=2023, type=int, help='seed')
args = parser.parse_args()

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=args.log_file, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

def generate_for_prompts(
    model, device, prompts: np.ndarray, trial: int, beam: int, topk: int, topp: float, tp: float, rp: float,  batch_size: int=32) -> Tuple[np.ndarray, np.ndarray]:
    '''
    beam 正整数，=1的话等于没有beam search   [1,10,1]
    topk：正整数，一般好像能到40或者50  [1,50,1]
    pa: penalty_alpha 不要了  
    topp: top_p:小于1的浮点数          [0.1,1,0.1]
    tp: typical_p: 小于1的浮点数       [0.1,1,0.1]
    rp: repetition_penalty: 浮点数     [0.1,1,0.1]
    '''
    generations = []
    losses = []
    generation_len = _SUFFIX_LEN + _PREFIX_LEN

    for i, off in enumerate(range(0, len(prompts), batch_size)):
        prompt_batch = prompts[off:off+batch_size]
        print(
            "Generating for batch ID {:05} of size {:04}, trial {:04}".format(i, len(prompt_batch),trial))
        prompt_batch = np.stack(prompt_batch, axis=0)
        input_ids = torch.tensor(prompt_batch, dtype=torch.int64)
        with torch.no_grad():
            # 1. Generate outputs from the model
            generated_tokens = model.generate(
                input_ids.to(device),
                max_length=generation_len,
                do_sample=True,
                num_beams=beam,
                top_k=topk,
                #penalty_alpha=pa,
                typical_p=tp,
                top_p=topp,
                repetition_penalty=rp,
                pad_token_id=50256  # Silences warning.
            ).cpu().detach()

            # 2. Compute each sequence's probability, excluding EOS and SOS.
            outputs = model(
                generated_tokens.to(device),
                labels=generated_tokens.to(device),

            )
            logits = outputs.logits.cpu().detach()
            logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
            loss_per_token = torch.nn.functional.cross_entropy(
                logits, generated_tokens[:, 1:].flatten(), reduction='none')
            loss_per_token = loss_per_token.reshape((-1, generation_len - 1))[:,-_SUFFIX_LEN-1:-1]
            likelihood = loss_per_token.mean(1)
            
            generations.extend(generated_tokens.numpy())
            losses.extend(likelihood.numpy())
    return np.atleast_2d(generations), np.atleast_2d(losses).reshape((len(generations), -1))


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


def parameters_lm(beam,topk,topp,tp,rp, model, device, exp_id):
    root_dir = "/data/ywc/auto_tuning/{}".format(exp_id)
    experiment_base = os.path.join(root_dir, _EXPERIMENT_NAME.value)
    generations_base = os.path.join(experiment_base, "generations")
    os.makedirs(generations_base, exist_ok=True)
    losses_base = os.path.join(experiment_base, "losses")
    os.makedirs(losses_base, exist_ok=True)
    prompts = load_prompts(_DATASET_DIR.value, "train_prefix.npy")[-1000:]
     # We by default do not overwrite previous results.
    all_generations, all_losses = [], []
    #if not all([os.listdir(generations_base), os.listdir(losses_base)]):
    if True:
        for trial in range(_NUM_TRIALS.value):
            os.makedirs(experiment_base, exist_ok=True)
            print('trial:',trial)
            generations, losses = generate_for_prompts(model, device, prompts,trial,beam,topk,topp,tp,rp)

            generation_string = os.path.join(generations_base, "{}.npy")
            losses_string = os.path.join(losses_base, "{}.npy")

            write_array(generation_string, generations, trial)
            write_array(losses_string, losses, trial)

            all_generations.append(generations)
            all_losses.append(losses)
        generations = np.stack(all_generations, axis=1)
        losses = np.stack(all_losses, axis=1)
    # else:  # Load saved results because we did not regenerate them.
    #     #import pdb;pdb.set_trace()
    #     generations = []
    #     for generation_file in sorted(os.listdir(generations_base)):
    #         file_ = os.path.join(generations_base, generation_file)
    #         generations.append(np.load(file_))
    #     # Generations, losses are shape [num_prompts, num_trials, suffix_len].
    #     generations = np.stack(generations, axis=1)[:,:100,:]

        # losses = []
        # for losses_file in sorted(os.listdir(losses_base)):
        #     file_ = os.path.join(losses_base, losses_file)
        #     losses.append(np.load(file_))
        # losses = np.stack(losses, axis=1)
    #for generations_per_prompt in [1, 10, 100]:
    for generations_per_prompt in [1]:
        limited_generations = generations[:, :generations_per_prompt, :]
        limited_losses = losses[:, :generations_per_prompt, :]

        print('generations shape:',limited_losses.shape)
        
        axis0 = np.arange(generations.shape[0])
        axis1 = limited_losses.argmin(1).reshape(-1)
        guesses = limited_generations[axis0, axis1, -_SUFFIX_LEN:]
        batch_losses = limited_losses[axis0, axis1]
        
        # with open("guess%d.csv"%generations_per_prompt, "w") as file_handle:
        #     print("Writing out guess with", generations_per_prompt)
        #     writer = csv.writer(file_handle)
        #     writer.writerow(["Example ID", "Suffix Guess"])

        #     order = np.argsort(batch_losses.flatten())
            
        #     # Write out the guesses
        #     for example_id, guess in zip(order, guesses[order]):
        #         row_output = [
        #             example_id, str(list(guesses[example_id])).replace(" ", "")
        #         ]
        #         writer.writerow(row_output)
        answers = np.load(os.path.join(_DATASET_DIR.value, "val_dataset.npy"))[:, -50:].astype(np.int64)
        print('guess and answer shape:', guesses.shape, answers.shape)
        #print('precision:',is_memorization(guesses, answers))
        #print('precision_multi:',precision_multiprompts(generations[:,:,-50:], answers,generations_per_prompt))
        error_100_ = error_100(guesses, answers)
        print('error100 number:',error_100_)
    return error_100_

def objective(trial):
    exp_number = trial.number
    job_id = exp_number % args.n_jobs
    # time.sleep(job_id * 1)
    device = torch.device("cuda:{}".format(exp_number % args.n_jobs))
    _MODEL = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = _MODEL.half().to(device).eval()
    beam = trial.suggest_int('beam', 3, 8, step=1)
    topk = trial.suggest_int('topk', 1, 50, step=1)
    topp = trial.suggest_float('top_p', 0.1, 1, step=0.1)
    tp = trial.suggest_float('typical_p', 0.1, 1, step=0.1)
    rp = trial.suggest_float('repetition_penalty', 0.1, 1, step=0.1)
    try:
        score = parameters_lm(beam,topk,topp,tp,rp,model, device, exp_number)
    except:
        score = np.nan
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)
    logging.info("Exp_number: {} beam: {} topk: {} top_p: {} typical_p: {} repetition_penalty: {} Score: {:.4f}".format(exp_number, beam, topk, topp, tp, rp, score))
    return score


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def main(_):
    #separate_dataset()
    init_seeds(args.seed)
    if args.search_algo == 'cma':
        cma = optuna.samplers.CmaEsSampler()
        study = optuna.create_study(sampler=cma, direction=args.search_direction)
    else:
        study = optuna.create_study(
            direction=args.search_direction,
            pruner=optuna.pruners.HyperbandPruner(),
        )

    init_dict = {
      "beam": 5,
      "topk": 10,
      "top_p": 1,
      "typical_p": 1,
      "repetition_penalty": 1,
    }

    study.enqueue_trial(init_dict)

    study.optimize(objective, n_trials=args.num_samples, n_jobs=args.n_jobs)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logging.info("Study statistics: ")
    # logging.info("  Number of finished trials: ", len(study.trials))
    # logging.info("  Number of pruned trials: ", len(pruned_trials))
    # logging.info("  Number of complete trials: ", len(complete_trials))

    logging.info("Best trial:")
    trial = study.best_trial

    logging.info("Value: {}".format(trial.value))
    # logging.info("Params: ")
    # for key, value in trial.params.items():
    #     logging.info("    {}: {}".format(key, value))
    logging.info("Total tuning parameters: {}".format(len(trial.params.items())))



if __name__ == "__main__":
    app.run(main)
