# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import argparse
import subprocess
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import torch

def seedTorch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    print(checkpoints)
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()

    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    seed: int,
    prompts: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 16,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

   
    prompts = [prompts]

    seedTorch(seed)
    torch.manual_seed(seed)

    
    results = generator.generate(
        prompts, max_gen_len=40000, temperature=temperature, top_p=top_p
    )
    return results

def letsgo(prompts, seed, ckpt):
    if ckpt == '65B':
        x = subprocess.run(['''torchrun''' ,'''--nproc_per_node''', '''8''', '''/notebooks/example.py''', '''--ckpt_dir''', '''/notebooks/LLaMA/65B''', '''--tokenizer_path''', '''/notebooks/LLaMA/tokenizer.model''', '''--seed''', '''12''', '''--prompts''', f'{prompts}', '--seed', f'{seed}'], capture_output=True)
        out = str(x.stdout).split(' seconds')[1]
        s = ''.join(out.splitlines())
        s = s.replace('/^\s+|\s+$/g', '');
        print(f'the generated text is: {s[2:-3]}')

        return s[2:-3]
    if ckpt == '30B':
        x = subprocess.run(['''torchrun''' ,'''--nproc_per_node''', '''4''', '''/notebooks/example.py''', '''--ckpt_dir''', '''/notebooks/LLaMA/30B''', '''--tokenizer_path''', '''/notebooks/LLaMA/tokenizer.model''', '''--seed''', '''12''', '''--prompts''', f'{prompts}', '--seed', f'{seed}'], capture_output=True)
        out = str(x.stdout).split(' seconds')[1]
        s = ''.join(out.splitlines())
        s = s.replace('/^\s+|\s+$/g', '');
        print(f'the generated text is: {s[2:-3]}')

        return s[2:-3]
    if ckpt == '13B':
        x = subprocess.run(['''torchrun''' ,'''--nproc_per_node''', '''2''', '''/notebooks/example.py''', '''--ckpt_dir''', '''/notebooks/LLaMA/13B''', '''--tokenizer_path''', '''/notebooks/LLaMA/tokenizer.model''', '''--seed''', '''12''', '''--prompts''', f'{prompts}', '--seed', f'{seed}'], capture_output=True)
        out = str(x.stdout).split(' seconds')[1]
        s = ''.join(out.splitlines())
        s = s.replace('/^\s+|\s+$/g', '');
        print(f'the generated text is: {s[2:-3]}')

        return s[2:-3]
    if ckpt == '7B':
        x = subprocess.run(['''torchrun''' ,'''--nproc_per_node''', '''1''', '''/notebooks/example.py''', '''--ckpt_dir''', '''/notebooks/LLaMA/7B''', '''--tokenizer_path''', '''/notebooks/LLaMA/tokenizer.model''', '''--seed''', '''12''', '''--prompts''', f'{prompts}', '--seed', f'{seed}'], capture_output=True)
        out = str(x.stdout).split(' seconds')[1]
        s = ''.join(out.splitlines())
        s = s.replace('/^\s+|\s+$/g', '');
        print(f'the generated text is: {s[2:-3]}')
        return s[2:-3]
                          
import gradio as gr



with gr.Blocks(css="#margin-top {margin-top: 15px} #center {text-align: center;} #description {text-align: center}") as demo:
    with gr.Row(elem_id="center"):
        gr.Markdown("# LLaMa Inference with Gradient")
    with gr.Row(elem_id = 'description'):
        gr.Markdown(""" To run LLaMA, be sure to select a model size that works for your machine. Single GPUs should always use '7B'.\n Start typing below and then click **Run to generate text** to see the output.""")
    with gr.Row():
        ckpt = gr.Radio(["7B", "13B", "30B", '65B'], label="Checkpoint directory", value = "30B", interactive = True)
        seed = gr.Slider(label = 'Seed', value = 42, minimum = 1, maximum = 10000, step =1, interactive = True)
        prompts = gr.Textbox(label = 'Prompt input', placeholder="What is your prompt?", value = 'my new invention is the', interactive = True)

    btn = gr.Button("Run to generate text")
    with gr.Row():
        out = gr.Textbox()
    with gr.Row():
        gr.Image('assets/logo.png').style(height = 53, width = 125, interactive = False)

    btn.click(fn=letsgo, inputs=[prompts, seed, ckpt], outputs=out)


demo.launch(share = True)

