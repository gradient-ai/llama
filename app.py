# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import gradio as gr
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
from datetime import datetime
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import torch

def seedTorch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import string
nonprint = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'

def remove_non_printable(s):
    lst = []
    for i in s:
        if i in nonprint:
            lst.append(i)
    return "".join(lst)

def run(prompts, seed, ckpt):
    now = datetime.now()
    dict1 = {'7B':1,'13B':2,'30B':4, '65B': 8}
    MP = dict1[ckpt]
    x = subprocess.run(['''torchrun''' ,'''--nproc_per_node''', f'{MP}', '''/notebooks/example.py''', '''--ckpt_dir''', f'/notebooks/LLaMA/{ckpt}', '''--tokenizer_path''', '''/notebooks/LLaMA/tokenizer.model''', '''--seed''', '''12''', '''--prompts''', f'{prompts}', '--seed', f'{seed}'], capture_output=True)
    out = str(x.stdout).split(' seconds')[1]
    s = ''.join(out.splitlines())
    s = s.replace('/^\s+|\s+$/g', '');
    print(f'''The text synthesis took {str(round(float(str(datetime.now()-now).strip("00:")),2))} seconds \n
                                                            \n
{s[2:-3]}''')
    return f'''The text synthesis took {str(round(float(str(datetime.now()-now).strip("00:")),2))} seconds \n
                                                            \n
{s[2:-3]}'''


with gr.Blocks(css="#margin-top {margin-top: 15px} #center {text-align: center;} #description {text-align: center}") as demo:
    with gr.Row(elem_id="center"):
        gr.Markdown("# LLaMa Inference with Gradient")
    with gr.Row(elem_id = 'description'):
        gr.Markdown(""" To run LLaMA, be sure to select a model size that works for your machine. Single GPUs should always use '7B'.\n Start typing below and then click **Run to generate text** to see the output.""")
    with gr.Row():
        ckpt = gr.Radio(["7B", "13B", "30B", '65B'], label="Checkpoint directory", value = "30B", interactive = True)
        seed = gr.Slider(label = 'Seed', value = 8019, minimum = 1, maximum = 10000, step =1, interactive = True)
        prompts = gr.Textbox(label = 'Prompt input', placeholder="What is your prompt?", value = 'my new invention is the', interactive = True)
        
    btn = gr.Button("Run to generate text")
    with gr.Row():
        out = gr.Markdown()
    with gr.Row():
        gr.Image('assets/logo.png').style(height = 53, width = 125, interactive = False)

    btn.click(fn=run, inputs=[prompts, seed, ckpt], outputs=out)


demo.launch(share = True)

