# LLaMA

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

## Run in a Free GPU powered Gradient Notebook

[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/gradient-ai/llama?machine=Free-GPU)

## Setup

In a conda env with pytorch / cuda available, run:

```
pip install -r requirements.txt
```

## Download

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

#### In Gradient

If you are working in a Gradient Notebook, then these models have been uploaded for you, and mounted to your Notebook automatically. The path to the model files from the `notebooks` working directory:

```
../datasets/llama/
```

## Inference

#### [Gradio App](https://gradio.app/)

To run the Gradio Application, run the following in the terminal or using line magic. The MP values will automatically be connected. Note that multi-gpu machines are likely necessary to run 13B (x2), 30B (x4), and 65B (x8) models.

```
python app.py
```

#### Original script

The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:

```
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE --tokenizer_path $TARGET_FOLDER/tokenizer.model --prompt <your prompt> --seed 42

```

Different models require different MP values:

| Model | MP  |
| ----- | --- |
| 7B    | 1   |
| 13B   | 2   |
| 33B   | 4   |
| 65B   | 8   |

## FAQ

- [1. The download.sh script doesn't work on default bash in MacOS X](FAQ.md#1)
- [2. Generations are bad!](FAQ.md#2)
- [3. CUDA Out of memory errors](FAQ.md#3)
- [4. Other languages](FAQ.md#4)

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## Model Card

See [MODEL_CARD.md](MODEL_CARD.md)

## License

See the [LICENSE](LICENSE) file.
