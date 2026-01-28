# Diffusion Transformer Module

A minimal Transformer encoder-decoder implementation in PyTorch for diffusion models. Designed to integrate with UNet-based image generation pipelines.  

This project demonstrates how to implement a Transformer using **PyTorch built-in modules** (`nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer`) for sequence modeling tasks in diffusion models.

---

## Features

- **Encoder & Decoder** implemented with PyTorch built-ins  
- **Positional Encoding** added manually to handle sequence order  
- **Dropout & Linear projection** for stable training and vocabulary output  
- Fully **batch-compatible** for training with image embeddings  
- Minimal dependencies — works without any custom `models` package  

---

---

## Transformer Architecture

### Encoder

- Embedding layer (`nn.Embedding`) → maps tokens to `d_model`-dim vectors  
- Positional encoding → preserves sequence order  
- Stacked `nn.TransformerEncoderLayer` layers  
- Output shape: `(seq_len, batch_size, d_model)`

### Decoder

- Embedding layer (`nn.Embedding`)  
- Positional encoding  
- Stacked `nn.TransformerDecoderLayer` layers with cross-attention to encoder output  
- Dropout + Linear projection → logits over vocabulary  

---

## Usage

### Install PyTorch

```bash
pip install torch



## References
PyTorch Transformer Documentation

Vaswani et al., Attention Is All You Need, 2017








## Note

This code is not complete and I am still learning if there is an issue please create a pull request so I can fix it

