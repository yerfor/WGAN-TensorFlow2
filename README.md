# WGAN-TensorFlow2
This sample, WGAN-TensorFlow2, implements a full Tensorflow2-based WGAN for image generation task. This sample is based on the [WGAN-gp](https://arxiv.org/pdf/1704.00028.pdf) paper.

## Prerequisites

1.  Tensorflow 2.0.0
- `conda install tensorflow-gpu=2.0.0`
2.  Opencv
    - `conda install -c menpo opencv`

## How to work

1. download  "Anime-Face-Dataset" , or your own dataset, and unpack it to `/datasets` directory.

2. open the terminal and run the `train.py`:

   - ```bash
     python train.py
     ```
## Performance
1. A visualization of Training Process is available [here](https://yerfor.github.io/2020/02/06/gan-01/WGAN-demo.gif).
2. After 10000 iterations:
![WGAN-demo](https://yerfor.github.io/2020/02/06/gan-01/10000.png)
