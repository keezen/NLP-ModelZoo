# NLP-ModelZoo
An NLP model zoo implemented in PyTorch.


## Requirements

PyTorch: >= 0.4.0


## Models Included

- TextCNN

Basic CNN-based network for text classification.

[Yoon Kim. Convolutional Neural Networks for Sentence Classification. 2014.](https://arxiv.org/abs/1408.5882)

- ESIM

Classic model for semantic matching.

[Chen, et al. Enhanced LSTM for Natural Language Inference. 2017.](https://arxiv.org/abs/1609.06038)

- Dynamic LSTM

LSTM with variable length, which is lacked in PyTorch.

- Transformer

Pure attention-based encoder which is spotlight.

[Vaswani, et al. Attention Is All You Need. 2017.](https://arxiv.org/abs/1706.03762)

- Biaffine

Biaffine transformation for attention computation and pair classification, taken from <https://github.com/zysite/biaffine-parser>.

[Dozat, et al. Deep Biaffine Attention for Neural Dependency Parsing. 2016.](https://arxiv.org/abs/1611.01734)
