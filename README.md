# Generative Replay Inspired by Hippocampal Memory Indexing

Code for the paper "[Generative Replay Inspired by Hippocampal Memory Indexing for Continual Language Learning](https://aclanthology.org/2023.eacl-main.65)"
In The 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL2023)
by Aru Maekawa, Hidetaka Kamigaito, Kotaro Funakoshi, and Manabu Okumra.

This code is based on the open source code from "[LAnguage-MOdeling-for-Lifelong-Language-Learning (LAMOL)](https://github.com/chho33/LAMOL)". Most of the settings follow to theirs.

## Examples

### Pretraining:

```
./pretrain.sh
```

### Training:

```
./train.sh --seq_train_type hmi-lamol --tasks sst srl woz.en
```

### Test:

```
./test.sh --seq_train_type hmi-lamol --tasks sst srl woz.en
```

## Acknowledgements

- We use the open source code of [LAMOL](https://github.com/chho33/LAMOL) provided by Cheng-Hao Ho and Fan-Keng Sun.
- We use the language model offered by [transformers](https://github.com/huggingface/transformers), a state-of-the-art natural language processing models library by Thomas Wolf et al.
- The implementation of MAS follows [MAS-Memory-Aware-Synapses](https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses), the Memory Aware Synapses method implementation code by Aljundi R. et al.
- The implementation of GEM follows [GradientEpisodicMemory](https://github.com/facebookresearch/GradientEpisodicMemory), the Gradient Episodic Memory method implementation code by Lopez-Paz, David et al.
- The implementation of fp16 (`fp16.py`, `fp16util.py`) is from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), the ongoing research training transformer language models at scale by NVIDIA.
- Data format conversion refer to [decaNLP](https://github.com/salesforce/decaNLP), the Natural Language Decathlon: Multitask Learning as Question Answering implementation code by Bryan McCann et al.

## Citation

```
@inproceedings{maekawa-etal-2023-generative,
    title = "Generative Replay Inspired by Hippocampal Memory Indexing for Continual Language Learning",
    author = "Maekawa, Aru  and
              Kamigaito, Hidetaka  and
              Funakoshi, Kotaro  and
              Okumura, Manabu",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.65",
    pages = "930--942",
}

@inproceedings{
    sun2020lamal,
    title={{\{}LAMAL{\}}: {\{}LA{\}}nguage Modeling Is All You Need for Lifelong Language Learning},
    author={Fan-Keng Sun and Cheng-Hao Ho and Hung-Yi Lee},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=Skgxcn4YDS}
}
```
