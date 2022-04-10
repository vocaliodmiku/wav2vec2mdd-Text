# wav2vec2mdd-Text
This repository is an implementation of the paper "Text-Aware End-to-end Mispronunciation Detection and Diagnosis."

**Abstract**
In this paper, we present a gating strategy that assigns more importance to the relevant audio features while suppressing irrelevant text information. Moreover, given the transcriptions, we design an extra contrastive loss to reduce the gap between the learning objective of phoneme recognition and MDD.

## Installation

### Requirements

* Linux, CUDA>=11, GCC>=5.4
  
* Python>=3.8

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n w2vText python=3.8
    ```
    Then, activate the environment:
    ```bash
    conda activate w2vText
    ```
  
* PyTorch>=1.6.1 (following instructions [here](https://pytorch.org/))

    For example, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install soundfile editdistance
    ```
* Fairseq

    We design the network via the fairseq package. If you are familar with fairseq, you can check wav2vec model::wav2vec_sigmoid and criterion::ctc_constrast. Otherwise, you should install the modified version as following:
    ```bash
    cd fairseq && pip install --editable .
    ```
    Alterantive, we can install "viterbi" package to omit the complex install process of flashlight binding:
    ```bash
    cd viterbi && python setup.py install
    ```
## Usage
   Before use following script to train and test model, you should check the data path (see *.tsv files in data directory) and reference path.

### Training

    ```bash
    sh run.sh
    ```

### Inference

    ```bash
    sh test.sh && sh mdd.sh
    ```
    
   Our best model result are included in diretory experiment/result, you can check it directly run "sh mdd.sh", and if you have any question about it, please contact us. Thanks!
    
## Cite
If you find this work useful in your research, please consider citing:
```bibtex
```
