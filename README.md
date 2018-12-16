An unofficial implementation of R-net in [PyTorch](https://github.com/pytorch/pytorch) and [AllenNLP](https://github.com/allenai/allennlp).

[Natural Language Computing Group, MSRA: R-NET: Machine Reading Comprehension with Self-matching Networks](https://www.microsoft.com/en-us/research/publication/mrc/)

However, I didn't reproduce the model described in this paper because some details are not very clear to me and the dynamic attention in self-matching requires too much memory. 

Thus, I implemented the variant of R-Net according to [HKUST-KnowComp/R-Net](https://github.com/HKUST-KnowComp/R-Net) (in Tensorflow).

The biggest difference between the original R-net and HKUST R-net is that:
* The original R-net does attention at each RNN step, which means the hidden state is also involved in the attention calculation. I call it dynamic attention.
* In HKUST R-Net Attentions in pair encoder and self-matching encoder are calculated before performing RNN.  I call it static attention.

Some details in [HKUST-KnowComp/R-Net](https://github.com/HKUST-KnowComp/R-Net) that improves performance:
* Question and Passage share the same GRU sentence encoder instead of using two GRU encoders respectively.
* The sentence encoder has three layers, but its output is the concat of the three layers instead of the output of the top layer.
* The GRUs in the pair encoder and the self-matching encoder have only one layer instead of three layers. 
* Variational dropouts are applied to (1) the inputs of RNNs (2) inputs of attentions 

Furthermore, this repo added ELMo and BERT word embeddings, which further improved the model's performance. 

### Dependency

* Python == 3.6
* [AllenNLP](https://github.com/allenai/allennlp) == 0.7.2
* PyTorch == 1.0



### Usage

```
git clone https://github.com/matthew-z/R-net.git
cd R-net
python main.py train configs/squad/r-net/hkust.jsonnet  // HKUST R-Net
```
Note that the batch size may be a bit too large for 11GB GPUs. Please try 64 in case of OOM Error by adding this arg: `-o '{"iterator.batch_size": 64}'`.

### Configuration

The models and hyperparameters are declared in `configs/`

* the HKUST R-Net: `configs/r-net/hkust.jsonnet` (79.4 F1)
  * +ELMo: `configs/r-net/hkust+elmo.jsonnet` (82.2 F1)
  * +BERT:  `configs/r-net/hkust+bert.jsonnet` (work in progress)
* the original R-Net: `configs/r-net/original.jsonnet`  (currently not workable)


### Performance

This implementation of HKUST R-Net can obtain 79.4 F1 and 70.5 EM on the validation set.
+ ELMo: 82.2 F1 and 74.4 EM.
+ BERT: work in progress.

<font color=red>Red: Training score</font>, <font color=green>Green: dev score</font>

<img src="img/f1.png" width="700"> 
<img src="img/em.png" width="700">

Note that validation score is higher than training because each validation has three acceptable answers, which makes validation easier than training. 

### Future Work

* Add ensemble training
* Add FP16 training

### Acknowledgement 

Thank  [HKUST-KnowComp/R-Net](https://github.com/HKUST-KnowComp/R-Net) for sharing their Tensorflow implementation of R-net. This repo is based on their work.