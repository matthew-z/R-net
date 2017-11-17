


**Work In Progress.**


An unofficial implementation of R-net in PyTorch.

Natural Language Computing Group, MSRA: R-NET: Machine Reading Comprehension with Self-matching Networks

See: https://www.microsoft.com/en-us/research/publication/mrc/



Python 3.5/3.6  and PyTorch 0.2


**Usage**

```
python main.py --batch_size 32

```

**Current Problems**

Char-Level embedding with RNN is too slow, so I removed it from my implementation.
I plan to try using CNN to do char-level embedding.
