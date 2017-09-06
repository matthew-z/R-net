


**Work In Progress.**


An unofficial implementation of R-net in PyTorch.

Natural Language Computing Group, MSRA: R-NET: Machine Reading Comprehension with Self-matching Networks

See: https://www.microsoft.com/en-us/research/publication/mrc/



Python 3.5/3.6  and PyTorch 0.2


**Current Problems**

It seems it is impossible to train the model on single GPU because it is a lack of memory for a reasonable batch size.
The next stage is to modify the code to train the model with mutiple GPU (e.g. 1 GPU per layer)
