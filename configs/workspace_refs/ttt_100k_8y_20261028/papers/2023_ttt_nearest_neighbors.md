# Test-Time Training on Nearest Neighbors for Large Language Models

- Authors: Moritz Hardt, Yu Sun
- Venue: ICLR 2024
- arXiv: 2305.18466 (first version 2023-05-29; this text from v3, 2024-02-02)
- Licence: arXiv non-exclusive distribution licence (http://arxiv.org/licenses/nonexclusive-distrib/1.0/)
- Text here: abstract only: the paper's licence does not permit redistributing its full text (arXiv metadata, abstracts included, is CC0)

## Abstract

Many recent efforts augment language models with retrieval, by adding retrieved data to the input context. For this approach to succeed, the retrieved data must be added at both training and test time. Moreover, as input length grows linearly with the size of retrieved data, cost in computation and memory grows quadratically for modern Transformers. To avoid these complications, we simply fine-tune the model on retrieved data at test time, using its standard training setup. We build a large-scale distributed index based on text embeddings of the Pile dataset. For each test input, our system retrieves its neighbors and fine-tunes the model on their text. Surprisingly, retrieving and training on as few as 20 neighbors, each for only one gradient iteration, drastically improves performance across more than 20 language modeling tasks in the Pile. For example, test-time training with nearest neighbors significantly narrows the performance gap between a small GPT-2 and a GPT-Neo model more than 10 times larger. Sufficient index quality and size, however, are necessary. Our work establishes a first baseline of test-time training for language modeling.
