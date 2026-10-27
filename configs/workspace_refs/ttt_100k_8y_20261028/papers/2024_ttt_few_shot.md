# The Surprising Effectiveness of Test-Time Training for Few-Shot Learning

- Authors: Ekin Akyürek, Mehul Damani, Adam Zweiger, Linlu Qiu, Han Guo, Jyothish Pari, Yoon Kim, Jacob Andreas
- Venue: ICML 2025
- arXiv: 2411.07279 (first version 2024-11-11; this text from v2, 2025-03-25)
- Licence: arXiv non-exclusive distribution licence (http://arxiv.org/licenses/nonexclusive-distrib/1.0/)
- Text here: abstract only: the paper's licence does not permit redistributing its full text (arXiv metadata, abstracts included, is CC0)

## Abstract

Language models (LMs) have shown impressive performance on tasks within their training distribution, but often struggle with structurally novel tasks even when given a small number of in-context task examples. We investigate the effectiveness of test-time training (TTT) -- temporarily updating model parameters during inference using a loss derived from input data -- as a mechanism for improving LMs' reasoning and few-shot learning capabilities. On the Abstraction and Reasoning Corpus (ARC), performing TTT with in-context examples yields up to $6\times$ higher accuracy compared to fine-tuned baselines -- reaching $53.0\%$ on the public validation set with an 8B-parameter LM and $61.9\%$ when ensembled with program-synthesis methods, matching average human performance. On BIG-Bench Hard (BBH), TTT on in-context examples surpasses standard few-shot prompting in the $10$-shot setting by $7.3$ percentage points ($50.5\%$ to $57.8\%$). Our findings highlight the limitations of in-context learning for novel tasks and demonstrate the potential of test-time training to enhance language model adaptability.
