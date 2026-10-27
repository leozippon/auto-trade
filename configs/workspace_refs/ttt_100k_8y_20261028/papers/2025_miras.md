# It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization

- Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
- Venue: arXiv preprint
- arXiv: 2504.13173 (first version 2025-04-17; this text from v1, 2025-04-17)
- Licence: CC BY 4.0 (http://creativecommons.org/licenses/by/4.0/)
- Text here: arXiv HTML, full text without references

## Abstract

Designing efficient and effective architectural backbones has been in the core of research efforts to enhance the capability of foundation models. Inspired by the human cognitive phenomenon of attentional bias-the natural tendency to prioritize certain events or stimuli-we reconceptualize neural architectures, including Transformers, Titans, and modern linear recurrent neural networks as associative memory modules that learn a mapping of keys and values using an internal objective, referred to as attentional bias. Surprisingly, we observed that most existing sequence models leverage either (1) dot-product similarity, or (2) L2 regression objectives as their attentional bias. Going beyond these objectives, we present a set of alternative attentional bias configurations along with their effective approximations to stabilize their training procedure. We then reinterpret forgetting mechanisms in modern deep learning architectures as a form of retention regularization, providing a novel set of forget gates for sequence models. Building upon these insights, we present Miras, a general framework to design deep learning architectures based on four choices of: (i) associative memory architecture, (ii) attentional bias objective, (iii) retention gate, and (iv) memory learning algorithm. We present three novel sequence models-Moneta, Yaad, and Memora-that go beyond the power of existing linear RNNs while maintaining a fast parallelizable training process. Our experiments show different design choices in Miras yield models with varying strengths. For example, certain instances of Miras achieve exceptional performance in special tasks such as language modeling, commonsense reasoning, and recall intensive tasks, even outperforming Transformers and other modern linear recurrent models.

## Body

### 1 Introduction

Designing efficient architectural backbones for sequence modeling is a key to enhance the capability of foundation models in domains ranging from language ( Vaswani et al., 2017 ; Behrouz et al., 2024b ) and computer vision ( Dosovitskiy et al., 2020 ) to computational biology ( Wang et al., 2024 ) and neuroscience ( Behrouz et al., 2024 ) . While Transformers ( Vaswani et al., 2017 ) , mainly due to their in-context learning and ability to learn at scale ( Kaplan et al., 2020 ) , have been firmly established as state-of-the-art (SOTA) models in sequence modeling, their quadratic time and space complexity limits their applicability in tasks that require long context modeling ( Dalal et al., 2025 ; Liu et al., 2024a ; Li et al., 2024 ) .

Recent efforts aim to overcome Transformer limitations in long-context modeling by designing efficient recurrent alternatives ( Neil et al., 2017 ; Smith et al., 2022 ; Behrouz et al., 2024b ) . Unlike Transformer’s linearly growing memory (i.e., the KV cache), these models compress the context into a fixed size memory, demanding improved memory management for comparable performance. To design more effective architectures, studies focus on improving memory capacity and its management by using/designing more expressive: (1) Learning rules: from Hebbian rule ( Hebb, 2005 ) to Delta rule ( Neil et al., 2017 ) ; (2) Forget gates: from LSTM’s ( Schmidhuber & Hochreiter, 1997 ) to Mamba2’s ( Dao & Gu, 2024 ) and then Titan’s forget gates ( Behrouz et al., 2024b ) ; and (3) More expressive memory architectures: from vector-valued memory in RetNet ( Sun et al., 2023 ) and LRU ( Orvieto et al., 2023 ) to neural deep memory in Titans ( Behrouz et al., 2024b ) and TTT ( Sun et al., 2024 ) .

At the core of these advancements lies a critical question: “what is the underlying design framework behind these sequence models, and how can these models be enhanced?”. Taking inspiration from the broad definitions of associative memory and learning in neuropsychology literature ( Okano et al., 2000 ) , several studies discuss the connection between Transformers and (linear) Recurrent Neural Networks (RNNs) with associative memory ( Hopfield, 1982 ; Ramsauer et al., 2021 ; Bietti et al., 2023 ) . These studies, however, either: (1) lack a universal explanation to fully illustrate the underlying learning algorithms, (2) are limited to a specific definition of associative memory and lack generalizability, and/or (3) are unable to describe standard, widely used components such as forget gate.

Contributions. Inspired by the human cognitive phenomenon of attentional bias—the natural tendency to prioritize certain events or stimuli—we re-conceptualize neural architectures, including Transformers, Titans, and other modern linear recurrent neural networks based on a broad definition of associative memory with attentional bias . We define and formalize the concept of attentional bias as the internal memory objective of sequence models (see Section 3 ) that aims to learn the underlying mapping between inputs (i.e., keys and values). Our formulation reveals that almost all existing sequence models are associative memories that leverage the same type of attentional bias. We reinterpret existing forgetting mechanisms in modern deep learning architectures as a form of retention $\ell_{2}$ -regularization for the attentional bias, and then provide a novel set of alternative retention gates (forget gate) for sequence models, providing new insights on how to balance learning new concepts and the retention of previously learned concepts.

Building upon our formulation of memory and forget gate, we present Miras [note: “Miras” is the translation of “Legacy” in several languages: such as Persian, Arabic, and Turkish. We choose this name since this framework provides clear steps for future studies to design powerful sequence models based on their task at hand.] , a fundamental framework to design novel sequence modeling architectures by four choice of: (1) Attentional bias (i.e., memory objective), (2) Retention gate, (3) Memory architecture, and (4) Memory learning algorithm (i.e., optimizer). We motivate and discuss several novel design choices, leading to novel architectures beyond existing sequence modeling architectures.

Finally, we focus on three novel variants of Miras — Moneta , Yaad , and Memora —that are based on attentional biases beyond simple $\ell_{2}$ -regression objective as well as novel retention gating mechanisms that are more robust than existing ones. We further perform experimental evaluations of these three variants on language modeling, common-sense reasoning, needle-in-haystack, and recall intensive tasks. The results illustrates the superior performance of these variants, outperforming state-of-the-art sequence models.

Roadmap. In Section 2 , we review literature and discuss relevant concepts that we use through the paper. In Section 3 , we present and discuss the broad definition of associative memory with formally defining the concept of attentional bias . We then discuss two viewpoints—Learning-Retaining and Follow-the-Regularized-Leader (FTRL)—to interpret sequence modeling through the lens of optimization and prove the generality of Learning-Retaining over FTRL. In Section 4 , we present our Miras framework and discuss how it unifies modern sequence models. In Section 5 , to show the potential of Miras framework, we discuss a variety of novel design choices for (1) attentional bias, and (2) retention gate (forget gate). Later in Section 5.3 , we present three novel sequence models as the variants of Miras , and then discuss how to train them in a parallelizable manner. Finally, our experimental evaluations are reported in Section 6 .

[Figure] Figure 1 : The overview of Miras framework. Miras is based on four critical choices of (1) memory architecture, (2) attentional bias, (3) retention gate, and (4) memory learning algorithm. In this framework, the memory architecture determines the model capacity to memorize; attentional bias is responsible for modeling the underlying mapping patterns; retention gate determines how to balance learning new concepts and the retention of previously learned concepts; and memory learning algorithm is responsible for memory management.

Table: Table 1 : Overview of recent sequence models in Miras framework perspective. Surprisingly, all models are using the same type of attentional bias and regularization (forget gate). Note that these architectural choices does not uniquely identify the backbone as there are other design choices (e.g., input-dependency, channel-wise parameters, etc.) as well as the use of other components such as attention, convolutions, etc. Note that for attentional bias and retention gate, we are referring to the original design of Miras , discussed in Equation 4 and Remark 1 .

| Model | Memory | Attentional Bias | Retention | Memory | Memory Write Operation |
| Architecture | Gate † | Algorithm |
| Shallow Memory |
| RetNet ( 2023 ) | Vector | Dot-Product | L2 | GD | $\mathcal{M}_{t}=\alpha\mathcal{M}_{t-1}+{\mathbf{v}_{t}\mathbf{k}_{t}^{\top}}$ |
| Transformer ( 2017 ) | Matrix | L2 | - | Nonparametric | $\mathcal{M}_{t}=\mathcal{M}_{t-1}\cup\{(\mathbf{k}_{t},\mathbf{v}_{t})\}$ |
| LA ( 2021 ) | Matrix | Dot-Product | - | GD | $\mathcal{M}_{t}=\mathcal{M}_{t-1}+\mathbf{v}_{t}\mathbf{k}_{t}^{\top}$ |
| DFW | Matrix | Dot-Product | L2 | GD | $\mathcal{M}_{t}=\left(\beta_{t}\alpha_{t}^{\top}\right)\odot\mathcal{M}_{t-1}+\mathbf{v}_{t}\mathbf{k}_{t}^{\top}$ |
| Lightening Attention ( 2025 ) | Matrix | Dot-Product | L2 | GD | $\mathcal{M}_{t}=\alpha\mathcal{M}_{t-1}+\mathbf{v}_{t}\mathbf{k}_{t}^{\top}$ |
| GLA ( 2024a ) | Matrix | Dot-Product | L2 | GD | $\mathcal{M}_{t}=\text{Diag}(\alpha_{t})\mathcal{M}_{t-1}+\mathbf{v}_{t}\mathbf{k}_{t}^{\top}$ |
| Mamba ( 2024 ) | Matrix | Dot-Product | L2 | GD | $\mathcal{M}_{t}=\alpha_{t}\mathcal{M}_{t-1}+\mathbf{v}_{t}\mathbf{k}_{t}^{\top}$ |
| HGRN2 ( 2024 ) | Matrix | L1 | L2 | GD | $\mathcal{M}_{t}=\text{Diag}(\alpha_{t})\mathcal{M}_{t-1}+\mathbf{v}_{t}(\mathbf{1}-\alpha_{t})^{\top}$ |
| DeltaNet ( 2017 ) | Matrix | L2 | - | GD | $\mathcal{M}_{t}=(\mathbf{I}-\beta_{t}\mathbf{k}_{t}\mathbf{k}_{t}^{\top})\mathcal{M}_{t-1}+\beta_{t}\mathbf{v}_{t}\mathbf{k}_{t}^{\top}$ |
| Longhorn ( 2024 ) | Matrix | L2 | - | Implicit GD | $\mathcal{M}_{t}=\left(\mathbf{I}-\frac{\beta_{t}\mathbf{k}_{t}\mathbf{k}^{\top}}{\mathbf{1}+\beta_{t}\mathbf{k}_{t}^{\top}\mathbf{k}_{t}}\right)\mathcal{M}_{t-1}+\left(\frac{\beta_{t}}{\mathbf{1}+\mathbf{k}_{t}^{\top}\mathbf{k}_{t}\beta_{t}}\odot\mathbf{x}_{t}\right)\mathbf{k}_{t}$ |
| TTT-Linear ( 2024 ) | Matrix | L2 | - | GD | $\mathcal{M}_{t}=\mathcal{M}_{t-1}-\eta\nabla\mathcal{L}(\mathcal{M}_{t-1},\mathbf{x}_{t})$ |
| Gated DeltaNet ( 2024 ) | Matrix | L2 | L2 | GD | $\mathcal{M}_{t}=\left(\alpha_{t}(\mathbf{I}-\beta_{t}\mathbf{k}_{t}\mathbf{k}_{t}^{\top})\right)\mathcal{M}_{t-1}+\beta_{t}\mathbf{v}_{t}\mathbf{k}_{t}^{\top}$ |
| RWKV-7 ( 2025a ) | Matrix | L2 | L2 | GD | $\mathcal{M}_{t}=\text{diag}(\alpha_{t})\left(\mathbf{I}-\beta_{t}\mathbf{k}_{t}\mathbf{k}_{t}^{\top}\right)\mathcal{M}_{t-1}+\beta_{t}\mathbf{v}_{t}\mathbf{k}_{t}^{\top}$ |
| DeltaProduct ( 2025 ) | Matrix | L2 | L2 | MGD ∗ | $\mathcal{M}_{t}=\left(\alpha_{t}\prod_{i=1}^{n}(\mathbf{I}-\beta_{t,i}\mathbf{k}_{t,i}\mathbf{k}_{t,i}^{\top})\right)\mathcal{M}_{t-1}+\sum_{j=1}^{n}\prod_{i=j}^{n}(\mathbf{I}-\beta_{t,i}\mathbf{v}_{j,i}\mathbf{k}_{j,i}^{\top})$ |
| Deep Memory |
| TTT-MLP ( 2024 ) | 2-layer MLP | L2 | - | GD | $\mathcal{M}_{t}=\mathcal{M}_{t-1}-\eta\nabla\mathcal{L}(\mathcal{M}_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})$ |
| Titans-LMM ( 2024b ) | $k$ -layer MLP | L2 | L2 | GD + Momentum | $\mathcal{M}_{t}=\alpha_{t}\mathcal{M}_{t-1}-\mathcal{S}_{t},\>\>\>\text{where}\>\>\mathcal{S}_{t}=\eta_{t}\mathcal{S}_{t-1}-\theta_{t}\nabla\mathcal{L}(\mathcal{M}_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})$ |
| Moneta ( ours ) | $2$ -layer MLP | L p | L q | GD | $A_{t}=\alpha_{t}A_{t-1}-\eta_{t}\nabla\ell_{p}(W_{i-1};\mathbf{k}_{t},\mathbf{v}_{t}),W_{t}=\frac{A_{t}}{\|A_{t}\|_{q}^{q-2}}$ |
| Yaad ( ours ) | $2$ -layer MLP | Huber | L2 | GD | $W_{t}=\alpha_{t}W_{t-1}-\begin{cases}\eta_{t}\>\nabla\ell_{2}(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})&\text{if}\quad\|\mathcal{M}(\mathbf{k}_{t})-\mathbf{v}_{t}\|\leq\delta_{t},\\ \eta_{t}\>\delta_{t}\nabla\ell_{1}(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})&\text{Otherwise}.\end{cases}$ |
| Memora ( ours ) | $2$ -layer MLP | L2 | KL | GD | $W_{t}=\mbox{Softmax}\left(\alpha_{t}\log(W_{t-1})-\eta_{t}\nabla\ell_{2}(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\right)$ |
| ∗ is using multiple rounds of GD per token. |
| † For the sake of clarity, we use L2 for all modified L2-like regularizations. However, in fact, only Titans and RWKV-7 are using L2 retention gate (see Section 4 ) |

### 2 Preliminaries and Background

In this section, we review the related studies and background concepts that we use through the paper.

Attention. Attention as the backbone of Transformers is a critical component that acts as their associative memory ( Bietti et al., 2023 ) . Given input $x\in\mathbb{R}^{N\times d_{\text{in}}}$ , causal attention computes output $\mathbf{y}\in\mathbb{R}^{N\times d_{\text{in}}}$ based on Softmax over input dependent key, value, and query matrices:

$\displaystyle\mathbf{Q}=x\mathbf{W}_{\mathbf{Q}},\qquad\mathbf{K}=x\mathbf{W}_{\mathbf{K}},\qquad\mathbf{V}=x\mathbf{W}_{\mathbf{V}},$ (1) $\displaystyle\mathbf{y}_{i}=\sum_{j=1}^{i}\frac{\exp\left(\mathbf{q}_{i}^{\top}\mathbf{k}_{j}/\sqrt{d_{\text{in}}}\right)\mathbf{v}_{j}}{\sum_{\ell=1}^{i}\exp\left(\mathbf{q}_{i}^{\top}\mathbf{k}_{\ell}/\sqrt{d_{\text{in}}}\right)},$ (2)

where $\mathbf{W}_{\mathbf{Q}},\mathbf{W}_{\mathbf{K}},$ and $\mathbf{W}_{\mathbf{V}}\in\mathbb{R}^{d_{\text{in}}\times d_{\text{in}}}$ are learnable parameters. While Transformers achieve significant improvements compared to traditional Recurrent Neural Networks (RNNs)—such as LSTM ( Schmidhuber & Hochreiter, 1997 ) , their complexity that requires at least $N\times d$ operators to calculate the output has been the main motivation for researchers to think about alternative architectures. We divide and review the research efforts to design alternative architectures into two groups: (1) Linear shallow memory recurrent models, (2) Deep memory modules.

(Linear) Recurrent Models. For many years, non-linear (gated) recurrent neural networks had been the de facto architectural backbones in deep learning ( Greff et al., 2016 ) . Their recurrent nature, however, results in non-parallelizable training, making their large scale training infeasible. To this end, in recent years, linear RNNs as alternatives to both Transformers and non-linear RNNs attract much attention mainly due to their parallelizable and linear-time training while maintaining competitive performance ( Yang et al., 2024b ; Sun et al., 2023 ; Peng et al., 2025 ) . Earlier variants of linear RNNs ( Yang et al., 2024a ; Sun et al., 2023 ; De et al., 2024 ) , which mostly are based on Hebbian learning rule ( Hebb, 2005 ) , aim to compress the data into their vector-valued (or matrix-valued) memory ( Katharopoulos et al., 2020 ; Sun et al., 2023 ; Yang et al., 2024a ; De et al., 2024 ; Liu et al., 2024 ) . Let $\mathcal{M}_{t}\in\mathbb{R}^{d\times n}$ be the memory ( $n=1$ means vector-valued memory), and $\mathbf{k},\mathbf{v}\in\mathbb{R}^{d}$ are keys and values (i.e., projection of input $x_{t}\in\mathbb{R}^{d}$ ), a simple general formulation for such linear RNNs can be written as:

$\displaystyle\mathcal{M}_{t}=A_{t}\ast\mathcal{M}_{t-1}+\mathbf{v}_{t}\mathbf{k}_{t}^{\top},$ (3)

where $\ast$ is an arbitrary associative operator and $A_{t}$ is a data-(in)dependent diagonal matrix or a scalar ( Yang et al., 2024b ) . Despite the efficiency that comes with the linear recurrent nature of these models, the memory can overflow mainly due to the additive (without replacement) nature of Hebbian learning rule, resulting in limited memory capacity and limited expressive power in in-context learning tasks. Moreover, the vector-valued memory of these architectures can limited their ability to learn/memorize large context window, mainly due to the limited expressive power of memory to learn the underlying patterns of data ( Behrouz et al., 2024b ; Sun et al., 2024 ) .

To address the above mentioned limitations, recurrent models that use a matrix-valued memory with Delta learning rule has gained popularity in recent years ( Neil et al., 2017 ; Schlag et al., 2021 ; Yang et al., 2024b ) . Despite significant advantages, even these delta-rule-based recurrent models face theoretical limitations ( Irie et al., 2023 ) with moderate performance in practice ( Yang et al., 2024b ) . Recently, several studies aim to improve the performance of such models by adding scalar or channel-wise forget gate mechanisms ( Yang et al., 2024 ; Peng et al., 2025a ) , , using negative eigenvalues ( Grazzi et al., 2024 ) , and multiple learning steps ( Siems et al., 2025 ) . They, however, still suffer from performance drop in long context, mainly due to the less expressive memory architectures ( Behrouz et al., 2024b ) .

Deep Memory Module: Titans and Test Time Training. To overcome the limited memory and to extend the effective context length of deep sequence models, more recent studies focus on a new generation of architectures with deep memory module ( Behrouz et al., 2024b ; Sun et al., 2024 ) . These architectures are built on the meta-learning perspective, where the memory is an MLP architecture that is updated using gradient descent (with momentum) ( Behrouz et al., 2024b ; Sun et al., 2024 ) . Sun et al. (2024) further provide a unifying perspective that how linear and softmax attention are respectively parametric and non-parameteric solutions of (kernel) regression loss but consider other modern linear RNNs outside of this class of models. Recently, in a concurrent work to ours, Wang et al. (2025) show that with additional simplification of modern RNNs (e.g., RetNet ( Sun et al., 2023 ) , Mamba ( Dao & Gu, 2024 ) ) they approximately place in the same class of models that internally optimize regression loss. It, however, still remains unanswered that “What is the underlying design framework behind these sequence models that can accurately unify existing architectures?” Moreover, the role of forget gates and its alternative choices in modern sequence models is surprisingly less explored.

### 3 Associative Memory, Attentional Bias, and Retention

Associative memory, which is an inseparable component of learning in humans ( Terry, 2017 ) , has been the inspiration for many artificial neural architectures in the literature ( Hopfield, 1982 ; Neil et al., 2017 ; Behrouz et al., 2024b ) . These studies, however, define instances of the concept of associative memory, limiting the architecture to a specific class of similarity metrics between entities (i.e., keys and values). That is, broadly speaking, associative memory is an operator that maps a set of keys $K$ to a set of values $V$ , and so to learn the underlying mapping patterns in data, it requires an objective that targets a type of memory and measures the quality of learned mappings:

###### Definition 3.1 (Associative Memory and Attentional Bias) .

Given a set of keys $\mathcal{K}\subseteq\mathbb{R}^{d_{k}}$ and values $\mathcal{V}\subseteq\mathbb{R}^{d_{v}}$ , associative memory is an operator $\mathcal{M}:\mathcal{K}\rightarrow\mathcal{V}$ . Learning the mapping of associative memory is based on an objective $\mathcal{L}$ , called Attentional Bias , that determines the type of memory and its tendency to prioritize some events:

$\displaystyle\mathcal{M}^{*}=\arg\min_{\mathcal{M}}\quad\mathcal{L}(\mathcal{M}(\mathcal{K});\mathcal{V}).$ (4)

A few remarks are in order:

###### Remark 1 .

When we parameterize the memory with parameter $W$ , we use $\mathcal{M}(W,\mathbf{k})$ . In this parameteric setting, the optimization problem in ( 4 ) should be performed over the parameter $W$ . Furthermore, in the parametric setup, we might use an additional regularization $\mathcal{R}(W)$ to control the retaining of the past data.

###### Remark 2 .

Learning the mapping between keys and values ( Equation 4 ) is a meta-learning problem, in which the attentional bias is optimized in the inner-loop and all other parameters of the neural network (e.g., linear projections, convolutions, etc.) are optimized in the outer-loop. Therefore, the model learns how to store the data into its parameters at test time ( Sun et al., 2024 ; Behrouz et al., 2024b ) .

#### 3.1 Learning to Memorize and to Retain Through the Lens of Optimization

3.1 translates the design of a neural architecture based on the concept of associative memory to learning the underlying mapping between keys and values, by minimizing an objective $\mathcal{L}$ . To optimize Equation 4 , one simple approach is to utilize the idea of gradient descent. Specifically, given a new pair of keys and values, we update the memory as:

$$W_{t}=W_{t-1}-\eta_{t}\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t}),$$ (5)

where, for simplicity, we use the definition $\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t}):=\mathcal{L}(\mathcal{M}(W;\mathbf{k}_{t}),\mathbf{v}_{t}$ ). Behrouz et al. (2024b) re-interpreter this formulation as a momentary surprise metric, where the model memorizes tokens that violates the expectation of the objective (i.e., being surprising to the memory). Although the choice of objective is an important step to fully interpret Equation 5 (which we discuss in detail in Section 5 ), there are different viewpoints to interpret this update rule in its general format, which later can help us to go beyond existing architectures:

#### 3.2 Viewpoint 1: Online Regression and Follow-The-Regularized-Leader

Equation ( 5 ) can be viewed as one step of online gradient descent over the sequence of the loss functions

$\displaystyle\ell(W;\mathbf{k}_{1},\mathbf{v}_{1}),\ell(W;\mathbf{k}_{2},\mathbf{v}_{2}),\ldots,\ell(W;\mathbf{k}_{t},\mathbf{v}_{t}),\ldots.$ (6)

It is well known that the online gradient descent can be viewed as a special case of Follow-The-Regularized-Leader (FTRL) algorithm with a special choice of loss functions ( Shalev-Shwartz, 2012 , Chapter 2) and ( Hazan, 2016 ) . Specifically, assuming $W_{0}=0$ , the update rule in ( 5 ) is equivalent to

$$W_{t}=\arg\min_{W}\quad\sum_{i=1}^{t}\langle W-W_{i-1},\nabla\ell(W_{i-1};\mathbf{k}_{i},\mathbf{v}_{i})\rangle+\frac{1}{2\eta}\|W\|_{2}^{2},$$ (7)

where the term $\langle W-W_{i-1},\nabla\ell(W_{i-1};\mathbf{k}_{i},\mathbf{v}_{i})\rangle$ is the local linear approximation of the original loss at time $i$ and the second term is a regularization term. While the first part $\sum_{i=1}^{t}\langle W-W_{i-1},\nabla\ell(W_{i-1};\mathbf{k}_{i},\mathbf{v}_{i})\rangle$ measures how well can the memory learn all the past tokens, the second term $\frac{1}{2\eta}\|W\|_{2}^{2}$ penalizes the memory update with respect to the size of memory.

Equation ( 7 ) uses linear approximation of the loss function and quadratic regularization. We can, however, in principle use other approximations of the loss function as well as other regularization functions, as used in the past in online optimization ( Shalev-Shwartz, 2012 ; Hazan, 2016 ) or in general optimization ( Mairal, 2015 ; Razaviyayn et al., 2013 ) . Such changes are the idea behind the development of other optimization algorithms such mirror descent. More specifically, we can generalize the update rule in ( 7 ) to the form:

$$W_{t}=\arg\min_{W\in\mathcal{W}}\quad\underset{\text{Attentional Bias}}{\underbrace{\sum_{i=1}^{t}\widehat{\ell}_{i}(W;\mathbf{k}_{i},\mathbf{v}_{i})}}+\underset{\text{Memory Stability}}{\underbrace{\frac{1}{\eta_{t}}\mathcal{R}_{t}(W)}}.$$ (FTRL Viewpoint)

In this update rule, the term $\sum_{i=1}^{t}\widehat{\ell}_{i}(W;\mathbf{k}_{i},\mathbf{v}_{i})$ aims at memorizing the tokens at test time, while the term $\mathcal{R}_{t}(W)$ regularizes the learning dynamics and take the size of the memory into account when updating it by a new incoming data. Choosing different loss functions $\widehat{\ell}_{i}(W;x_{i})$ and the regularization term $\frac{1}{\eta_{t}}\mathcal{R}_{t}(W)$ can lead to different algorithms such as (online) gradient descent or mirror descent. In this generalization, $\eta_{t}$ to can be data-dependent. Moreover, we will allow imposing constraint $\mathcal{W}$ on the choice $W$ .

#### 3.3 Viewpoint 2: Learning the Latest Token While Retaining Previous Information

Another way to interpret the update rule ( 5 ) is to view it as learning from the latest key-value pair $(\mathbf{k}_{i},\mathbf{v}_{i})$ (via using its gradient or surprise metric), while staying close to the previous state $W_{t-1}$ to retain the previously memorized tokens. Formally, ( 5 ) is equivalent to

$$W_{t}=\arg\min_{W}\quad\langle W-W_{t-1},\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\rangle+\frac{1}{2\eta_{t}}\|W-W_{t-1}\|_{2}^{2}$$

The first term locally approximates $\ell(W;\mathbf{k}_{t},\mathbf{v}_{t})$ around the previous state $W_{t-1}$ , while the last term regularizes deviations from $W_{t-1}$ . This form can generalize to

$\displaystyle W_{t}=\arg\min_{W\in\mathcal{W}}\>\>\underset{\text{Attentional Bias}}{\underbrace{\widetilde{\ell}_{t}(W;\mathbf{k}_{t},\mathbf{v}_{t})}}+\underset{\text{Retention}}{\underbrace{\mbox{Ret}_{t}\left(W,W_{t-1}\right)}},$ (Learning-Retaining Viewpoint)

where the term $\widetilde{\ell}_{t}(W;\mathbf{k}_{t},\mathbf{v}_{t})$ is an approximation of $\ell(W;\mathbf{k}_{t},\mathbf{v}_{t})$ and minimizing it corresponds to Learning from the new concepts $(\mathbf{k}_{t},\mathbf{v}_{t})$ . The second term $\mbox{Ret}_{t}\left(W,W_{t-1}\right)$ regularizes the changes in $W$ to make the learning dynamics stable and to retain previously learned knowledge. This Retention function may have local and global components:

$$\mbox{Ret}_{t}\left(W,W_{t-1}\right)=\underset{\text{Local Retention}}{\underbrace{\frac{1}{\eta_{t}}\mbox{D}_{t}\left(W,W_{t-1}\right)}}+\underset{\text{Global Retention}}{\underbrace{\frac{1}{\alpha_{t}}\mbox{G}_{t}\left(W\right)}}.$$

Here, the term $\mbox{D}_{t}\left(W,W_{t-1}\right)$ , which is a premetric that controls the deviations from $W_{t-1}$ , aims at retaining previously learned knowledge. The coefficient $\eta_{t}$ can be viewed as a meta in-context learning rate, where larger values of $\eta_{t}$ leads to learning more from new concepts, while allowing higher forgetting of previously learned concepts. The second term is a global retention that controls the change of the memory with respect to its size. The special instances of the above viewpoint (e.g., without global retention, with implicit closed-form solution, and/or with limited memory structure) have been the motivation behind some of the recent studies such as Liu et al. (2024) .

#### 3.4 Further Discussions on the Two Viewpoints

The ( FTRL Viewpoint ) and ( Learning-Retaining Viewpoint ) are connected through the lens of online optimization. For example, as discussed above, by choosing linear approximation of the loss and quadratic regularization/retention, they can both cover online gradient descent update in ( 5 ) as a special case. One straightforward way to make the connection explicit is by defining the premetric $\mbox{D}_{t}(W;W^{\prime})$ based on the previous loss functions and the regularization, as described in Proposition 3.2 below:

###### Proposition 3.2 .

Let $\eta_{t}=\eta$ and define $h_{t}(W):=\sum_{i=1}^{t-1}\widehat{\ell}_{i}(W;\mathbf{k}_{i},\mathbf{v}_{i})+\frac{1}{\eta}R(W)$ . Assume $\mathcal{W}=\mathbb{R}^{d}$ and the function $h_{t}(W)$ is strictly convex in $W$ and let $\mathcal{D}_{h}(\cdot,\cdot)$ be the Bregman divergence defined by function $h(\cdot)$ , i.e., $\mathcal{D}_{h}(W,W^{\prime})=h(W)-h(W^{\prime})-\langle\nabla h(W^{\prime}),W-W^{\prime}\rangle$ . Set $\mbox{Ret}_{t}(W,W^{\prime})=\mathcal{D}_{h}(W,W^{\prime})$ and $\widetilde{\ell}_{t}(W;x_{t})=\widehat{\ell}_{t}(W;x_{t})$ in ( Learning-Retaining Viewpoint ). Then, the update rule in ( Learning-Retaining Viewpoint ) is equivalent to the update rule in ( FTRL Viewpoint ).

We provide the proof in Appendix B . The above proposition shows that ( Learning-Retaining Viewpoint ) can also explain the approaches obtained by ( FTRL Viewpoint ), under some mild assumptions. Hence, ( Learning-Retaining Viewpoint ) may be seen as a more general version. This is why we focus on this viewpoint in most of our derivations in the next sections.

###### Remark 3 .

Given the above viewpoint, we can see that even by using additional global regularization there is no memory erasing or forgetting process (a common term in modern architectures ( Behrouz et al., 2024b ; Yang et al., 2024 ) ) but the model might decide to not retain the past state of the memory. Interestingly, this observation also matches the human memory process, where brain does not erase memories but they might become inaccessible due to retrieval failures ( Robertson, 2002 ) . Therefore, instead of calling it a forget gate, later on, we use “Retention Gate” to refer to this term.

###### Remark 4 .

As we discuss in Section 4 and summarize in Table 1 , most existing modern sequence models are optimizing associative memory objective (attentional bias in Equation 4 ) using gradient descent. Therefore, to provide further intuition about the connection of existing sequence models as well as their online learning interpretations, we discuss the above two viewpoints that are limited to gradient descent-based update rules. Our initial definition of attentional bias and associative memory in Equation 4 , however, is broader and can be optimized by any optimization algorithm (e.g., even Newton’s method, or non-parametric solutions).

### 4 Miras : Learning to Memorize with Robust and Expressive Memory

Building upon our definition of associative memory, attentional bias, and previous viewpoints, we present Miras framework that not only accurately unifies existing backbone architectures but it also provides insights on how to design the next generation of sequence models. As discussed earlier in Section 3 , learning an associative memory can be interpreted as a meta-learning task, in which the associative memory learns how to compress and store data into its parameters at test time. The architecture of the memory in such tasks is particularly important as in longer contexts, the expressivity of the memory structure can limit its ability to learn the underlying patterns. Therefore, the first choice to design a sequence model is the structure of the memory. Given the structure of the memory, parameterized by a set of parameters $W$ , as discussed earlier, we aim to minimize a loss function $\ell(W;\cdot,\cdot)$ with a retention regularizer $\mbox{Ret}(\cdot)$ via a learning algorithm (e.g., gradient descent). Accordingly, Miras requires four design choices:

- 1. Memory Structure: This choice specifies the architecture of the memory. For example, this architecture can be a vector, a linear function, a Multilayer Perceptron (MLP) layer, or even more complex structures. We may restrict the choice of $W$ to be within a certain region, e.g., $W$ to lie within an $L_{2}$ ball to avoid infinite values or unstable training.

- 2. Attentional Bias: A key choice is the attentional bias objective $\mathcal{L}(\cdot)$ in Equation 4 . We can even consider different approximations of the loss function, (e.g., $\widehat{\ell}(\cdot,\cdot)$ in ( FTRL Viewpoint ) or $\widetilde{\ell}(\cdot,\cdot)$ in ( Learning-Retaining Viewpoint )). The choice of attentional bias determines how memory memorizes the context, maps the inputs, and prioritizes the events.

- 3. Memory Stability and Retention: Another key choice is the retention regularizer $\mathcal{R}(\cdot)$ (e.g., $\mathcal{R}_{t}(\cdot)$ in ( FTRL Viewpoint ) and $\mbox{Ret}_{t}(\cdot)$ in ( Learning-Retaining Viewpoint )). In parametric setups, this choice balances learning with retention of past state. An effective retention gate is key to the good performance in long context tasks.

- 4. Memory Algorithm: Finally, this choice specifies the learning algorithm that we use to optimize the memory objective. One may use gradient descent, gradient descent with momentum, or any other algorithm (including finding non-parametric solutions).

The above choices are major design choices for designing backbone sequence models in neural architectures. There are, however, minor decisions that can distinguish models; i.e., data-dependent or independent parameters, scalar or channel-wise learning rate/retaining gate, etc. Next, we discuss the overview of how existing architectures fit into Miras framework.

RNNs with Hebbian Rule. The first generation of modern recurrent architectures (e.g., Linear attention ( Katharopoulos et al., 2020 ) , RetNet ( Sun et al., 2023 ) , Mamba ( Gu & Dao, 2024 ) , and GLA ( Yang et al., 2024a ) ) are based on Hebbian-like (e.g., gated Hebbian) learning rule ( Hebb, 2005 ) . We let attentional bias be the dot product similarity. That is, given a memory $\mathcal{M}\in\mathbb{R}^{d\times n}$ and $\mathbf{k},\mathbf{v}\in\mathbb{R}^{d}$ , we define $\tilde{\ell}_{t}:=-2\langle\mathcal{M}_{t}\mathbf{k}_{t},\mathbf{v}_{t}\rangle$ and local retention as $\text{Ret}_{t}(\mathcal{M},\mathcal{M}_{t-1})=\|\mathcal{M}_{t}-\alpha\mathcal{M}_{t-1}\|_{F}^{2}$ . Using Equation Learning-Retaining Viewpoint and gradient descent as the optimizer (i.e., memory learning algorithm), the memory update rule is:

$\displaystyle\mathcal{M}_{t}=\alpha\mathcal{M}_{t-1}+\mathbf{v}_{t}\mathbf{k}_{t}^{\top}.$ (8)

When (1) $\alpha=1$ , memory update is equivalent to Linear Attention (LA) ( Katharopoulos et al., 2020 ) ; (2) $\alpha\in\mathbb{R}$ is a learnable parameter, resulting architecture is either lightening attention ( $n>1$ ) ( Li et al., 2025 ) or RetNet ( $n=1$ ) ( Sun et al., 2023 ) ; and (3) $\alpha_{t}\in\mathbb{R}$ are data-dependent learnable parameters, resulting sequence model is Mamba2 ( Dao & Gu, 2024 ) .

RNNs with Delta Rule. To improve the memory management and to enhance the memory capacity of the above group, several studies suggest using delta rule ( Neil et al., 2017 ; Schlag et al., 2021 ) as the learning algorithm in recurrent neural networks (e.g., DeltaNet ( Schlag et al., 2021 ) , Longhorn ( Liu et al., 2024 ) , and RWKV7 ( Peng et al., 2025a ) ). In this part, we recall that where $\mathcal{M}\in\mathbb{R}^{d\times n}$ , delta rule is equivalent to optimizing MSE objective $\|\mathcal{M}_{t}\mathbf{k}_{t}-\mathbf{v}_{t}\|^{2}_{2}$ with $\text{Ret}_{t}(\mathcal{M},\mathcal{M}_{t-1})=\|\mathcal{M}_{t}-\alpha\mathcal{M}_{t-1}\|_{F}^{2}$ as local retention, and stochastic gradient descent as optimizer: ( $\eta_{t}$ is defined in Equation Learning-Retaining Viewpoint )

$\displaystyle\mathcal{M}_{t}=\alpha\left(\mathbf{I}-\eta_{t}\mathbf{k}_{t}\mathbf{k}_{t}^{\top}\right)\mathcal{M}_{t-1}+\mathbf{v}_{t}\mathbf{k}_{t}^{\top}.$ (9)

When (1) $\alpha=1$ , memory update is equivalent to DeltaNet ( Schlag et al., 2021 ) ; and (2) $\alpha_{t}\in\mathbb{R}^{m}$ are data-dependent learnable parameters, resulting sequence model is either Gated DeltaNet ( Yang et al., 2024 ) ( $m=1$ ), or RWKV7 ( Peng et al., 2025a ) ( $m=d$ ). Therefore, RNNs with delta rule are special instances of Miras .

Beyond Delta Rule. As discussed earlier, while delta rule with its value replacement strategy is more powerful than Hebbian-like learning rules, it suffers from theoretical limitations ( Irie et al., 2023 ) and achieves moderate performance in practice ( Yang et al., 2024b ) . Therefore, several studies have focused on update rules beyond delta rule. Recently, Titans ( Behrouz et al., 2024b ) suggests using non-linear MSE objective of $\|\mathcal{M}_{t}(\mathbf{k}_{t})-\mathbf{v}_{t}\|^{2}_{2}$ with both local and global retention of $\mbox{D}_{t}=\|W_{t}-W_{t-1}\|_{F}^{2}$ and $\mbox{G}_{t}=\|W_{t}\|^{2}_{2}$ and optimize it with gradient descent with momentum [note: The retention gate (forget gate) in Titans is different from Mamba2 and Gated DeltaNet that we discussed above. The main difference comes from the case of full memory erase. While Mamba2 gating removes the entire memory and treats the next token as the first ever seen data, Titans use a “ cold start ” strategy and use the previous state of the memory to measure the surprise of the incoming token before fully erasing the memory.] . Therefore, Titans-LMM is a special instance of Miras , where we use the abovementioned attentional bias and retention regularizations, and gradient descent with momentum as the optimizer.

Another example of such models is Mesa-layer, in which the model uses $\sum_{i=1}^{t}\|\mathcal{M}_{t}(\mathbf{k}_{i})-\mathbf{v}_{i}\|^{2}_{2}$ as the attentional bias objective with $\|\mathcal{M}_{t}\|_{2}^{2}$ as the retention regularization. Since these models uses Newton’s method to optimize such an objective, they provide a more expressive update rule than delta rule. We further discuss a set of new learning algorithms beyond delta rule in Section 5 .

Attention. As discussed by Sun et al. (2024) , softmax attention is a non-parameteric solution of $\ell_{2}$ -MSE loss function (i.e., $\|W\mathbf{k}-\mathbf{v}\|_{2}^{2}$ ) with Nadaraya-Watson estimator. Therefore, softmax attention is an instance of Miras , when we find the non-parameteric solution to the MSE loss with Nadaraya-Watson estimator, without retention.

### 5 Beyond Existing Attentional Biases and Retention Gates

As discussed in the previous section, existing work focus only on linear/quadratic choices for the attentional bias or retention gate. In particular, the loss function $L(\mathcal{M}(\mathbf{k}_{t}),\mathbf{v}_{t})$ is defined as $L(\mathcal{M}(\mathbf{k}_{t}),\mathbf{v}_{t})=c_{t}\|\mathcal{M}(\mathbf{k}_{t})-\mathbf{v}_{t}\|^{2}$ for some (learnable) constant $c_{t}$ in prior work. Also the regularization term $R_{t}(W)$ or the parametric $\mbox{D}_{t}$ is considered as a quadratic/linear function. In addition, almost all prior work consider $W$ to be the entire $\mathbb{R}^{d}$ space. However, in general there could be various choices for all the three aforementioned design choices. To illustrate the potential and flexibility of our designed framework, here, we review some of the potential design choices for attentional bias and retention gate in Miras . For the sake of clarity, we discuss all these attentional bias and memory retention gates based on using gradient descent as the optimizer, and so based on the provided two view points. However, these attentional bias objectives and retention regularizers can be directly used in Equation 4 and optimized by using any other optimization algorithms, resulting in different update rules.

#### 5.1 Alternative Attentional Biases

Variant 1: $\ell_{p}$ -Attentional Bias. As discussed in the main body, attentional bias defines the “similarity metric” and measures how well memory can recall the value, given its corresponding key. Although $\ell_{2}$ regression loss often is a natural choice, it is sensitive to noise in the data. A natural extension is to use $\ell_{p}$ -norm class of objectives. That is, let $\mathcal{M}$ be the memory, $\mathbf{k}$ be the keys, and $\mathbf{v}$ be the values, we define $\ell_{p}$ -attentional bias as:

$\displaystyle\mathcal{L}(\mathcal{M}(W,\mathbf{k}_{t});\mathbf{v}_{t})=\|\mathcal{M}(\mathbf{k}_{t})-\mathbf{v}_{t}\|^{p}_{p},$ (10)

where $p\in\mathbb{R}^{\geq 1}$ and $\|.\|_{p}$ is the $p$ -norm. Although depending on the distribution of the data, we might want to use different values of $p$ (see Section 6 ), different values of $p$ can result in memory architectures with interesting properties. For the sake of simplicity, let memory be a matrix, i.e., $W\in\mathbb{R}^{m\times d}$ and $\mathcal{M}(W,\mathbf{k}_{t})=W\mathbf{k}_{t}$ , the closed form can be derived as:

$\displaystyle W_{t}=W_{t}-\eta_{t}\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})=W_{t}-p\>\eta_{t}\>\>\left(\mbox{Sign}(W\mathbf{k}_{t}-\mathbf{v}_{t})\odot|W\mathbf{k}_{t}-\mathbf{v}_{t}|^{p-1}\right)\>\mathbf{k}_{t}^{\top}.$ (11)

Let $p=1$ , the recurrence is simplified as:

$\displaystyle W_{t}=W_{t}-\eta_{t}\>\>\mbox{Sign}(W_{t}\mathbf{k}_{t}-\mathbf{v}_{t})\>\mathbf{k}_{t}^{\top},$ (12)

which means that the memory has only two values of $-1$ and $1$ . We call this variation value-less associative memory, in which we store entities (keys) but map them into two extreme class of -1 and +1.

###### Remark 5 .

One of the critical challenges to use the above update rule is in the backpropagation process, in which $\mbox{Sign}(\cdot)$ and $|\cdot|$ are non-differentiable and so might cause unstable training. To overcome this issue, we use $\mbox{Sign}(x)\approx\mbox{tanh}\left(\alpha x\right),$ and $|x|=\sqrt{x^{2}+\epsilon},$ as the smooth approximators of these functions.

One simple interpretation for such behavior (i.e., value-less memory) is similar to the coping mechanism in humans ( Loftus, 1993 ) , in which the memory does not store the values for extreme events. This interpretation of protective memory in extreme events motivates our next variant.

Variant 2: Huber Loss: Memory with Coping Mechanism. While $\ell_{2}$ -norm objective is a common choice for many statistical and machine learning tasks, it is known to be sensitive to outliers and extreme samples.This sensitivity extends to the use of $\ell_{2}$ loss for attentional bias. To address this and drawing motivation from robust regression literature, we suggest utilizing the Huber loss-type ( Huber, 1992 ; Hastie et al., 2009 ) as the attentional bias, thereby reducing the negative impact of the outlier data on the memory learning process.

We can apply Huber-type loss in three different ways: The first approach is to define the summation of the Huber loss across different coordinates as the total loss, i.e.,

$$\ell(W;\mathbf{k}_{t},\mathbf{v}_{t})=\sum_{j}\mathcal{H}(\mathcal{M}(W,\mathbf{k}_{t})_{j}-\mathbf{v}_{t,j}),$$

where $\mathcal{M}(W,\mathbf{k}_{t})_{j}$ and $\mathbf{v}_{t,j}$ denote the $j$ -th coordinate of $\mathcal{M}(W,\mathbf{k}_{t})$ and $\mathbf{v}_{t}$ respectively. The function $\mathcal{H}(\cdot):\mathbb{R}\mapsto\mathbb{R}$ is the Huber loss defined as

$$\mathcal{H}(a)=\left\{\begin{array}[]{lc}\frac{1}{2}a^{2}&\textrm{if}\;|a|\leq\delta\\ \delta\left(|a|-\frac{1}{2}\delta\right)&\textrm{if}\;|a|>\delta.\end{array}\right.$$ (13)

Utilizing this attentional bias can lead to various memory update rules. For example, for the matrix form memory $\mathcal{M}(W,\mathbf{k}_{t})=W\mathbf{k}_{t},$ the update rule is given by

$\displaystyle W_{t}=W_{t-1}-\eta_{t}\left[\left((W\mathbf{k}_{t}-\mathbf{v}_{t})\mathbf{k}_{t}^{T}\right)\odot\left(\mathbf{I}(|W\mathbf{k}_{t}-\mathbf{v}_{t}|\leq\delta_{t})\mathbf{1}^{\top}\right)+\left(\delta_{t}\mbox{Sign}(W\mathbf{k}_{t}-\mathbf{v}_{t})\mathbf{k}^{\top}\right)\odot\left(\mathbf{I}(|W\mathbf{k}_{t}-\mathbf{v}_{t}|>\delta_{t})\mathbf{1}^{\top}\right)\right]$ (14)

In this formulation, the parameter $\delta_{t}$ decides the type of the memory used for each block of memory ( $\ell_{2}$ -norm objective or value-less) based on the context, making the memory more robust to outliers.

The second approach is to define the Huber-type loss based on the $\ell_{2}$ loss over all coordinates, i.e.,

$$\ell(W;\mathbf{k}_{t},\mathbf{v}_{t})=\mathcal{H}(\|\mathcal{M}(W,\mathbf{k}_{t})-\mathbf{v}_{t}\|_{2}).$$

For simplicity of derivations, assume matrix memory $M(W,\mathbf{k}_{t})=W\mathbf{k}_{t}$ . Then using gradient descent for updating memory leads the memory update rule

$\displaystyle W_{t}=W_{t-1}-\eta_{t}\>\begin{cases}\left(\mathcal{M}(W_{t-1},\mathbf{k}_{t})-\mathbf{v}_{t}\right)\mathbf{k}_{t}^{T}&\text{if}\quad\|\mathcal{M}(W_{t-1},\mathbf{k}_{t})-\mathbf{v}_{t}\|_{2}\leq\delta_{t},\\ \delta_{t}\frac{\left(\mathcal{M}(W_{t-1},\mathbf{k}_{t})-\mathbf{v}_{t}\right)}{\|\mathcal{M}(W_{t-1},\mathbf{k}_{t})-\mathbf{v}_{t}\|_{2}}\mathbf{k}_{t}^{T}&\text{Otherwise}.\end{cases}$ (15)

Again, in the form ( 15 ), the parameter $\delta_{t}$ decides the type of the memory used ( $\ell_{2}$ -norm objective or normalized version) based on the context, making the memory more robust to outliers.

Finally, in the third approach, we present a smooth mixture method, in which the memory decides if for an incoming data it is better to use $\ell_{2}$ or $\ell_{1}$ attentional bias:

$\displaystyle W_{t}=W_{t-1}-\begin{cases}\eta_{t}\>\nabla\ell_{2}(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})&\text{if}\quad\|\mathcal{M}(\mathbf{k}_{t})-\mathbf{v}_{t}\|\leq\delta_{t},\\ \eta_{t}\>\delta_{t}\nabla\ell_{1}(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})&\text{Otherwise}.\end{cases}$ (16)

The role of parameter $\delta_{t}$ is the same as above.

Variant 3: Memory Robust to Value Shifts. Following the robustness requirement discussed in the previous section, we aim to design a memory mechanism that exhibits resilience against small shifts in the value parameter. A natural approach in this context is to employ a robust optimization formulation. Specifically, we define the loss function as the worst-case $\ell_{2}$ distance between the predicted memory output and the perturbed true value:

$\displaystyle\mathcal{L}(\mathcal{M}(W,\mathbf{k}_{t});\mathbf{v}_{t})=\max_{\|\bm{\delta}\mathbf{v}_{t}\|_{2}\leq\Delta}\frac{1}{2}\|\mathcal{M}(W,\mathbf{k}_{t})-(\mathbf{v}_{t}+\bm{\delta}\mathbf{v}_{t})\|_{2}^{2}.$ (17)

This formulation seeks the memory parameters $W$ that perform well even under the adverse local perturbation of the true value $\mathbf{v}_{t}$ within an $\ell_{2}$ ball of radius $\Delta$ . To solve the maximization problem in ( 17 ), we find the optimal perturbation $\bm{\delta}\mathbf{v}_{t}^{*}$ . By solving this problem with respect to $\bm{\delta}\mathbf{v}_{t}$ , we arrive at:

$$\bm{\delta}\mathbf{v}_{t}^{*}=\Delta\frac{-\mathcal{M}(W,\mathbf{k}_{t})+\mathbf{v}_{t}}{\|\mathcal{M}(W,\mathbf{k}_{t})-\mathbf{v}_{t}\|_{2}}$$

Substituting this optimal perturbation back into the loss function ( 17 ), we obtain the robust loss:

$$\mathcal{L}(\mathcal{M}(W,\mathbf{k}_{t});\mathbf{v}_{t})=\frac{1}{2}\|\mathcal{M}(W,\mathbf{k}_{t})-\mathbf{v}_{t}\|_{2}^{2}+\Delta\|\mathcal{M}(W,\mathbf{k}_{t})-\mathbf{v}_{t}\|_{2}+\frac{1}{2}\Delta^{2}.$$

This robust loss function is a combination of the standard $\ell_{2}$ loss and a term proportional to the $\ell_{2}$ norm of the error, scaled by the robustness parameter $\Delta$ . The value of $\Delta$ thus controls the trade-off between fitting the nominal data and ensuring robustness against value perturbations.

For simplicity of the derivations, let us consider a constant value for $\Delta$ , an Euclidean retention gate $\mbox{Ret}_{t}\left(W,W_{t-1}\right)=\|W-W_{t-1}\|^{2}$ , and an attentional bias term $\widetilde{\ell}(W;\mathbf{k}_{t},\mathbf{v}_{t})=\langle W-W_{t-1},\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\rangle$ . Furthermore, to simplify the memory operation, we assume a linear matrix memory model $\mathcal{M}(W,\mathbf{k}_{t})=W\mathbf{k}_{t}$ . Under these assumptions, we can derive the memory update mechanism using gradient descent on the robust loss:

$$W_{t}=W_{t-1}-\eta\left(\big(\mathcal{M}(W_{t-1},\mathbf{k}_{t})-\mathbf{v}_{t}\big)\mathbf{k}_{t}^{\top}+\Delta\frac{\mathcal{M}(W_{t-1},\mathbf{k}_{t})-\mathbf{v}_{t}}{\|\mathcal{M}(W_{t-1},\mathbf{k}_{t})-\mathbf{v}_{t}\|_{2}}\mathbf{k}_{t}^{\top}\right)$$

In this update rule, the parameter $\Delta$ , which governs the influence of the robustness term, can also be treated as a learnable parameter, allowing the model to adapt its robustness based on the observed data.

#### 5.2 Alternative Retention Gates

Variant 1: Memorization Over A Scaled Probability Simplex Via $f$ -Divergence. A common technique in learning to prevent numerical instabilities and exploding values is to restrict the search space to a bounded domain. Following this principle, to avoid numerical instabilities, we can constrained the variable $W_{t}$ to lie within a (scaled) probability simplex. In other words, we can restrict the state to lie in the constraint set

$\displaystyle\mathcal{W}=\{W\,\mid\,\|W\|_{1}=c\;\textrm{and}\;W_{jl}\geq 0,\;\forall j,l\}.$

In this set, each matrix $W$ can be viewed as a measure. Thus, in ( Learning-Retaining Viewpoint ), we can utilize divergences over measures to define our premetric. For example, we can use $f$ -divergence measure ( Polyanskiy & Wu, 2025 , Def 4.9) , ( Csiszar, 1967 ) to define $\mbox{D}_{t}(\cdot,\cdot)$ . More specifically, let $f(\cdot)$ be a smooth strictly convex function from $\mathbb{R}^{+}$ to $\mathbb{R}$ with $f(1)=0$ . Then, we can define the $f-$ divergence between $W$ and $W^{\prime}$ as

$$\mbox{D}_{t}(W,W^{\prime})=\sum_{jl}W^{\prime}_{jl}\,f\left(\frac{W_{jl}}{W^{\prime}_{jl}}\right).$$

It is known that $f$ -divergence is zero if and only if $W=W^{\prime}$ ; see Polyanskiy & Wu, 2025 , Theorem 2.3 . Using the above premetric as the retention gate and setting $\widetilde{\ell}(W;\mathbf{k}_{t},\mathbf{v}_{t})=\langle W-W_{t-1},\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\rangle$ in ( Learning-Retaining Viewpoint ), we get the update rule

$$W_{t}=W_{t-1}\odot g\left(-\zeta_{t}-\eta_{t}\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\right).$$ (18)

Here $g(\cdot)$ is the inverse of the mapping $f^{\prime}$ , i.e., $g(f^{\prime}(\tau))=\tau,\;\forall\tau$ ; the operator $\odot$ denotes the Hadamard (elementwise) product, and $\zeta_{t}$ should be chosen such that $\|W_{t}\|_{1}=c$ . Notice that since the function $f(\cdot)$ is strictly convex and smooth, its derivative is strictly increasing and hence $g(\cdot)$ is well defined. Conversely, for any strictly monotone function $g(\cdot)$ , we can find its inverse function $g^{-1}$ (which is strictly increasing) and define $f(\tau)=\textrm{const}+\int_{\tau^{\prime}=0}^{\infty}g^{-1}(\tau^{\prime})d\tau^{\prime}$ . The term const should be chosen such that $f(1)=0$ . Then the update rule in ( 18 ) can be interpreted by the $f$ -divergence regularization, as explained above. Therefore, one can directly choose a continuous monotonically increasing function $g(\cdot)$ and use ( 18 ) for memory update.

Specializing to KL divergence. Let us further make the above update rule explicit by using special function $f$ . If we choose $f(\tau)=\tau\ln(\tau)$ , then the $f$ -divergence becomes the widely used KL divergence measure $D_{t}(W,W_{t-1})=\sum_{jl}W_{jl}\log\left(\frac{W_{jl}}{(W_{t})_{jl}}\right)$ . In addition, we can also utilize the Shannon entropy as the global retention by regularizing deviations from uniform distribution, i.e., $G_{t}(W)=\sum_{jl}W_{jl}\log(W_{jl})$ . Combining these choices of the local and global retention gates, we obtain the overall retention gate

$$\mbox{Ret}_{t}(W,W_{t-1})=\frac{1}{\eta_{t}}\sum_{jl}W_{jl}\log\left(\frac{W_{jl}}{(W_{t})_{jl}}\right)+\frac{1}{\alpha_{t}}\sum_{jl}W_{jl}\log(W_{jl})$$

Choosing the attentional bias $\widetilde{\ell}(W;\mathbf{k}_{t},\mathbf{v}_{t})=\langle W-W_{t-1},\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\rangle$ and the above retention gate will lead to the update rule

$\displaystyle W_{t}=\arg\min_{W}\;$ $\displaystyle\langle W-W_{t-1},\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\rangle+\frac{1}{\eta_{t}}\sum_{jl}W_{jl}\log\left(\frac{W_{jl}}{(W_{t})_{jl}}\right)+\frac{1}{\alpha_{t}}\sum_{jl}W_{jl}\log(W_{jl})$ (19) s.t. $\displaystyle\sum_{jl}W_{jl}=c,\;W_{jl}\geq 0,\;\forall jl$ (20)

Attaching the Lagrange multiplier to the first constraint, the KKT conditions implies

$$\left(\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\right)_{jl}+\left(\frac{1}{\eta_{t}}+\frac{1}{\alpha_{t}}\right)\left(1+\log W_{jl}\right)-\frac{1}{\eta_{t}}\log\left((W_{t-1})_{jl}\right)+\mu_{t}=0,\quad\forall j,l$$

where $\mu_{t}$ should be chosen such that $\sum_{jl}W_{jl}=c$ . Rearranging the terms and defining $\lambda_{t}=\frac{1/\alpha_{t}}{1/\alpha_{t}+1/\eta_{t}}$ , $\eta^{\prime}_{t}=\frac{1}{1/\alpha_{t}+1/\eta_{t}},$ we get the update rule

$$W_{t}\leftarrow c\;\mbox{Softmax}\left((1-\lambda_{t})\log(W_{t-1})-\eta^{\prime}_{t}\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\right)$$ (21)

where $\lambda_{t}\in(0,1)$ and $\eta^{\prime}\in\mathbb{R}^{+}$ are the parameters that can be learned during training. The Softmax operator ensures that the output lies in the set $\mathcal{W}$ .

Notice that while all above calculations are done for a matrix $W$ , similar update rule holds for other forms of parameters such as when $W$ is a neural network (or when the parameter $W$ is normalized per slice).

Variant 2: Elastic Net Regularization: Hard and Soft Forgetting. Elastic net is a powerful and popular tool in regression analysis to balance the feature selection capabilities of LASSO ( Tibshirani, 1996 ) and bias reduction properties of Ridge regression ( Hilt & Seegrist, 1977 ; Hoerl & Kennard, 1970 ) . It has been widely used in different applications due to its ability to handle high-dimensional data and mitigate the effects of multicollinearity. Given this success, a natural question is what happens if we use this regularization scheme in our context.

Let us start based on ( Learning-Retaining Viewpoint ) to design our memorization scheme. In ( Learning-Retaining Viewpoint ), we discussed that the loss function $\widetilde{\ell}_{t}(W;\mathbf{k}_{t},\mathbf{v}_{t})$ is an approximation of the original function $\ell(\cdot)$ , measuring our goodness-of-fit. Regularizing this loss with elastic net regularizer, we obtain the approximation

$$\widetilde{\ell}_{t}(W;\mathbf{k}_{t},\mathbf{v}_{t})=\langle W-W_{t-1},\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\rangle.$$

with a global retention of $\mbox{G}_{t}(W)=\frac{1}{2\beta}\|W\|_{2}^{2}+\frac{1}{\alpha}\|W\|_{1}$ . To fully specify the update rule of ( Learning-Retaining Viewpoint ), we also need to specify the premetric functions $\mbox{D}_{t}(\cdot,\cdot)$ . For the sake of keeping the update rule simple (and parallelizable), we can choose

$$\mbox{D}_{t}(W,W_{t-1})=\frac{1}{2}\|W-W_{t-1}\|_{2}^{2}.$$

These choices of the attentional bias and retention gate leads to the following update rule:

$$W_{t}=\mathcal{S}_{\gamma}\left(\lambda W_{t-1}-\zeta\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\right),$$ (22)

where $\gamma=\frac{\eta\beta}{\alpha(\eta+\beta)}$ , $\lambda=\frac{\beta}{\beta+\eta}$ , $\zeta=\eta\lambda$ , and $\mathcal{S}_{\gamma}$ is the soft thresholding operator, applied element-wise. For each element, this operator is defined as

$$\mathcal{S}_{\gamma}(z)=\mbox{sign}(z)\max\left\{0,|z|-\gamma\right\}.$$

In other words, for large values of $z$ , $\mathcal{S}_{\gamma}(z)$ makes $z$ closer to zero by $\gamma$ amount. If it is already in the $\gamma$ -vicinity of zero, then it makes it zero (hard forget).

Equation ( 22 ) can be viewed as a combination of soft forgetting (obtained by multiplying $W$ by $\lambda\in(0,1)$ , and a hard forgetting (if it is smaller than $\gamma$ ). The hyperparameters $\gamma$ , $\lambda$ , and $\zeta$ can be learned. Notice that since the shrinkage operator is not differentiable, we can approximate it with its smooth approximation. For example, we can use $\mathcal{S}_{\gamma}(z)\approx\frac{|z|*\arctan(z/\gamma)}{\pi/2}$ .

Variant 3: Elastic Net Regularization: Forgetting via Soft-thresholding. The elastic net regularizer can also be used in the ( FTRL Viewpoint ). In particular, in ( FTRL Viewpoint ), we can set

$$\frac{1}{\eta_{t}}R_{t}(W)=\frac{1}{\eta}\|W\|^{2}+\frac{1}{\alpha}\|W\|_{1}$$

and use $\widehat{\ell}(W;x_{i})=\langle W-W_{i-1},\nabla\ell(W_{i-1};x_{i})\rangle$ . Assuming initialization at $W_{0}=0$ , these choices of attentional bias and retention gate leads to the update rules:

$\displaystyle A_{t}$ $\displaystyle=A_{t-1}-\eta\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})$ $\displaystyle W_{t}$ $\displaystyle=\mathcal{S}_{\eta/\alpha}\left(A_{t}\right)$ (23)

Here $\mathcal{S}_{\eta/\alpha}(\cdot)$ is the soft-thresholding operator with parameter $\eta/\alpha$ , which can be smoothly as explained in Variant 1.1.

Variant 4: General $L_{q}$ Memory Stability. Existing work is based on the retention gate choices $\mbox{D}_{t}(W,W_{t-1})=\|W-W_{t-1}\|^{2}_{F}$ or $R(W)=\|W\|_{2}^{2}$ . However, one can choose other choices of retention gate. For example, in ( FTRL Viewpoint ), we can choose $L_{q}$ norm as the regularizer $R(W)$ . More specifically, for $1<q\leq 2$ , we can set

$$\frac{1}{\eta_{t}}R(W)=\frac{1}{2\eta(q-1)}\|W\|_{q}^{2}.$$

Using this retention gate and choosing $\widehat{\ell}_{i}(W;\mathbf{k}_{t},\mathbf{v}_{t})=\langle W-W_{i-1},\nabla\ell(W_{i-1};\mathbf{k}_{t},\mathbf{v}_{t})\rangle$ in ( FTRL Viewpoint ), leads to the update rule $W_{t}=-\eta\frac{A_{t}}{\|A_{t}\|_{p}^{p-2}},$ where $p=\frac{q}{q-1}$ and $A_{t}=\sum_{i=1}^{t}\nabla\ell(W_{i-1};\mathbf{k}_{t},\mathbf{v}_{t})$ ; see Shalev-Shwartz, 2012 , Section 2.6 . Here, $\odot$ denotes the Hadamard (element-wise) product and $|\cdot|$ is the element-wise absolute value operator. Assuming $W_{0}=0$ , this update rule can be recursively written as:

$$A_{t}=A_{t-1}-\eta\nabla\ell(W_{i-1};\mathbf{k}_{t},\mathbf{v}_{t}),\quad\textrm{and}\quad W_{t}=\frac{A_{t}}{\|A_{t}\|_{p}^{p-2}}.$$

Variant 5: Bregman Divergence as Retention Gate.. Another natural choice is to use Bregman divergence as retention gate, leading to a mirror descent-type algorithms. In particular, given a smooth strictly convex function $f(\cdot):\mathbb{R}\mapsto\mathbb{R}$ , we can define the function $F(W)=\sum_{jl}f(W_{jl})$ . Based on this choice of function $F$ , we define the Bregman divergence

$$D_{t}(W,W^{\prime})=F(W)-F(W^{\prime})-\langle W^{\prime},W-W^{\prime}\rangle$$

as our parametric function. Utilizing this retention gate and choosing $\widetilde{\ell}_{t}(W;\mathbf{k}_{t},\mathbf{v}_{t})=\langle W-W_{t-1},\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\rangle$ in ( Learning-Retaining Viewpoint ), we obtain the update rule

$$W_{t}=g\left(-\eta\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})+F^{\prime}(W_{t-1})\right).$$

Here, $F^{\prime}$ is the mapping obtained by applying $f^{\prime}(\cdot)$ (the derivative of $f$ ) element-wise to all entries of its input matrix argument. The function $g$ is the inverse of the mapping $F^{\prime}(\cdot)$ , i.e., $g(F^{\prime}(W))=W$ .

If we choose $f(\tau)=\frac{\tau^{2}}{2}$ , then $F^{\prime}(W)$ becomes the identity mapping and so is $g$ . Therefore, the above update becomes simple gradient descent with no nonlinearity involved in the update rule. However, other choices of $f(\cdot)$ introduces additional nonlinearity in $g(\cdot)$ , which can enhance the expressivity of our memory. For example, we can choose the function $f(\cdot)$ so that its derivative becomes the inverse sigmoid function, i.e., $f^{\prime}(\tau)=\ln\left(\frac{\tau}{1-\tau}\right)$ with $f^{\prime}:(0,1)\mapsto\mathbb{R}$ . Since $f^{\prime}(\cdot)$ is strictly increasing, then the function $f(\cdot)$ (and hence $F(\cdot)$ ) is strictly convex. Therefore, the Bregman divergence is well defined. Moreover, the inverse of the function $f^{\prime}(\cdot)$ becomes the sigmoid function, i.e., $g(\tau)=\sigma(\tau)=\frac{\exp(\tau)}{1+\exp(\tau)}$ with $g:\mathbb{R}\mapsto(0,1)$ . Then, the update of the memory becomes

$$W_{t}=\sigma\left(\ln\left(\frac{W_{t}}{1-W_{t}}\right)-\eta\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\right),$$

where $\sigma$ is the sigmoid function operated element-wise on the entries of $W$ , and the division operator $\frac{W_{t}}{1-W_{t}}$ is also performed element-wise. This update rule guarantees that the elements of $W_{t}$ remains within the interval $(0,1)$ .

#### 5.3 Miras ’s Variants: Moneta , Yaad , and Memora

In the previous section we discussed different potential choices for attentional bias and retention gate to show the generality and the potential of Miras . In this section, building upon our framework, we present three novel sequence models, each of which designed based on a different motivation, and discuss how they can leverage fast parallel training.

Moneta . Given $p,q\in\mathbb{R}^{\geq 1}$ , we design $(p,q)$ - Moneta as the variant of Miras as follows: (1) For the choice of memory architecture, we use an MLP with 2 layers with expansion factor of 4 and GELU activation function ( Hendrycks & Gimpel, 2016 ) . We also use residual connections and layer norm, resulting in $\mathcal{M}(x)=x+\texttt{LN}(W_{1}\sigma(W_{2}x))$ . (2) We choose $\ell_{p}$ -attentional bias (introduced in Equation 11 ) for Moneta . (3) For the choice of retention gate, we use the hybrid of $\ell_{q}$ retention gate $\frac{1}{2(q-1)}\|W\|_{q}^{2}$ (see Section 5.2 for details) and the standard $\ell_{2}$ regularization $\frac{1}{\beta}\|W\|_{2}^{2}$ . (4) Finally, we use gradient descent as the memory learning algorithm. The above choices, result in the following recurrent formula for the memory module:

$\displaystyle A_{t}=\alpha_{t}A_{t-1}-\eta_{t}\nabla\ell_{p}(W_{i-1};\mathbf{k}_{t},\mathbf{v}_{t}),\quad\textrm{and}\quad W_{t}=\frac{A_{t}}{\|A_{t}\|_{q}^{q-2}}.$ (24)

Notably the gradient can be calculated using:

$\displaystyle\nabla\ell(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})=p\>\eta_{t}\>\>\left(\mbox{Sign}(W\mathbf{k}_{t}-\mathbf{v}_{t})\odot|W\mathbf{k}_{t}-\mathbf{v}_{t}|^{p-1}\right)\>\mathbf{k}_{t}^{\top}.$ (25)

We use $(p,q)=(3,4)$ .

Yaad . Building upon our discussion on the importance of robust memory that protects itself from extreme events (tokens), we design Yaad based on Huber objective. That is, in Miras , for the choice of memory structure, we follow Moneta and use an MLP with the same architecture as above; for the choice of attentional bias, we use Huber loss (defined in Equation 16 ); for the choice retention gate, for the sake of simplicity, we use a combination of local and global retention as $\mbox{Ret}_{t}(W,W_{t-1})=\frac{1}{2\theta_{t}}\|W-W_{t-1}\|_{\text{F}}^{2}+\frac{1}{\beta_{t}}\|W\|_{2}^{2}$ , which is equivalent to the “forget gate” mechanism introduced by Behrouz et al. (2024b) ; and finally, we simply use gradient descent as the memory learning algorithm. Given the above choices, we can write the resulted memory learning process as follows:

$\displaystyle W_{t}=\alpha_{t}W_{t-1}-\begin{cases}\eta_{t}\>\nabla\ell_{2}(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})&\text{if}\quad\|\mathcal{M}(\mathbf{k}_{t})-\mathbf{v}_{t}\|\leq\delta_{t},\\ \eta_{t}\>\delta_{t}\nabla\ell_{1}(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})&\text{Otherwise}.\end{cases}$ (26)

Note that for improving the expressive power, in all architectures, we decouple the learning rate $\eta$ and the retention gate rate $\alpha$ , resulting in an independent parameter $\beta_{t}\in[0,1]^{d}$ .

Memora . Finally, in Memora , we use the idea of elastic net regularization (i.e., hard and soft retention). To this end, in Miras : (1) For the choice of memory architecture, similar to above variants, we use an MLP (the same architecture as the previous variants). (2) For the choice of attentional bias, we use simple $\ell_{2}$ regression loss. (3) For the choice of retention gate we use KL divergence as in Equation 21 . (4) Finally, we optimize the memory using gradient descent, resulting in the following update rule:

$\displaystyle W_{t}=\mbox{Softmax}\left(\alpha_{t}\log(W_{t-1})-\eta_{t}\nabla\ell_{2}(W_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})\right)$ (27)

[Figure] Figure 2 : Visualization of the Miras ’s variant architecture, their hybrid counterpart with SWA, and block design of Miras layer.

#### 5.4 Architecture Backbone and Fast Training

Architectural Backbone. For the architectural backbone, we fully follow recent studies ( Behrouz et al., 2024b ; Yang et al., 2024 ) : We replace attention modules with our variants of Miras in Llama’s macro architecture with MLPs with SwiGLU (.) activation, rotary positional encodings (RoPE) ( Su et al., 2024 ) , and RMSNorm ( Zhang & Sennrich, 2019 ) . For Miras layer block, we follow the recent modern linear recurrent models ( Yang et al., 2024 ; Behrouz et al., 2024b ) , and incorporate a 1D depthwise-separable convolution layer (with kernel size of 4) after each of the query, key, and value projections. For the sake of training stability, we also use $\ell_{2}$ normalization to $\mathbf{q}$ and $\mathbf{k}$ . The output of Miras layer block is normalized and gated with a linear layer ( Mehta et al., 2023 ) .

Channel-wise Parameters. For learnable parameters of $\eta_{t},\delta_{t}$ and the retention gate of $\alpha_{t}$ we use channel-wise parametrization, i.e., $\eta_{t},\delta_{t},\alpha_{t}\in\mathbb{R}^{d}$ . While gaining more expressive power, this parametrization results in significant parameter increase. To mitigate this issue, following Peng et al. (2025a) , we use low-rank projections to project the input into $\mathbb{R}^{k}$ and then to $\mathbb{R}^{d}$ , where $k$ is a hyperparameter (usually 32 or 64). The backbone architecture is illustrated in Figure 2 .

Hybrid Models. We also evaluate the hybrid version of Miras ’s variants. For hybrid models, we follow the Samba ( Ren et al., 2024 ) architecture, in which we sequentially combine our Miras layer with Sliding Window Attention (SWA). The illustration of hybrid model Figure 2 .

Parallelizable Training. While the design of Miras ’s variant are theoretically well-motivated, their recurrence is non-linear, potentially make their straightforward training slow for large scales. In this section, we build upon the work of Behrouz et al. (2024b) ; Sun et al. (2024) to make the training parallelizable. The main idea is to divide the sequence into chunks with size $b$ (usually is 16 or 64) and calculate the gradient for all tokens in the current chunk with respect to the last state of the memory in the previous chunk. That is, we use $\nabla\ell(\mathcal{M}_{t^{\prime}};\mathbf{k}_{t},\mathbf{v}_{t})$ instead of $\nabla\ell(\mathcal{M}_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})$ , where $t^{\prime}$ is the last state in the previous chunk.

Given the above trick, we can calculate all gradients at once and make the recurrence inside each chunk linear. However, to fully take advantage of accelerators, we need to reformulate the process as matrix multiplication. For Moneta , for the sake of clarity, assume $q=2$ . We follow the same algorithm as Behrouz et al. (2024b) and expand the recurrence as follows:

$\displaystyle\mathcal{M}_{t}$ $\displaystyle=\alpha_{t}\mathcal{M}_{t-1}-\eta_{t}\nabla\ell(\mathcal{M}_{t-1};\mathbf{k}_{t},\mathbf{v}_{t})$ $\displaystyle=\beta_{t}\mathcal{M}_{0}-\sum_{i=1}^{t}\eta_{i}\frac{\beta_{t}}{\beta_{i}}\nabla\ell(\mathcal{M}_{t^{\prime}};\mathbf{k}_{i},\mathbf{v}_{i}),$ (28)

where $t^{\prime}=t-\texttt{mod}(t,b)$ , and $\beta_{i}=\prod_{j=1}^{i}\alpha_{j}$ . For the sake of clarity, we focus on the first chunk, i.e., $t=b$ and so $t^{\prime}=0$ , and explain the process for the case that $\mathcal{M}_{t}=W_{t}$ is linear. The process for 2-layer MLPs and other chunks is similar. Using $\ell_{p}$ loss function, we have:

$\displaystyle\nabla\ell(W_{0};\mathbf{k}_{t},\mathbf{v}_{t})=p\left(\mbox{Sign}(W_{0}\mathbf{k}_{t}-\mathbf{v}_{t})\odot|W_{0}\mathbf{k}_{t}-\mathbf{v}_{t}|^{p-1}\right)\>\mathbf{k}_{t}^{\top}$ $\displaystyle\Rightarrow$ $\displaystyle\sum_{i=1}^{b}\eta_{i}\frac{\beta_{b}}{\beta_{i}}\nabla\ell(W_{0};;\mathbf{k}_{i},\mathbf{v}_{i})=p\mathbf{E}_{b}\odot\mathbf{B}_{b}\odot\mbox{Sign}(W\mathbf{k}_{t}-\mathbf{v}_{t})\odot(|W_{0}\>\mathbf{K}-\mathbf{V}|^{p-1})\>\mathbf{K}^{\top},$ (29)

where $\mathbf{E}_{b}=\begin{bmatrix}\eta_{1}&\eta_{2}&\dots&\eta_{b}\end{bmatrix}$ and $\mathbf{B}_{b}$ is defined analogously on $\frac{\beta_{b}}{\beta_{i}}$ s. For the sake of stablity in training, we use $\texttt{Sign}(x)\approx\texttt{tanh}\left(\alpha x\right)$ and $|x|=\sqrt{x^{2}+\epsilon}$ , where $\epsilon>0$ is a small number (i.e., $\epsilon=1e-6$ ). As discussed in Equation 24 , the case that $q\neq 2$ appears as a normalization term on the memory. Similar to Titans ( Behrouz et al., 2024b ) and TTT ( Sun et al., 2024 ) , we do not apply this non-linearity inside each chunk and instead use it at the end of each chunk.

For Yaad , the process is very similar to the above. We calculate the gradient of both $\ell_{1}$ and $\ell_{2}$ loss and use a masking based on $\|\mathcal{M}(\mathbf{k}_{t})-\mathbf{v}_{t}\|\leq\delta_{t}$ .

For Memora , the update rule has two non-linear part, i.e., softmax and $\log$ , making the model hardly parallelizable. To this end, as discussed above, we use its linear version inside each chunk and its non-linear version across chunks. However, using both log and softmax at the end of each chunk removes the effect of log. To this end, we consider a lag tokens after each chunk (i.e., tokens with index $i=kb+1$ , where $b$ is the chunk size and $k\in\mathbb{Z}^{+}$ ). That is, let $\mathcal{M}_{0}$ be the last state of the memory in previous chunk, we have:

$\displaystyle\mathcal{M}_{1}=\mbox{Softmax}\left(\alpha_{1}\log(\mathcal{M}_{0})-\eta_{1}\nabla\ell_{2}(\mathcal{M}_{0};\mathbf{k}_{1},\mathbf{v}_{1})\right),$ (30)

and then we use $\mathcal{M}_{1}$ for the next chunk. Again, for the sake of clarity, assume that memory is linear, i.e., $\mathcal{M}_{1}=W_{1}$ :

$\displaystyle\nabla\ell(W_{1};\mathbf{k}_{t},\mathbf{v}_{t})=(W_{1}\mathbf{k}_{t}-\mathbf{v}_{t})\>\mathbf{k}_{t}^{\top}$ (31) $\displaystyle\Rightarrow$ $\displaystyle\sum_{i=1}^{b}\eta_{i}\frac{\beta_{b}}{\beta_{i}}\nabla\ell(W_{1};;\mathbf{k}_{i},\mathbf{v}_{i})=\mathbf{E}_{b}\odot\mathbf{B}_{b}\odot(W_{1}\>\mathbf{K}-\mathbf{V})\>\mathbf{K}^{\top},$ (32)

where matrices are defined the same as for Equation 29 .

### 6 Experiments

In our experimental evaluations, we aim to answer three main questions: (1) Does different attentional biases results in different architectures in practice? (2) How does different types of retention gates (i.e., retention gate) affect the performance of the model in long context? (3) How do Memora , Moneta , and Yaad perform in downstream tasks compare to baselines?

Setup. We train our models with training context window of size 4096 using either FineWeb-Edu dataset ( Penedo et al., 2024 ) (for LM and common-sense reasoning tasks) or C4 dataset ( Raffel et al., 2020 ) (for scaling patterns). We use model sizes of 120M, 340M, 760M, and 1.3B parameters. We train small models (120M and 340M) on 15B tokens sampled from the dataset, the medium size model (760M) on 30B tokens, and the large model on 100B tokens. Baseline results are reported by Behrouz et al. (2024b) .

#### 6.1 Language Modeling and Common-sense Reasoning

We follow recent studies ( Yang et al., 2024 ; Yang et al., 2024b ; Behrouz et al., 2024b ) and first focus on the perplexity in language modeling and also commonsense reasoning tasks. The results for Memora , Yaad , Moneta and also baselines with size of 340M, 760, and 1.3B are reported in Table 2 . All of our variants outperforms all the baselines including Transformer++, modern linear recurrent models and hybrid methods. The superior performance compared to hybrid models is particularly important as all of our variants are pure recurrent (attention-free). Among the three variants of Miras , while Moneta achieves slightly weaker performance than Memora , and Yaad , the other two variants are close and depending on the task and model size, the best model can vary.

Table: Table 2 : Performance of Miras ’s variants and baselines on language modeling and common-sense reasoning tasks. Hybrid models are marked with ∗ . The best results of simple and hybrid models are highlighted. In largest scale, we compare our simple models with even hybrid models and highlight the best results.

| Model | Wiki. | LMB. | LMB. | PIQA | Hella. | Wino. | ARC-e | ARC-c | SIQA | BoolQ |
|  | ppl $\downarrow$ | ppl $\downarrow$ | acc $\uparrow$ | acc $\uparrow$ | acc_n $\uparrow$ | acc $\uparrow$ | acc $\uparrow$ | acc_n $\uparrow$ | acc $\uparrow$ | acc $\uparrow$ |
| 340M params / 15B tokens |
| Transformer++ | 31.52 | 41.08 | 30.76 | 62.98 | 34.76 | 50.53 | 45.21 | 24.05 | 36.81 | 58.24 |
| RetNet | 32.50 | 49.73 | 28.24 | 62.61 | 34.15 | 50.91 | 44.27 | 23.62 | 36.79 | 59.72 |
| GLA | 28.51 | 43.02 | 28.73 | 64.05 | 35.96 | 50.00 | 54.19 | 24.29 | 37.13 | 58.39 |
| Mamba | 30.83 | 40.21 | 29.94 | 63.79 | 35.88 | 49.82 | 49.24 | 24.56 | 35.41 | 60.07 |
| DeltaNet | 28.65 | 47.30 | 28.43 | 63.52 | 35.95 | 49.63 | 52.68 | 25.37 | 37.96 | 58.79 |
| TTT | 27.44 | 34.19 | 30.06 | 63.97 | 35.71 | 50.08 | 53.01 | 26.11 | 37.32 | 59.83 |
| Gated DeltaNet | 27.01 | 30.94 | 34.11 | 63.08 | 38.12 | 51.60 | 55.28 | 26.77 | 34.89 | 59.54 |
| Moneta (ours) | 26.19 | 29.31 | 35.70 | 63.99 | 39.23 | 52.04 | 55.96 | 27.15 | 37.29 | 60.22 |
| Yaad (ours) | 26.61 | 29.11 | 34.09 | 64.93 | 39.86 | 51.12 | 54.75 | 28.64 | 33.82 | 60.29 |
| Memora (ours) | 27.16 | 30.44 | 33.68 | 65.21 | 39.17 | 51.23 | 53.40 | 27.99 | 34.1 | 59.29 |
| 760M params / 30B tokens |
| Transformer++ | 25.21 | 27.64 | 35.78 | 66.92 | 42.19 | 51.95 | 60.38 | 32.46 | 39.51 | 60.37 |
| RetNet | 26.08 | 24.45 | 34.51 | 67.19 | 41.63 | 52.09 | 63.17 | 32.78 | 38.36 | 57.92 |
| Mamba2 | 22.94 | 28.37 | 33.54 | 67.90 | 42.71 | 49.77 | 63.48 | 31.09 | 40.06 | 58.15 |
| DeltaNet | 24.37 | 24.60 | 37.06 | 66.93 | 41.98 | 50.65 | 64.87 | 31.39 | 39.88 | 59.02 |
| TTT | 24.17 | 23.51 | 34.74 | 67.25 | 43.92 | 50.99 | 64.53 | 33.81 | 40.16 | 59.58 |
| Gated DeltaNet | 21.18 | 22.09 | 35.54 | 68.01 | 44.95 | 50.73 | 66.87 | 33.09 | 39.21 | 59.14 |
| Samba ∗ | 20.63 | 22.71 | 39.72 | 69.19 | 47.35 | 52.01 | 66.92 | 33.20 | 38.98 | 61.24 |
| Gated DeltaNet-H2 ∗ | 19.88 | 20.83 | 39.18 | 68.95 | 48.22 | 52.57 | 67.01 | 35.49 | 39.39 | 61.11 |
| Moneta (ours) | 21.18 | 21.94 | 38.02 | 69.55 | 49.16 | 53.01 | 67.47 | 36.09 | 40.53 | 63.18 |
| Yaad (ours) | 20.99 | 21.57 | 37.85 | 69.14 | 50.02 | 53.93 | 67.78 | 36.27 | 41.01 | 63.34 |
| Memora (ours) | 22.28 | 22.31 | 38.19 | 67.82 | 49.30 | 53.28 | 63.57 | 36.15 | 40.94 | 62.96 |
| Moneta -H (ours) | 18.72 | 20.13 | 40.59 | 70.84 | 50.13 | 54.17 | 67.64 | 36.79 | 40.87 | 62.43 |
| Yaad -H (ours) | 18.59 | 19.80 | 40.22 | 69.51 | 50.48 | 53.69 | 68.04 | 36.55 | 40.28 | 61.94 |
| Memora -H (ours) | 18.24 | 20.55 | 39.91 | 69.06 | 49.84 | 52.88 | 66.90 | 36.12 | 40.99 | 61.75 |
| 1.3B params / 100B tokens |
| Transformer++ | 18.53 | 18.32 | 42.60 | 70.02 | 50.23 | 53.51 | 68.83 | 35.10 | 40.66 | 57.09 |
| RetNet | 19.08 | 17.27 | 40.52 | 70.07 | 49.16 | 54.14 | 67.34 | 33.78 | 40.78 | 60.39 |
| Mamba2 | 16.56 | 12.56 | 45.66 | 71.87 | 55.67 | 55.24 | 72.47 | 37.88 | 40.20 | 60.13 |
| DeltaNet | 17.71 | 16.88 | 42.46 | 70.72 | 50.93 | 53.35 | 68.47 | 35.66 | 40.22 | 55.29 |
| Gated DeltaNet | 16.42 | 12.17 | 46.65 | 72.25 | 55.76 | 57.45 | 71.21 | 38.39 | 40.63 | 60.24 |
| Samba ∗ | 16.13 | 13.29 | 44.94 | 70.94 | 53.42 | 55.56 | 68.81 | 36.17 | 39.96 | 62.11 |
| Gated DeltaNet-H2 ∗ | 15.91 | 12.55 | 48.76 | 72.19 | 56.88 | 57.77 | 71.33 | 39.07 | 41.91 | 61.55 |
| Moneta (ours) | 15.52 | 11.47 | 47.88 | 73.16 | 56.14 | 59.09 | 72.53 | 40.32 | 41.91 | 61.18 |
| Yaad (ours) | 15.18 | 11.89 | 47.23 | 72.81 | 56.46 | 59.02 | 72.14 | 40.05 | 40.73 | 61.86 |
| Memora (ours) | 15.90 | 12.04 | 48.67 | 73.10 | 55.99 | 57.36 | 71.55 | 37.92 | 40.19 | 61.34 |

#### 6.2 Scaling Pattern

To evaluate the scaling pattern of models and for comparing them with baseline, in this section, we plot their performance with varying the model size and the context window.

[Figure] Figure 3 : Scaling patterns when increasing ( Left ) model size, ( Middle ) sequence length (model size = 340M) (3) ( Right ) sequence length (model size = 760M) on C4 dataset.

Context Length. We first vary the training context length from 2K to 32K for two version of our model with size 340M and 760M. The results are reported in Figure 3 (Middle and Right). All three variants of Miras scales better than state-of-the-art baselines when increasing the context length. We attribute this superior performance to: (1) expressive memory architecture. Contrary to baselines like Mamba2 and GSA that uses vector- and matrix-valued memory, our variants are using 2-layer MLPs with more expressive power to learn from longer sequences. (2) The choice of retention gate and attentional bias: All of our three variants go beyond the standard attentional biases and retention gates. These choices can help the memory to better manage its fixed-size capacity.

Model Size. We also report the #FLOPs vs. perplexity of our models and baselines in Figure 3 (Left). All three variants outperforms all baselines given almost the same budget of FLOPs. These results, once again support the importance of powerful memory design.

#### 6.3 Needle In Haystack

To evaluate the effective context window of our models and baselines, we use needle-in-haystack task. In this task, we evaluate the model on retrieving a piece of information (i.e., the “needle”) from long distractor texts (i.e., the “haystack”). We focus on the Single NIAH (S-NIAH) task from RULER benchmark ( Hsieh et al., 2024 ) and evaluate our models and baselines on sequences with length 1K, 2K, 4K, and 8K. The results are reported in Table 3 . All our variants outperforms all the baselines with a considerable margin. Interestingly, Moneta shows better performance than others when the data is synthetic noise (S-NIAH-PK). This observation validates the effectiveness of $p$ -norm objective and retention gates as they are more robust to noise.

Table: Table 3 : Performance of Moneta , Yaad , Memora , and baselines on NIAH task from RULER benchmark. The best results with highest accuracy are highlighted.

| Model | S-NIAH-PK | S-NIAH-N | S-NIAH-W | Average |
| 2K | 4K | 8K | 2K | 4K | 8K | 1K | 2K | 4K |
| Mamba2 | 98.6 | 61.4 | 31.0 | 98.4 | 55.8 | 14.2 | 62.2 | 42.2 | 4.2 | 52.0 |
| DeltaNet | 96.8 | 98.8 | 98.6 | 47.2 | 15.4 | 12.8 | 85.2 | 46.2 | 20.0 | 57.9 |
| Gated DeltaNet | 89.8 | 91.4 | 90.0 | 99.2 | 91.8 | 26.4 | 86.4 | 82.6 | 24.4 | 75.8 |
| TTT | 98.4 | 98.8 | 98.0 | 60.2 | 36.6 | 10.2 | 85.8 | 78.8 | 28.0 | 66.1 |
| Moneta | 99.4 | 98.8 | 98.8 | 99.4 | 99.4 | 92.8 | 92.2 | 88.2 | 70.8 | 93.5 |
| Yaad | 99.2 | 98.6 | 94.4 | 99.8 | 98.6 | 93.2 | 91.8 | 89.6 | 67.4 | 92.9 |
| Memora | 99.2 | 98.8 | 92.6 | 98.4 | 99.2 | 93.2 | 92.4 | 88.2 | 70.4 | 92.1 |

#### 6.4 Ablation Study

In this section we perform ablation studies to validate if different design choices that we discussed through the paper are positively contributing for achieving better results.

The Effect of $p$ on Performance. We first evaluate the effect of $p$ on the performance of Moneta . We vary the value of $p\in\{1,1.5,2,2.8,3,3.2,4\}$ and context window from 2K to 16K. The results are reported in Section 6.4 . Interestingly, there is no monotone pattern when increasing the value of $p$ and the best performance is achieved when $p=3$ , while $p=4$ achieves the worst performance. Also, although different values of $p$ results in different memory modules with varied performance, the scaling pattern when increasing the context length is almost the same.

The Effect of $q$ on Performance. Similarly, we evaluate the effect of $q$ by varying it in $\{2,3,4,5\}$ . Interestingly, contrary to $p$ , the value of $q$ can change the scaling pattern when increasing the context length. The main reason for this observation is that the value of $q$ determines the retention gate and a powerful retention gate can improve the memory management, resulting in better performance.

The Effect of Design. To evaluate the architectural design choices, we perform an ablation study on Yaad . The results are in Section 6.4 . The first row, reports the performance of Yaad , while (1) the second row removes the retention (i.e., $\beta=1$ ), (2) third row make $\delta$ input independent, (3) the third row removes $\ell_{2}$ -loss from the Huber loss, (4) the forth row removes the $\ell_{1}$ condition, and (5) the last row replaces the MLP with a linear layer. These results indicate that all design choices are contributing to the performance of the model.

[Figure] Figure 4 : The effect of parameters $p$ and $q$ on the performance with different context length.

[Figure] Table 4 : Ablation study on the components of Yaad .

### 7 Conclusion

In this paper, we present Miras , a general framework that explains the connection of online optimization and test time memorization. Miras framework can explain the role of several standard architectural choices in the literature (e.g., forget gate) and helps design next generation of architectures that are capable of managing the memory better. Building upon our framework, we present three novel sequence models, each of which with its own (dis)advantages. Our experimental evaluations show that all these variants are more powerful than Transformers and linear RNNs, in various downstream tasks. In this work, we present a diverse set of variants using Miras . In future, exploring these alternative architectures for different downstream tasks is an interesting future direction.

### Appendix A Additional Related Work

Modern Linear RNNs. Recent efforts aim to overcome Transformers quadratic cost and limitations in long-context modeling by designing efficient recurrent alternatives ( Tiezzi et al., 2024 ) , mainly due to fast inference and training of such models. The first generation of models–such as RetNet ( Sun et al., 2023 ) , LRU ( Orvieto et al., 2023 ) , RWKV ( Peng et al., 2023 ) , S5 ( Smith et al., 2023 ) , and S4 ( Gu et al., 2022 ) –uses data-independent transition matrix mechanism with Hebbian-like update rule. The second generation of such models started to incorporate input-dependent parameters into such linear architectures (e.g., Griffin ( De et al., 2024 ) , SSMs ( Hasani et al., 2023 ; Behrouz et al., 2024a ; Dao & Gu, 2024 ) , RWKV6 ( Peng et al., 2024 ) ), and/or use more expressive memory updating rule based on delta rule ( Peng et al., 2025a ; Schlag et al., 2021 ; Yang et al., 2024b ; Yang et al., 2024 ; Liu et al., 2024 ) . The next generation of models, extend the memory architecture to deep models, while using delta-rule-like update rule ( Sun et al., 2024 ) , or momentum-based update rule ( Behrouz et al., 2024b ) . Recently, to further enhance the performance of delta-rule-based sequence models, Siems et al. (2025) suggest using multiple gradient descent update per token, resulting in more expressive sequence models in state tracking tasks.

In addition to the above fast linear recurrent sequence models, several studies have focused on (interpretable) non-linear RNNs ( Csordás et al., 2024 ; Merrill et al., 2024 ; Lim et al., 2024 ; Schöne et al., 2025 ; Karami & Mirrokni, 2025 ; Von et al., 2023 ; Gonzalez et al., 2024 ) , and how their training can be faster ( Gonzalez et al., 2024 ; Lim et al., 2024 ; Schöne et al., 2025 ) . However, due to the recurrent nature of such models, parallelizing them in larger scales is still challenging.

Fast Weight Programs. The idea of interpretation of linear layers as the key-value associative memory system backs to Hopfield networks ( Hopfield, 1982 ) and then fast weight programs, in which dynamic fast programs are incorporated into recurrent neural networks as writable memory ( Schlag et al., 2021 ; Schmidhuber, 1992 ; Schmidhuber, 1993 ) . The two learning rules of Hebbian ( Hebb, 2005 ) and delta rule ( Prados & Kak, 1989 ) are the most popular learning rules for them, which have been extensively explored in the literature ( Munkhdalai & Yu, 2017 ; Schmidhuber, 1992 ; Munkhdalai et al., 2019 ; Schlag et al., 2021 ; Irie et al., 2021 ; Yang et al., 2024b ; Yang et al., 2024 ) .

Test Time Training. The key ideas of learning at test time backs to early studies on local learning Bottou & Vapnik, 1992 , in which each test data is trained on its neighbors before making a prediction ( Zhang et al., 2006 ; Gandelsman et al., 2022 ) . Later applying this idea on modern architectures, it has shown promising performance in diverse downstream tasks such as vision tasks ( Jain & Learned-Miller, 2011 ; Mullapudi et al., 2019 ) , video generation ( Dalal et al., 2025 ) , etc., mostly due to their ability to mitigate out-of-distribution samples.

Hopfield Networks. We build Miras based on the concept of associative memory in its broad form, where we learn an underlying mapping between keys and values. One of the earliest studies that discuss building neural architectures based on associative memory is Hopfield Networks ( Hopfield, 1982 ) , in which associative memory is defined as the minimizing the energy function required to store keys and values. While traditional Hopfield networks has limited applicability in recent years (mainly due to limited capacity of vector-valued memory and energy function), several recent studies aim to improve their capacity by various techniques ( Krotov, 2021 ; Li et al., 2024a ; Krotov & Hopfield, 2016 ) , including extending the energy function of such models based on exponential kernels ( Krotov & Hopfield, 2016 ; Lucibello & Mézard, 2024 ) , and discuss their connection to Transformers ( Ramsauer et al., 2021 ; Hu et al., 2024 ) .

Unifying Frameworks. In recent years, there have been growing efforts to understand the underlying mechanism of sequence models and unify (a subset of) them through a single perspective. Dao & Gu (2024) present SSD framework to connect linear Transformers and (a subset of) linear recurrent models through the lens of associative operators and structured matrices. The SSD framework, however, is limited to models with vector or matrix-valued memory that are updated using a Hebbian-like update rules. Later, Liu et al. (2024) present an online learning perspective on (a subset of) linear recurrent models. While this framework can also explain more expressive recurrent models based on delta rule, it is limited to online learners (i.e., models that optimize their internal associative memory using stochastic optimizers, such as stochastic gradient descent) with matrix-valued memory. Several modern sequence models, such as Transformers ( Vaswani et al., 2017a ) or Titans ( Behrouz et al., 2024b ) cannot be expressed in this framework. Sun et al. (2024) further provide a unifying perspective on how linear and softmax attention are respectively parametric and non-parameteric solutions of (kernel) regression loss but consider other modern linear RNNs outside of this class of models, mainly due to limiting the objective to be regression loss. Recently, in a concurrent work to ours, Wang et al. (2025) also force models to have the same attentional bias objective and show that with additional simplification of modern RNNs (e.g., RetNet ( Sun et al., 2023 ) , Mamba ( Dao & Gu, 2024 ) ) they approximately place in the same class of models that internally optimize regression loss. However, this simplification, fully change the understanding of underlying update rules in these models. For example, contrary to Wang et al. (2025) , Miras can distinguish models with Hebbian-like update (with dot product similarity) and delta rule update (with regression loss). Furthermore, all presented sequence models in this work (e.g., Moneta , Memora , Yaad ) as well as models like HGRN2 ( Qin et al., 2024 ) are placed outside of this class of models, due to their different attentional bias.

### Appendix B Proof of Proposition 3.2

Here we present the proof of Proposition 3.2 . For the sake of completeness, let us first re-state this Proposition.

Proposition 3.2 . Let $\eta_{t}=\eta$ and define $h_{t}(W):=\sum_{i=1}^{t-1}\widehat{\ell}_{i}(W;\mathbf{k}_{i},\mathbf{v}_{i})+\frac{1}{\eta}R(W)$ . Assume $\mathcal{W}=\mathbb{R}^{d}$ and the function $h_{t}(W)$ is strictly convex in $W$ and let $\mathcal{D}_{h}(\cdot,\cdot)$ be the Bregman divergence defined by function $h(\cdot)$ , i.e., $\mathcal{D}_{h}(W,W^{\prime})=h(W)-h(W^{\prime})-\langle\nabla h(W^{\prime}),W-W^{\prime}\rangle$ . Set $\mbox{Ret}_{t}(W,W^{\prime})=\mathcal{D}_{h}(W,W^{\prime})$ and $\widetilde{\ell}_{t}(W;x_{t})=\widehat{\ell}_{t}(W;x_{t})$ in ( Learning-Retaining Viewpoint ). Then, the update rule in ( Learning-Retaining Viewpoint ) is equivalent to the update rule in ( FTRL Viewpoint ).

###### Proof.

Let $\{\widehat{W}_{1},\widehat{W}_{2},\ldots\}$ be the sequence of parameters obtained by ( FTRL Viewpoint ) and $\{\widetilde{W}_{1},\widetilde{W}_{2},\ldots\}$ be the sequence of parameters obtained by ( Learning-Retaining Viewpoint ). To show both update rules are equivalent, it suffices to show that the above two sequences are the same if they are initialized at the same point. We prove this statement by induction. First of all, since both sequences are initialized at the same point, the induction base is satisfied (i.e. $\widetilde{W}_{1}=\widehat{W}_{1}$ . Now, assume by induction hypothesis that

$$\widetilde{W}_{t-1}=\widehat{W}_{t-1}.$$ (33)

To complete the induction, we need to show $\widetilde{W}_{t}=\widehat{W}_{t}$ . To this end, notice that, by ( Learning-Retaining Viewpoint ), we have

$$\widetilde{W}_{t}=\arg\min_{W}\quad\widetilde{\ell}_{t}(W,\mathbf{k}_{t},\mathbf{v}_{t})+\mbox{Ret}_{t}(W,\widetilde{W}_{t-1})$$

Using the choice of the Attentional Bias and the Retention function in the Proposition, we obtain

$$\begin{split}\widetilde{W}_{t}=\arg\min_{W}\quad&\widehat{\ell}_{t}(W,\mathbf{k}_{t},\mathbf{v}_{t})+\sum_{i=1}^{t-1}\widehat{\ell}_{i}(W,\mathbf{k}_{i},\mathbf{v}_{i})+\frac{1}{\eta}R(W)-\sum_{i=1}^{t-1}\widehat{\ell}_{i}(\widetilde{W}_{t-1},\mathbf{k}_{i},\mathbf{v}_{i})\\ &-\frac{1}{\eta}R(\widetilde{W}_{t-1})-\Bigg\langle\sum_{i=1}^{t-1}\nabla\widehat{\ell}_{i}(\widetilde{W}_{t-1},\mathbf{k}_{i},\mathbf{v}_{i})+\frac{1}{\eta}\nabla R(\widetilde{W}_{t-1}),W-\widetilde{W}_{t-1}\Bigg\rangle.\end{split}$$ (34)

Ignoring the constant terms and using the induction hypothesis ( 33 ), we get

$$\begin{split}\widetilde{W}_{t}=\arg\min_{W}\quad&\widehat{\ell}_{t}(W,\mathbf{k}_{t},\mathbf{v}_{t})+\sum_{i=1}^{t-1}\widehat{\ell}_{i}(W,\mathbf{k}_{i},\mathbf{v}_{i})+\frac{1}{\eta}R(W)\\ &-\Bigg\langle\sum_{i=1}^{t-1}\nabla\widehat{\ell}_{i}(\widehat{W}_{t-1},\mathbf{k}_{i},\mathbf{v}_{i})+\frac{1}{\eta}\nabla R(\widehat{W}_{t-1}),W-\widehat{W}_{t-1}\Bigg\rangle.\end{split}$$ (35)

On the other hand, recall that $\{\widehat{W}_{1},\widehat{W}_{2},\ldots\}$ is obtained by ( FTRL Viewpoint ). Therefore, we have

$$\widehat{W}_{t-1}=\arg\min_{W}\quad\sum_{i=1}^{t-1}\widehat{\ell}_{i}(W;\mathbf{k}_{i},\mathbf{v}_{i})+\frac{1}{\eta}\mathcal{R}_{t}(W).$$

Thus, we have

$$\sum_{i=1}^{t-1}\nabla\widehat{\ell}_{i}(W_{t-1},\mathbf{k}_{i},\mathbf{v}_{i})+\frac{1}{\eta}\nabla R(W_{t-1})=0.$$ (36)

Combining ( 36 ) and ( 35 ), we obtain

$$\widetilde{W}_{t}=\arg\min_{W}\quad\sum_{i=1}^{t}\widehat{\ell}_{i}(W,\mathbf{k}_{i},\mathbf{v}_{i})+\frac{1}{\eta}R(W).$$

This implies $\widetilde{W}_{t}=\widehat{W}_{t}$ , which completes the proof. ∎

### Appendix C Experimental Setup

We perform experimental evaluation on the language modeling ( Merity et al., 2017 ; Paperno et al., 2016 ) , common-sense reasoning ( Bisk et al., 2020 ; Zellers et al., 2019 ; Sakaguchi et al., 2021 ; Clark et al., 2018 ; Clark et al., 2019 ) , and long context needle-in-haystack tasks ( Hsieh et al., 2024 ) . We compare our models with the state-of-the-art linear recurrent models, Transformers, and hybrid models (recurrent + attention). More specifically we compare with Transformer++ ( Touvron et al., 2023 ) , RetNet ( Sun et al., 2023 ) , Gated Linear Attention (GLA) ( Yang et al., 2024a ) , Mamba ( Gu & Dao, 2024 ) , Mamba2 ( Dao & Gu, 2024 ) , DeltaNet ( Yang et al., 2024b ) , TTT ( Sun et al., 2024 ) , and Gated DeltaNet ( Yang et al., 2024 ) .

Table: Table 5 : Architectural Details.

| Model | Block | Dim | Head | Peak LR | Token |
| 170M | 12 | 768 | 16 | 3e-3 | 15B |
| 340M | 24 | 1024 | 16 | 1.5e-3 | 15B |
| 780M | 24 | 1536 | 16 | 1.25e-3 | 30B |
