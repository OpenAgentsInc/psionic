# Chapter 4: Designing the Model Architecture

Designing the Model Architecture
Now that we have our experimental framework in place, it’s time to make the big decisions that will define our model. Every
choice we make, from model size to attention mechanisms to tokenizer, creates constraints and opportunities that will
affect model training and usage.
Remember the training compass: Before making any technical choices, we need clarity on the why and what . Why are we
training this model, and what should it look like?
It sounds obvious, but as we explained earlier, being deliberate here shapes our decisions and keeps us from getting lost in
the endless space of possible experiments. Are we aiming for a SOTA model in English? Is long context a priority? Or are we
trying to validate a new architecture? The training loop may look similar in all these cases, but the experiments we run and
the trade-offs we accept will be different. Answering these questions early helps us decide how to balance our time between
data and architecture work and how much to innovate in each before starting the run.
So, let’s lead by example and walk through the goals that guided SmolLM3’s design. We wanted a strong model for on-
device applications with competitive multilingual performance, solid math and coding capabilities, and robust long context
handling. As we mentioned previously, this led us to a dense model with 3B parameters: large enough for strong
capabilities but small enough to fit comfortably on phones. We went with a dense transformer rather than MoE or hybrid,
given the memory constraints of edge devices and our project timeline (roughly three months).
Once our goals are clear, we can start making the technical decisions that will bring them to life. In this chapter, we’ll go
through our systematic approach to these core decisions: architecture, data, and hyperparameters. Think of this as the
strategic planning phase—getting these fundamentals right will save you from costly mistakes during the actual training
marathon.
Test every change, no matter how small. Don’t underestimate the impact of that seemingly innocent library upgrade or the
commit that “only changed two lines.” These small changes can introduce subtle bugs or performance shifts that will
contaminate your results. You need a library with a strong test suite on the cases that matter to you to avoid regression.
We had a working recipe from SmolLM2 for English at a smaller scale (1.7B parameters), but scaling up meant revalidating
everything and tackling new challenges, like multilinguality and extended context length. Having defined goals shaped our
approach. For example, in SmolLM2, we struggled to extend the context length at the end of pretraining, so for SmolLM3 we
made architectural choices from the start—like using NoPE and intra-document masking (see later)—to maximize our
chances of getting it right, and it worked.

Architecture Choices
If you look at recent models like Qwen3, Gemma 3, or DeepSeek-V3, you’ll see that despite their differences, they all share
the same foundation: the transformer architecture introduced in 2017 (Vaswani et al., 2023). Its fundamental structure
hasn’t changed much over the years, but there have been refinements to its core components. Whether you’re building a
dense model, a mixture of experts model, or a hybrid architecture, you’re working with the same building blocks.
The refinements emerged from teams pushing for better performance and tackling specific challenges: memory constraints
during inference, training instability at scale, the need to handle longer contexts. Some modifications, like shifting from
multi-head attention to more compute-efficient attention variants like grouped query attention (Ainslie et al., 2023), have
been widely adopted. Others, such as different positional encoding schemes, are still being debated. Eventually, today’s
experiments will crystallize into tomorrow’s baselines.
So what do LLMs actually use today? Let’s look at what leading models have converged on. Unfortunately, not all models
disclose their training details, but we have enough transparency from families like DeepSeek, OLMo, Kimi, and SmolLM to
see the current landscape:
If you don’t understand some of these terms yet, such as MLA, NoPE, or WSD, don’t worry; we’ll explain each one in this
section. For now, just notice the variety: different attention mechanisms (MHA, GQA, MLA), positional encodings (RoPE,
NoPE, partial RoPE), and learning rate schedules (cosine, multi-step, WSD).
Looking at this long list of architecture choices, it’s a bit overwhelming to even figure out where to start. As in most such
situations, we’ll take it step by step and gradually build up all the necessary know-how. We’ll focus on the simplest base
architecture first (a dense model) and investigate each architectural aspect in detail. Later, we’ll dive deep into MoE and
hybrid models and discuss when using them is a good choice. Finally, we’ll explore the tokenizer, an often-overlooked and
underrated component. Should we use an existing one or train our own? How do we even evaluate whether our tokenizer is
good?
We’ll begin with the core of every LLM: the attention mechanism.
📍Ablation setup
Throughout the rest of this chapter, we validate most of the architectural choices through ablations using the setup
described in the previoius chapter: our 1B baseline model (following the Llama 3.2 1B architecture) trained on 45B tokens
from a mix of FineWeb-Edu, FineMath, and Python-Edu. For each experiment, we show both training loss curves and
downstream evaluation scores to assess the impact of each modification. You can find the configs for all the runs in
HuggingFaceTB/training-guide-nanotron-configs.
Model ArchitectureParametersTraining TokensAttention Context Length (Final)Pos
DeepSeek LLM 7BDense 7B 2T GQA 4k RoP
DeepSeek LLM 67BDense 67B 2T GQA 4k RoP
DeepSeek-V2MoE 236B (21B active)8.1T MLA 128k Part
DeepSeek-V3MoE 671B (37B active)14.8T MLA 128k Part
MiniMax-01 MoE + hybrid456B (45.9 active)11.4T Linear attention + GQA4M Part
Kimi K2 MoE 1T (32B active)15.5T MLA 128k Part
OLMo 2 7B Dense 7B 5T MHA 4k RoP
SmolLM3 Dense 3B 11T GQA 128k NoP

ATTENTION
One of the most active areas of research around transformer architectures is the attention mechanism. While feedforward
layers dominate compute during pretraining, attention becomes the main bottleneck at inference (especially with long
contexts), where it drives up compute cost and GPU memory requirements, reducing throughput. Let’s take a quick tour
around the main attention mechanisms and how they trade off capacity and speed.
How Many Heads for My Attention?
Multi-head attention (MHA) is the standard attention mechanism introduced with the original transformer architecture
(Vaswani et al., 2023). The main idea is that you have N attention heads each independently doing the same retrieval task:
Transform the hidden state into queries, keys, and values, then use the current query to retrieve the most relevant token by
match on the keys (K), and finally forward the value (V) associated with the matched tokens. At inference time, we don’t
need to recompute the KV values for past tokens; we can reuse them. The memory for past KV values is called the KV
cache . As context windows grow, this cache can quickly become an inference bottleneck and consume a large share of
GPU memory. Here’s a simple calculation to estimate the KV cache size  for the Llama 3 architecture with MHA and a
sequence length of 8,192:
Note that the leading factor of 2 comes from storing both key and value caches. As you can see, the cache size increases
linearly with sequence length—but context windows have grown exponentially, now reaching millions of tokens. Improving
the efficiency of the cache would make scaling context at inference time much easier.
The natural question to ask is: Do we really need new KV values for each head? Probably not, and both multi-query attention
(MQA) (Shazeer, 2019) and grouped query attention (GQA) (Ainslie et al., 2023) address this. The simplest option is to
share the KV values across all heads, thus dividing the size of the KV cache by  (a 64× decrease for Llama 3 70B)!
This is the idea of MQA, and it was used in some models, like StarCoder, as an alternative to MHA. However, this approach
might give away a bit more attention capacity than we are willing to. Another option is to share the KV values across groups
of heads—e.g., we might have four heads sharing the same KV values. This is the GQA approach, which strikes a middle
ground between MQA and MHA.
You can see a visual explanation of each attention mechanism in the following graphic:
s  
KV
   
s  
KV=2×n ×seq×n  ×n  ×dim  
bytes layers heads heads
=2×2×8192×32×32×128=4 GB (Llama 3 8B)
=2×2×8192×80×64×128=20 GB (Llama 3 70B)
( 1 )
n  
heads
More recently, DeepSeek-V2 introduced multi-head latent attention (MLA) (DeepSeek-AI et al., 2024), which uses a different
strategy to compress the cache (also used in V3): Rather than reducing the number of KV values, it reduces their size and
simply stores a latent variable that can be decompressed into KV values at runtime. With this approach, DeepSeek
managed to reduce the cache to the equivalent of GQA with 2.25 groups while giving stronger performance than MHA! To
make it work with RoPE (Rotary Positional Embedding), a small tweak with an extra small latent vector is needed. For
DeepSeek-V2,  was chosen for the main latent variable and  for the RoPE part, for a total of
 . This is used for both K and V simultaneously, thus dropping the leading factor of 2.
4∗dim  
head 1/2∗dim  
head
4.5∗dim  
head

The following table compares the attention mechanisms discussed in this section. For simplicity, we compare the
parameters used per token; if you want to compute total memory, simply multiply by bytes per parameter (typically 2) and
sequence length.
Attention Mechanism KV Cache Parameters per Token
MHA
MQA
GQA
MLA
Now let’s see how these attention mechanisms fare in real experiments!
Ablation—GQA Beats MHA
Changing the number of KV heads affects parameter count, especially for the MHA case. For consistency, we adjust the
number of layers for the MHA run, since it would otherwise have a 100M+ parameter discrepancy; for the rest, we keep the
=2×n  ×heads n  ×layers dim  
head
=2×1×n  ×layers dim  
head
=2×g×n  ×layers dim   (typically g=2,4,8 )head
=4.5×n  ×layers dim  
head
Our baseline model uses 32 query heads and 8 KV heads, which corresponds to GQA with ratio 32 / 8 = 4. How would
performance change if we used MHA, or if we went with fewer KV heads and a higher GQA ratio?
A simplified illustration of multi-head attention (MHA), grouped query attention (GQA), multi-query attention (MQA), and multi-head latent
attention (MLA). Through jointly compressing the keys and values into a latent vector, MLA significantly reduces the KV cache size during
inference.

default 16 layers. We’ll compare MHA, MQA, and four setups for GQA (ratios 2, 4, 8, 16). You can find the Nanotron configs
here.
Attention TypeQuery HeadsKV HeadsLayers Parameter Count Notes
MQA 32 1 16 1.21B
GQA (ratio 16)32 2 16 1.21B
GQA (ratio 8) 32 4 16 1.22B Our baseline
GQA (ratio 4) 32 8 16 1.24B
GQA (ratio 2) 32 16 15 1.22B Reduced layers
MHA 32 32 14 1.20B Reduced layers
GQA (ratio 2) 32 16 16 1.27B Too large—not ablated
MHA 32 32 16 1.34B Too large—not ablated
Looking at the ablation results, we find that MQA and GQA with 16 groups (using only 1 and 2 KV heads, respectively)
underperform MHA significantly. On the other hand, GQA configurations with 2, 4, and 8 groups roughly match MHA
performance:


The results are consistent across both loss curves and downstream evaluations. We observe this clearly in benchmarks like
HellaSwag, MMLU, and ARC, while benchmarks like OpenBookQA and WinoGrande show a bit of noise.
Based on these ablations, GQA is a solid alternative to MHA. It preserves performance while being more efficient at
inference. Some recent models have adopted MLA for even greater KV cache compression, though it hasn’t been as widely
adopted yet. We didn’t ablate MLA, since it wasn’t implemented in Nanotron at the time of the ablations. For SmolLM3, we
used GQA with 4 groups.
Beyond the attention architecture itself, the attention pattern we use during training also matters. Let’s take a look.
Document Masking
How we apply attention across our training sequences impacts both computational efficiency and model performance. This
brings us to document masking and the broader question of how we structure our training samples in the dataloader.
Here’s what this looks like in practice:
During pretraining, we train with fixed sequence lengths, but our documents have variable lengths. A research paper might
have 10k tokens, while a short code snippet might only have a few hundred. How do we fit variable-length documents into
fixed-length training sequences? Padding shorter documents to reach our target length wastes compute on meaningless
padding tokens. Instead, we use packing : We shuffle and concatenate documents with end-of-sequence (EOS) tokens, then
split the result into fixed-length chunks matching the sequence size.


A training sequence might contain one complete file if it’s long enough to fill our 4k context, but most files are shorter than
that, so sequences generally contain concatenations of multiple random files.
With standard causal masking, tokens can attend to all previous tokens in the packed sequence. In the example above, a
token in the Python script of file 4 can attend to the granola bars recipe, the function definition, and the climate change
article.
Let’s take a look at what a typical 4k pretraining context would contain. A quick analysis reveals that a substantial portion
(about 80–90%) of files in the Common Crawl and GitHub datasets are shorter than 2k tokens. The following chart
examines token distributions for the more recent datasets used throughout this guide:
More than 80% of documents in FineWeb-Edu, DCLM, FineMath, and Python-Edu also contain fewer than 2k tokens. This
means with a 2k or 4k training sequence and standard causal masking, the vast majority of tokens would spend compute
attending to unrelated documents packed together.
Longer documents in PDFs
While most web-based datasets consist of short documents, PDF-based datasets contain substantially longer content.
FinePDFs documents are on average 2× longer than web text, and mixing them with FineWeb-Edu and DCLM documents
improves performance.
File 1: "Recipe for granola bars..." (400 tokens) <EOS>
1
File 2: "def hello_world()..." (300 tokens) <EOS>  
2
File 3: "Climate change impacts..." (1000 tokens) <EOS>
3
File 4: "import numpy as np..." (3000 tokens) <EOS>
4
...
5
 
6
After concatenation and chunking into 4k sequences:
7
Sequence 1: [File 1] + [File 2] + [File 3] + [partial File 4]
8
Sequence 2: [rest of File 4] + [File 5] + [File 6] + ...
9


Besides the issue of computational inefficiency, Zhao et al. (2024) find that this approach introduces noise from unrelated
content that can degrade performance. They suggest using intra-document masking , where we modify the attention mask
so tokens can only attend to previous tokens within the same document. The following visualization illustrates the
difference.
Zhu et al. (2025) found similar benefits from intra-document masking in SkyLadder, but they offer a different explanation:
They found that shorter context lengths work better for training, and intra-document masking effectively reduces the average
context length.
These plots from SkyLadder demonstrate multiple findings: (a) shorter contexts often perform better during pretraining (lower validation
perplexity), (b) intra-document masking (IntraDoc) achieves lower perplexity than both random packing (Random) and semantic grouping
(BM25), (c) the shorter context advantage holds even without positional encoding, and (d) IntraDoc creates a distribution skewed toward
shorter effective context lengths.

Meta also trained Llama 3 (Grattafiori et al., 2024) with intra-document masking; they found limited impact during short
context pretraining but significant benefits for long context extension, where the attention overhead becomes more
significant. In addition, the ProLong paper (Gao et al., 2025) showed that using document masking to extend Llama 3 8B’s
context in continual pretraining benefits both long context and short context benchmarks.
We decided to run an ablation on our 1B baseline model and test whether document masking impacted short context
performance. You can find the config here. To enable document masking in Nanotron, simply set the _use_doc_masking
flag to true :
The results showed identical loss curves and downstream evaluation scores compared to standard causal masking, as
shown in the following charts.
model_config:
1
  _attn_implementation: flash_attention_2
2
  _fused_rms_norm: true
3
  _fused_rotary_emb: true
4
- _use_doc_masking: false
5
+ _use_doc_masking: true
6
 
7


As with Llama 3, we don’t observe a noticeable impact on short context tasks, except for a small improvement on PIQA.
However, document masking becomes crucial when scaling to long sequences to speed up the training. This is particularly
important for our long context extension, where we scale from 4k to 64k tokens (as detailed in “The Training Marathon”). We
therefore adopted it for SmolLM3 throughout the full training run.
In this section, we’ve covered how attention processes sequences. Now let’s look at another major parameter block in
transformers: the embeddings.
EMBEDDING SHARING
If you look at the config of our baseline ablation model, one thing that’s different from a standard transformer is embedding
sharing, enabled by the flag tie_word_embeddings .
LLMs have two embedding components: the input embeddings, which serve as a token-to-vector lookup table (of size
vocab_size × hidden_dim ), and the output embeddings, which are the final linear layer mapping hidden states to vocabulary
logits ( hidden_dim × vocab_size ). In the classic case where these are separate matrices, there are a total of 2 ×
vocab_size × hidden_dim embedding parameters. Therefore, in small language models, embeddings can constitute a large
portion of the total parameter count, especially if the vocabulary size is large. This makes embedding sharing (reusing input
embeddings in the output) a natural optimization for small models.


Larger models don’t typically use this technique, since embeddings represent a smaller fraction of their parameter budget.
For example, total embeddings without sharing account for only 13% of the parameters in Llama 3.2 8B and 3% in Llama
3.1 70B, as shown in the following pie chart.


Ablation—Models with Tied Embeddings Match Larger Untied Variants
Let’s assess the impact of embedding sharing on our ablation model. We’ll draw insights from MobileLLM’s comprehensive
ablations on this technique at 125M scale, which demonstrated that sharing reduced the number of parameters by 11.8%
with minimal accuracy degradation.
Since untied embeddings increase our parameter count from 1.2B to 1.46B, we will train another model with untied
embeddings but fewer layers so it matches the baseline 1.2B model in parameter count. We’ll then compare the two 1.2B
models—our baseline with tied embeddings (16 layers) and an untied version with 12 layers—with the 1.46B model with
untied embeddings and the same layer count as our baseline as an additional reference point. You can find the Nanotron
configs here.




The loss and evaluation results demonstrate that our baseline 1.2B model with tied embeddings achieves comparable
performance to the 1.46B untied equivalent on all the benchmarks except for WinoGrande, despite having 18% fewer
parameters. The 1.2B model with untied embeddings and reduced layers (12 vs. 16) underperforms both configurations,
exhibiting higher loss and lower downstream evaluation scores. This suggests that increasing model depth provides greater
benefits than untying embeddings at equivalent parameter budgets.
Based on these results, we kept tied embeddings for our SmolLM3-3B model.
Embeddings alone don’t capture the order of tokens in a sequence, however; providing this information is the role of
positional encodings. In the next section, we will look at how positional encoding strategies have evolved, from standard
RoPE to newer approaches like NoPE (No Positional Embedding) that enable more effective modeling for long contexts.
POSITIONAL ENCODING AND LONG CONTEXT
When transformers process text, they face a fundamental challenge: They naturally have no sense of word order, since they
consume entire sequences simultaneously through parallel attention operations. This enables efficient training but creates
a problem. Without explicit position information, “Adam beats Muon” looks similar to “Muon beats Adam” from the model’s
perspective.
The solution is positional embeddings : mathematical encodings that give each token a unique “address” in the sequence.
But as we push toward longer and longer contexts—from the 512 tokens of early BERT to today’s multi-million-token models
—the choice of positional encoding becomes increasingly critical for both performance and computational efficiency.
The Evolution of Positional Encoding
Early transformers used simple absolute positional embeddings , essentially learned lookup tables that mapped each
position (1, 2, 3…) to a vector that got added to token embeddings **** (Vaswani et al., 2023). This worked fine for short


sequences but had a major limitation: Models’ max input sequence lengths were limited to the max input sequence lengths
they were trained on. They had no out-of-the-box capability to generalize to longer sequences.
ALiBi (Attention with Linear Biases) (Press et al., 2022), in particular, modifies the attention scores based on token
distance. The further apart two tokens are, the more their attention gets penalized through simple linear biases applied to
attention weights. You can find a detailed implementation of ALiBi on labml.ai.
The technique that has dominated recent large language models, however, is RoPE (Su et al., 2023).
RoPE: Position as Rotation
RoPE’s core insight is to encode position information as rotation angles in a high-dimensional space. Instead of adding
position vectors to token embeddings, RoPE rotates the query and key vectors by angles that depend on their absolute
positions.
The intuition is that we treat each pair of dimensions in our embeddings as coordinates on a circle and rotate them by an
angle determined by:
The token’s position in the sequence
Which dimension pair we’re working with (different pairs rotate at different frequencies, which are exponents of a
base/reference frequency)
The field thus evolved toward relative positional embeddings that capture the distance between tokens rather than their
absolute positions. This makes intuitive sense: whether two words are 3 positions apart matters more than whether they’re
at positions (5,8) versus (105,108).

This code might seem complex, so let’s break it down with a concrete example. Consider the word “fox” from the phrase
“The quick brown fox.” In our baseline 1B model, each attention head works with a 64-dimensional query/key vector. RoPE
groups this vector into 32 pairs: (x₁, x₂), (x₃, x₄), (x ₅ , x ₆ ), and so on. We work on pairs because we rotate around circles in
2D space. For simplicity, let’s focus on the first pair, (x₁, x₂). The word “fox” appears at position 3 in our phrase, so RoPE
will rotate this first dimension pair by:
import torch
1
 
2
def apply_rope_simplified(x, pos, dim=64, base=10000):
3
    """
4
    Rotary Positional Embedding (RoPE)
5
 
6
    Idea:
7
    - Each token has a position index p (0, 1, 2, ...).
8
    - Each pair of vector dimensions has an index k (0 .. dim/2 - 1).
9
    - RoPE rotates every pair [x[2k], x[2k+1]] by an angle θ_{p,k}.
10
 
11
    
12
    Formula:
13
      θ_{p,k} = p * base^(-k / (dim/2))
14
 
15
    - Small k (early dimension pairs) → slow oscillations → capture long-range info.
16
    - Large k (later dimension pairs) → fast oscillations → capture fine detail.
17
 
18
    """
19
    rotated = []
20
    for i in range(0, dim, 2):
21
        k = i // 2  # index of this dimension pair
22
 
23
        # Frequency term: higher k → faster oscillation
24
        inv_freq = 1.0 / (base ** (k / (dim // 2)))
25
        theta = pos * inv_freq  # rotation angle for position p and pair k
26
 
27
        cos_t = torch.cos(torch.tensor(theta, dtype=x.dtype, device=x.device))
28
        sin_t = torch.sin(torch.tensor(theta, dtype=x.dtype, device=x.device))
29
 
30
        x1, x2 = x[i], x[i+1]
31
 
32
        # Apply 2D rotation
33
        rotated.extend([x1 * cos_t - x2 * sin_t,
34
                        x1 * sin_t + x2 * cos_t])
35
 
36
    return torch.stack(rotated)
37
    
38
    
39
## Q, K: [batch, heads, seq, d_head]
40
Q = torch.randn(1, 2, 4, 8)
41
K = torch.randn(1, 2, 4, 8)
42
 
43
## 👉  apply RoPE to Q and K *before* the dot product
44
Q_rope = torch.stack([apply_rope(Q[0,0,p], p) for p in range(Q.size(2))])
45
K_rope = torch.stack([apply_rope(K[0,0,p], p) for p in range(K.size(2))])
46
 
47
scores = (Q_rope @ K_rope.T) / math.sqrt(Q.size(-1))
48
attn_weights = torch.softmax(scores, dim=-1)
49

Our base frequency is 10,000, but for the first dimension pair ( k =0) our exponent is 0, so the base frequency doesn’t
affect the calculation (we raise to the power of 0). The following visualization illustrates this.
Now the magic happens, when two tokens interact through attention. The dot product between their rotated representations
directly encodes their relative distance through the phase difference between their rotation angles (where m and n are the
token positions):
rotation_angle = position × θ ₀  
1
                = 3 × (1/10000^(0/32))
2
                = 3 × 1.0 
3
                = 3.0 radians 
4
                = 172° degrees
5


The attention pattern depends only on ( m-n ), so tokens that are five positions apart will always have the same angular
relationship, regardless of their absolute positions in the sequence. Therefore, the model learns distance-based patterns
that work at any absolute position in the sequence and can extrapolate to longer sequences.
Configuring RoPE Frequency
In practice, most LLM pretraining starts with relatively short context lengths (2–4k tokens) using RoPE base frequencies like
10k or 50k. Training with very long sequences from the start would be computationally expensive due to attention’s
quadratic scaling with sequence length and the limited availability of long context data (samples > 4k context length), as we
saw earlier when we looked at document masking. Research also suggests it can hurt short context performance (Zhu et
al., 2025). Models typically start by learning short-range correlations between words, so long sequences don’t help much.
The typical approach is to do most pretraining with shorter sequences, then do continual pretraining or spend the final few
hundred billion tokens on longer sequences. However, as sequence lengths grow, the rotation angles, which are
proportional to token positions, also grow, which can cause attention scores for distant tokens to decay too rapidly (Rozière
et al., 2024; Xiong et al., 2023):
The solution is to increase the base frequency as the sequence length is increased in order to prevent such decaying, using
methods like ABF and YaRN.
RoPE ABF (RoPE with Adjusted Base Frequency) (Xiong et al., 2023b) addresses the attention decay problem in long
contexts by increasing the base frequency in RoPE’s formulation. This adjustment slows down the rotation angles between
token positions, preventing distant tokens’ attention scores from decaying too rapidly. ABF can be applied in a single stage
(direct frequency boost) or multiple stages (gradual increases as context grows). The method is straightforward to
implement and distributes embedded vectors with increased granularity, making distant positions easier for the model to
differentiate.
While simple and effective, ABF’s uniform scaling across all dimensions may not be optimal for extremely long contexts.
YaRN (Yet another RoPE extensioN) (Peng et al., 2023) takes a more sophisticated approach by interpolating frequencies
unevenly across RoPE dimensions using a ramp or scaling function. Unlike ABF’s uniform adjustment, YaRN applies different
scaling factors to different frequency components, optimizing the extended context window. It includes additional techniques
like dynamic attention scaling and temperature adjustment in attention logits, which help preserve performance at very large
context sizes. YaRN enables efficient “train short, test long” strategies, requiring fewer tokens and less fine-tuning for
robust extrapolation. While more complex than ABF, YaRN generally delivers better empirical performance for extremely long
contexts by providing smoother scaling and mitigating catastrophic attention loss. It can also be leveraged in inference
alone without any fine-tuning.
These frequency adjustment methods slow down the attention score decay effect and maintain the contribution of distant
tokens. For instance, the training of Qwen3 involved increasing the frequency from 10k to 1M using ABF as the sequence
length was extended from a 4k to a 32k context window (the team then applied YaRN to reach 131k, 4× extrapolation).
Note that there’s no strong consensus on optimal values, and it’s usually good to experiment with different RoPE values
during the context extension phase to find what works best for your specific setup and evaluation benchmarks.
Most major models today use RoPE: Llama, Qwen, Gemma, and many others. The technique has proven robust across
different model sizes and architectures (dense, MoE, hybrid).
Hybrid Positional Encoding Approaches
As models push toward increasingly large contexts (Meta AI, 2025; Yang et al., 2025), however, even RoPE starts to hit
performance challenges. The standard approach of increasing RoPE’s frequency during long context extension has
dot_product(RoPE(x, m), RoPE(y, n)) = Σ ₖ  [x ₖ  * y ₖ  * cos((m-n) * θ ₖ )]
1
θ = position x 1 / (base^(k/(dim/2)))
1

limitations when evaluated on long context benchmarks more challenging than Needle in a Haystack (NIAH) (Kamradt,
2023), such as RULER and HELMET (Hsieh et al., 2024; Yen et al., 2025). Newer techniques have been introduced to help.
We started this section by saying that transformers need positional information to understand token order, but recent
research has challenged this assumption. What if explicit positional encodings weren’t necessary after all?
NoPE (Kazemnejad et al., 2023) trains transformers without any explicit positional encoding, allowing the model to implicitly
learn positional information through causal masking and attention patterns. The authors show that this approach
demonstrates better length generalization compared to ALiBi and RoPE. Without explicit positional encoding to extrapolate
beyond training lengths, NoPE naturally handles longer contexts. In practice, though, NoPE models tend to show weaker
performance than RoPE models on short context reasoning and knowledge tasks (B. Yang et al., 2025). This suggests that
while explicit positional encodings may limit extrapolation, they provide useful inductive biases for tasks within the training
context length.
Given these trade-offs, B. Yang et al. (2025) suggest that combining different positional encoding strategies might be
interesting. The hybrid approach they introduce, RNoPE, alternates between RoPE and NoPE layers throughout the model.
RoPE layers provide explicit positional information and handle local context with recency bias, while NoPE layers improve
information retrieval across long distances. This technique was recently used in Llama 4, Command A, and SmolLM3.
📍Naming convention
We’ll call RNoPE “NoPE” for the rest of this guide, to keep things simple. (You’ll often see people use “NoPE” to mean
“RNoPE” in discussions of long context models.)
Ablation—NoPE Matches RoPE on ShortContext** Documents
Let’s test the hybrid NoPE approach. We’ll compare a pure RoPE 1B ablation baseline against a NoPE variant that removes
positional encoding every fourth layer and a third configuration combining NoPE with document masking, to test the
interaction between these techniques. Our base question is: Can we maintain strong short context performance while
gaining long context capabilities?



The loss and evaluation results show similar performance across all three configurations, indicating that NoPE maintains
strong short context capabilities while providing the foundation for better long context handling. Given these results, we
adopted the NoPE + document masking combination for SmolLM3.
Another complementary idea is to apply RoPE on only a subset of the model dimension. Unlike RNoPE, which alternates
entire layers between RoPE and NoPE, partial RoPE mixes them within the same layer. Recent models such as GLM‐4.5 (5
Team et al., 2025) and MiniMax-01 (MiniMax et al., 2025) adopt this strategy, but it was also present in older models such
as GPT-J (Wang & Komatsuzaki, 2021). You will also see partial RoPE in every model using MLA, since it’s a must-have for
reasonable inference costs.
🔧Technical explanation: Why partial RoPE is essential for MLA
MLA makes inference efficient with projection absorption: Instead of storing per-head keys  , it caches a small shared
latent  and merges the head’s query/key maps so each score is cheap. With  and 
 , define  to get:
You then compute with  against the tiny cache  (no per-head  is stored). RoPE breaks this because
it inserts a pair-dependent rotation between the two maps. With full-dimension RoPE,
so you can’t pre-merge  and  into a fixed  . Partial RoPE provides the fix: Split head dimensions 
 , apply no rotation on the big block (absorb as before:  ), and apply RoPE on only a small block.
k  
i(h)
c =i x W ∈i c Rd  
c q  =t(h) x W  
t q(h) k  =i(h)
c Ei (h) U =(h) W  Eq(h) (h)
s  =t,i(h)
 (q  ) k  =
 d  
k
1 t(h) ⊤ i(h)
 (x U ) c 
 d  
k
1 t (h) ⊤i
  =q~t(h) x U ∈t (h) Rd  
c c 
i k
s  =t,i(h)
 (x W  ) (c E )
 d  
k
1 t q(h) ⊤
depends on t−i
 R  
t−i i (h)
W  
q(h) E(h) U(h) d  =k
d  +nope d  
rope (x U  ) c 
t nope(h) ⊤i


Limiting Attention Scope for Long Contexts
So far, we’ve explored how to handle positional information for long contexts: activating RoPE, disabling it (NoPE), applying it
partially on some layers (RNoPE) or on some hidden dimensions (partial RoPE), or adjusting its frequency (ABF, YaRN).
These approaches modify how the model encodes position to handle sequences longer than those seen during training. But
there’s a complementary strategy: Instead of adjusting positional encodings, we can limit which tokens attend to each
other.
To see why this matters, consider a model pretrained with sequences of 8 tokens. At inference time, we want to process 16
tokens (more than the training length). Positions 8–15 are out of distribution for the model’s positional encodings. While
techniques like RoPE ABF address this by adjusting position frequencies, attention scope methods take a different
approach: They strategically restrict which tokens can attend to each other, keeping attention patterns within familiar ranges
while still processing the full sequence. This reduces both computational cost and memory requirements.
The following diagram compares five strategies for handling our 16-token sequence with a pretraining window of 8.

Chunked attention divides the sequence into fixed-size chunks, where tokens can only attend within their chunk. In our
example, the 16 tokens are split into two 8-token chunks (0 to 7 and 8 to 15). Notice how tokens 8 through 15 cannot
attend back to the earlier chunk at all. This creates isolated attention windows that reset at chunk boundaries. Llama 4
(Meta AI, 2025) uses chunked attention with 8,192-token chunks in RoPE layers (three out of four decoder layers), while
NoPE layers maintain full context access. This reduces memory requirements by limiting the KV cache size per layer, though
the inability of tokens in one chunk to attend to previous chunks may impact some long context tasks.
Sliding window attention (SWA) , popularized by Mistral 7B (Child et al., 2019; Jiang et al., 2023), takes a different
approach based on the intuition that recent tokens are most relevant. Instead of hard chunk boundaries, each token
attends only to the most recent N tokens. As shown in the diagram, each token can see up to 8 positions back, creating a
sliding window that moves continuously through the sequence. Notice how token 15 can attend to positions 8 through 15,
while token 10 attends to positions 3 through 10. The window slides forward, maintaining local context across the entire


sequence without the artificial barriers of chunking. Gemma 3 combines SWA with full attention in alternating layers, similar
to how hybrid positional encoding approaches mix different strategies.
Dual Chunk Attention (DCA) (An et al., 2024) is a training-free method that extends chunked attention while maintaining
cross-chunk information flow. In our example, we use chunk size s =4, dividing the 16 tokens into 4 chunks (visualize 4×4
squares along the diagonal). DCA combines three mechanisms:
1. Intra-chunk attention, where tokens attend normally within their chunk (the diagonal pattern)
2. Inter-chunk attention, where queries use position index c − 1=7 to attend to previous chunks, creating relative positions
capped at 7
3. Successive chunk attention with local window w =3 that preserves locality between neighboring chunks
This keeps all relative positions within the training distribution (0 to 7) while maintaining smooth transitions across chunk
boundaries. DCA enables models like Qwen2.5 to support ultra-long context windows (up to 1 million tokens) at inference
time, without requiring continual training on million-token sequences.
📊Attention sinks
An interesting phenomenon emerges in transformer models with long contexts: The model assigns unusually high attention
scores to the initial tokens in the sequence, even when they aren’t semantically important. These initial tokens act as a
stabilization mechanism for the attention distribution, serving as a “sink” where attention can accumulate (Xiao et al., 2024).
The practical insight is that keeping the KV cache of just the initial few tokens alongside a sliding window of recent tokens
largely maintains performance when context exceeds the cache size. This simple modification enables models to handle
much longer sequences without fine-tuning or performance degradation.
Modern implementations leverage attention sinks in different ways. The original research suggests adding a dedicated
placeholder token during pretraining to serve as an explicit attention sink. More recently, models like gpt-oss implement
attention sinks as learned per-head bias logits that are appended to the attention scores rather than actual tokens in the
input sequence. This approach achieves the same stabilization effect without modifying the tokenized inputs.
Interestingly, gpt-oss also uses bias units in the attention layers themselves, a design choice rarely seen since GPT-2. While
these bias units are generally considered redundant for standard attention operations (empirical results from Dehghani et al.
show minimal impact on test loss), they can serve the specialized function of implementing attention sinks. The key insight:
Whether implemented as special tokens, learned biases, or per-head logits, attention sinks provide a stable “anchor” for
attention distributions in long context scenarios, allowing the model to store generally useful information about the entire
sequence even as context grows arbitrarily long.
We’ve now covered the core components of attention: the different head configurations that balance memory and compute
(MHA, GQA, MLA), the positional encoding strategies that help models understand token order (RoPE, NoPE, and their
variants), and the attention scope techniques that make long contexts tractable (sliding windows, chunking, and attention
sinks). We’ve also examined how embedding layers should be configured and initialized. These architectural choices define
how your model processes and represents sequences.
But having the right architecture is only half the battle. Even well-designed models can suffer from training instability,
especially at scale. Let’s look at some techniques that help keep training stable.
IMPROVING STABILITY
We’re now going to turn to one of the biggest challenges in LLM pretraining: instabilities. Often manifesting as loss spikes
or sudden jumps in training loss, these issues become especially common at scale.

While we’ll dive deeper into the different types of spikes and how to handle them in “The Training Marathon” (exploring
topics such as floating-point precision, optimizers, and learning rate), certain architectural and training techniques can also
help us reduce instability, so let’s take a moment to study them. We’ll cover a few simple techniques used in recent large-
scale training runs (e.g., OLMo 2 (OLMo et al., 2025) and Qwen3 (A. Yang, Li, et al., 2025)) to improve stability: Z-loss,
removing weight decay from embeddings, and QK-norm.
Z-loss
Z-loss (Chowdhery et al., 2022) is a regularization technique that prevents the final output logits from growing too large by
adding a penalty term to the loss function. The regularization encourages the denominator of the softmax over the logits to
stay within a reasonable range, which helps maintain numerical stability during training:
The ablation results on our 1B model show that adding Z-loss doesn’t impact the training loss or downstream performance.
For SmolLM3, we ended up not using it because our Z-loss implementation introduced some training overhead that we
hadn’t optimized by the time we started training.
L =z-loss λ⋅log(Z)2


Removing Weight Decay from Embeddings
Weight decay is commonly applied to all model parameters as a regularization technique, but OLMo et al. (2025) found that
excluding embeddings from weight decay improves training stability. The reasoning is that weight decay causes embedding
norms to gradually decrease during training, which can lead to larger gradients in early layers since the Jacobian of layer
normalization is inversely proportional to the input norm (Takase et al., 2025).
We tested this approach by training three configurations: our baseline with standard weight decay, a variant with no weight
decay on embeddings, and a third configuration combining all our adopted changes (no weight decay on embeddings +
NoPE + document masking) to ensure there were no negative interactions between techniques. The loss curves and
evaluation results were nearly identical across all three configurations, so we adopted all three changes in SmolLM3
training.




QK-norm
QK-norm (Dehghani et al., 2023) applies layer normalization to both the query and key vectors before computing attention.
This technique helps prevent attention logits from becoming too large and has been used in many recent models to improve
stability.
However, B. Yang et al. (2025) found that QK-norm degrades performance on long context tasks. Their analysis revealed
that QK-norm results in lower attention mass on relevant tokens (needles) and higher attention mass on irrelevant context.
They argue this occurs because the normalization operation removes magnitude information from the query–key dot
product, which makes the attention logits closer in terms of magnitude. For this reason, we didn’t use QK-norm in
SmolLM3. (Additionally, as a small 3B-parameter model, it faces less risk of training instability compared to the larger
models where QK-norm has proven most beneficial.)
ADDITIONAL CONSIDERATIONS
Beyond the components we’ve covered so far, there are a few other architectural decisions worth noting for completeness.
To initialize parameters, modern models typically use truncated normal initialization (mean=0, std=0.02 or std=0.006) or an
initialization scheme like μP (G. Yang & Hu, 2022) (for instance, Cohere’s Command A (Cohere et al., 2025)). This could be
another topic for ablations.
In terms of activation functions, SwiGLU has become a de facto standard in modern LLMs (except Gemma 2, which uses
GeGLU, and NVIDIA, which uses ReLU^2 (Nvidia et al., 2024; NVIDIA et al., 2025)), replacing older choices like ReLU or
GELU.
At a broader scale, architectural layout choices also play a role in shaping model behavior. Although the total parameter
count largely determines a language model’s capacity, how those parameters are distributed across depth and width also


matters. Petty et al. (2024) found that deeper models outperform equally sized wider ones on language modeling and
compositional tasks until the benefit saturates. This “deep and thin” strategy works well for sub-billion-parameter LLMs in
MobileLLM ablations (Z. Liu et al., 2024), whereas wider models tend to offer faster inference thanks to greater parallelism.
Modern architectures reflect this trade-off differently, as Sebastian Raschka notes in his LLM architecture comparison.
We have now covered the most important aspects of the dense transformer architecture worth optimizing for your training
run. However, recently other architectural interventions that concern the model as a whole have emerged, including MoE and
hybrid models. Let’s take a look what they have to offer, starting with MoEs.
GOING SPARSE: MOE
The intuition of mixture of experts models is that we don’t need the full model for every token prediction, similarly to how
our brain activates different areas depending on the task at hand (e.g., the visual or motor cortex). For an LLM, this could
mean that the parts that learned about coding syntax don’t need to be used when the model performs a translation task. If
we can do this well, we can save a lot of compute as it means we only need to run parts of the full model at inference time.
On a technical level, MoEs have a simple goal: Grow total parameters without increasing the number of “active” parameters
for each token. Somewhat simplified, the total parameters impact the total learning capacity of the model, while the active
parameters determine the training cost and inference speed. That’s why you see many frontier systems (e.g., DeepSeek-V3,
Kimi K2, and closed source models like Gemini, Grok, etc.) using MoE architectures these days. The following plot from the
Ling 1.5 paper (L. Team et al., 2025) compares the scaling laws of MoE and dense models.
If this is your first time encountering MoEs, don’t worry, the mechanics are not complicated. Let’s start with the standard
dense architecture and see what changes are necessary for the MoE (figure by Sebastian Raschka):


With MoEs, we replace the single multilayer perceptron (MLP) with multiple MLPs (“experts”) and add a learnable router
before the MLPs. For each token, the router selects a small subset of experts to execute. This is where the distinction
between total parameters and active parameters comes from: The model has many experts, but any given token only uses a
few.
Designing an MoE layer raises a few core questions:
Expert shape & sparsity: Should you use many small experts or fewer large ones? How many experts should be active
per token, and how many do you need in total (i.e., the sparsity or “top- k ”)? Should some experts be universal and thus
always active?
Utilization & specialization: How do you select the routed experts and keep them well used (i.e., avoid idle capacity)
while still encouraging them to specialize? In practice, this is a load-balancing problem that has a significant impact on
training and inference efficiency.
Here, we focus on one objective: Given a fixed compute budget, how do we choose an MoE configuration that minimizes
loss? That’s a different question from pure system efficiency (throughput/latency), and we’ll come back to it later.
Much of this section follows the analysis in Ant Group’s MoE scaling laws paper (Tian et al., 2025). We’ll use their notion of
efficiency leverage (EL) . Simply put, EL measures how much dense compute you’d need to match the loss achieved by an
MoE design where the unit of measurement is floating-point operations (FLOPs). A higher EL means the MoE configuration is
delivering more loss improvement per unit of compute compared to dense training.


"
Let’s take a closer look at how we can set up the sparsity of the MoE to improve the efficiency leverage.
Sparsity/Activation Ratio
Our goal in this section is to find out which MoE setting is best. Asymptotically, it’s easy to see that the two extremes are
not ideal settings. On the one hand, activating all experts all the time brings us back to the dense setting where all
parameters are used all the time. On the other hand, if the number of active parameters is very low (as an extreme, think of
just one parameter being active), clearly it won’t be enough to solve a task even in a narrow domain. So, we need to find
TL;DR: More sparsity → Better FLOPs efficiency →
Diminishing returns at very high sparsity → Sweet spot
depends on your compute budget.


some middle ground. Before we get deeper into finding the optimal setup, it’s useful to define two quantities, the activation
ratio and its inverse, the sparsity:
From a compute perspective, the cost is driven by active parameters only. If you keep the number (and size) of activated
experts fixed and increase the total number of experts, your inference/training FLOPs budget stays more or less the same,
but you’re adding model capacity, so the model should generally be better as long as you train long enough.
There are some interesting empirical takeaways if you survey recent MoE papers: Holding the number and size of active
experts fixed, increasing the total number of experts (i.e., lowering activation ratio/increasing sparsity) improves loss, with
diminishing returns once sparsity gets very high.
Consider the following figures:
The Kimi K2 plot (K. Team et al., 2025) shows both effects: Higher sparsity improves performance, but gains taper off
as sparsity grows.
The Ant Group plot (Tian et al., 2025) shows the same conclusion, with the additional result that higher-sparsity MoEs
benefit more from increasing compute.
activation ratio =  
#total experts#activated experts
sparsity=  =#activated experts
#total experts
 
activation ratio
1



The following table lists the sparsity of some MoE models:
The recent trend is clear: MoE models are getting sparser. That said, the optimal sparsity still depends on hardware and
end-to-end efficiency. For example, Step-3 targets peak efficiency and intentionally doesn’t max out sparsity to fit specific
hardware and bandwidth constraints, while gpt-oss-20b has a low sparsity due to on-device memory constraints (the
passive expert still takes some memory).
Granularity
Beyond sparsity, we need to decide how large each expert should be. This is captured by granularity , a metric introduced by
Ant Group. Let’s pin down what we mean by this term, as terminology varies across papers and some use slightly different
Model Total Experts Activated per Token (Incl. Shared)Sparsity
Mixtral-8×7B 8 2 4.0
Grok-1 8 2 4.0
Grok-2 8 2 4.0
OLMoE-1B-7B-0924 64 8 8.0
gpt-oss 20b 32 4 8
Step-3 48 routed + 1 shared = 493 routed + 1 shared = 4 12.25
GLM-4.5-Air 128 routed + 1 shared = 1298 routed + 1 shared = 9 14.3
Qwen3-30B-A3B 128 8 16.0
Qwen3-235B-A22B 128 8 16.0
GLM-4.5 160 routed + 1 shared = 1618 routed + 1 shared = 9 17.8
DeepSeek-V2 160 routed + 2 shared = 1626 routed + 2 shared = 8 20.25
DeepSeek-V3 256 routed + 1 shared = 2578 routed + 1 shared = 9 28.6
gpt-oss 120b 128 4 32
Kimi K2 384 routed + 1 shared = 3858 routed + 1 shared = 9 42.8
Qwen3-Next-80B-A3B-Instruct512 routed + 1 shared = 51310 total active + 1 shared = 1146.6

formulas. Here, we’ll use the definition that matches the plots we reference:
A higher granularity value corresponds to having more experts with smaller dimension (given a fixed number of parameters).
This metric is a ratio between the expert dimension (  ) and the model dimension (  ).
In dense models, a common rule of thumb is to have the dimension of the MLP set to  . If 
 (following Krajewski et al. (2024)), you can loosely view granularity as how many experts it would take to match the dense
MLP width (  ).
Still, here’s a table with the different values for some recent MoE releases:
Model Year
Mixtral-8x7B 4,096 14,336 0.571 2023
gpt-oss-120b 2880 2880 2.0 2025
gpt-oss-20b 2880 2880 2.0 2025
Grok 2 8,192 16,384 1.0 2024
StepFun Step-3 7,168 5,120 2.8 2025
OLMoE-1B-7B 2,048 1,024 4.0 2025
Qwen3-30B-A3B 2,048 768 5.3 2025
Qwen3-235B-A22B 4,096 1,536 5.3 2025
GLM-4.5-Air 4,096 1,408 5.8 2025
DeepSeek-V2 5,120 1,536 6.6 2024
GLM-4.5 5,120 1,536 6.6 2025
Kimi K2 7,168 2,048 7.0 2025
DeepSeek-V3 7168 2048 7.0 2024
Qwen3-Next-80B-A3B 2048 512 8.0 2025
Let’s talk about how granularity shapes behavior. From Ant Group’s paper:
G=   with α=d  
expert
α∗d  
model 2 or 4
d  
expert d  
model
d  =intermediate 4∗d  
model α=4 4d  =model d  =intermediate Gd  
expert
This interpretation is only a rough heuristic: Modern MoE designs often allocate much larger total capacity than a single
dense MLP, so the one-to-one match breaks down in practice. The Ant Group team chose  , which is simply a
different normalization choice. For consistency, we will pick this convention and stick to it.
α=2
(d )model (d  )expert (G=2d  /d  )model expert

Granularity doesn’t look like the primary driver of EL—it helps, especially for values above 2, but it’s not the dominant factor
determining the loss. And there’s a sweet spot: Pushing granularity higher helps up to a point, and then gains flatten. So,
granularity is a useful tuning knob with a clear trend toward higher values in recent releases, but it shouldn’t be optimized in
isolation.
Another method that is used widely to improve MoEs is the concept of shared experts . Let’s have a look!
Shared Experts
A shared expert setup routes every token to a small set of always-on experts. These shared experts absorb the basic,
recurring patterns in the data so the remaining experts can specialize more aggressively. In practice, you usually don’t need
many of them; model designers commonly choose one, at most two. As granularity increases (e.g., moving from a Qwen3-
style setting to something closer to Qwen3-Next), shared experts tend to become more useful. Looking at the following plot
from Tian et al. (2025), the overall impact is modest; it doesn’t dramatically change the EL. A simple rule of thumb works
well in most cases: Just use one shared expert. This matches choices in models like DeepSeek-V3, Kimi K2, and Qwen3-
Next and tends to maximize efficiency without adding unnecessary complexity.

So, a shared expert is an expert where some tokens are always routed through. What about the other experts? How do we
learn when to route to each one, and make sure that we don’t just use a handful of experts and leave others idle? Next we’ll
discuss load balancing, which tackles exactly that problem.
Load Balancing
Load balancing is the critical piece in MoE. If it’s set up poorly, it can undermine every other design choice. To see why poor
load balancing will cause us a lot of pain, consider a very simple distributed training setup where we have four GPUs, and
we distribute the four experts of our model evenly across the GPUs. If the routing collapses and all tokens are routed to
expert 1, this means that only a quarter of our GPUs are utilized, which is very bad for training and inference efficiency. The
effective learning capacity of our model will also decrease, as not all experts are activated.
To address this issue, we can we can add an extra loss term to the router. Here you can see the standard auxiliary loss–
based load balancing:
This simple formula uses just three factors: the coefficient  determines the strength of the loss,  is the traffic fraction
(the fraction of tokens going through expert  ), and  is the probability mass, which simply sums the probability of the
tokens going through the expert. They are both necessary:  corresponds to the actual balancing, while  is smooth and
differentiable, allowing the gradient to flow.
If we achieve perfect load balancing, we get  . However, we need to be careful how we tune  : If it’s too
small, we don’t guide routing enough, and if it’s too big, routing uniformity becomes more important than the primary
language model loss.
💡Loss-free load balancing
It’s also possible to achieve balancing without an explicit loss term. DeepSeek-V3 (DeepSeek-AI et al., 2025) introduced a
simple bias term added to the affinity scores that go into the routing softmax. If a router is overloaded, the score is
decreased a bit (by a constant factor  ), thus making it less likely to be selected, and it’s increased by  if the expert is
underutilized. This simple adaptive rule achieves load balancing.
A key detail is the scope at which you compute routing statistics: Are  and  computed per local batch (each worker’s
mini-batch) or globally (aggregated across workers/devices)? The Qwen team’s analysis (Qiu et al., 2025) shows that when
there isn’t enough token diversity in each local batch, local computation can hurt both expert specialization (a good proxy for
routing health) and overall model performance. Expert specialization is the phenomenon where one or more experts are
activated more often than others for a specific domain. In other words, if a local batch is narrow, its routing stats become
L  =Bal α  f P 
i=1
∑
N 
r
i i
α f 
i
i P 
i
f 
i P 
i
f =i P =i 1/N 
r α
γ γ
f 
i P 
i

noisy/biased and don’t lead to good balancing. This implies that we should use global statistics (or at least cross-device
aggregation) whenever feasible. Notably, at the time of that paper, many frameworks—including Megatron—computed these
statistics locally by default.
The following plot from the Qwen paper illustrates the difference between micro-batch and global batch aggregation and its
impact on performance and specialization:
Generally, ablating architecture choices around MoE is tricky as there is an interplay with many aspects. For example, the
usefulness of a shared expert might depend on the granularity of the model. It’s worth spending some time to make sure
you have a good set of experiments to really get the insights you are looking for!
We’ve covered the fundamentals of MoEs, but there’s still more to discover. Here’s a non-exhaustive list of items you might
want to explore:
Zero-computation experts, MoE layer rescaling, and training monitoring (the LongCat-Flash paper)
Orthogonal loss load balancing (as in ERNIE 4.5)
Scheduling the load-balancing coefficient over training
Architecture/optimization interactions with MoE, like:
Whether optimizer rankings change for MoE
How to apply μP to MoE
How to adapt the learning rate for MoE (since the experts don’t see the same number of token per batch)
Number of dense layers at the start
We leave it up to you, eager reader, to go further down the rabbit hole, while we now move on to the last major architecture
choice: hybrid models!
EXCURSION: HYBRID MODELS
A recent trend is to augment the standard dense or MoE architecture with state space models (SSMs) or linear attention
mechanisms (MiniMax et al., 2025; Zuo et al., 2025). These new classes of models try to address some of the
fundamental weaknesses of transformers in dealing efficiently with very long contexts. They take a middle ground between
recurrent models, which can process arbitrarily long contexts with linear scaling but may struggle to fully leverage contextual


information, and transformers, which get very expensive at long context lengths but can leverage the patterns in the context
very well.
There have been some studies that aimed to understand the strengths and weaknesses of SSMs. For example, Waleffe et
al. (2024) looked at Mamba models (a form of SSM) and found that they perform well on many benchmarks but
underperform on MMLU and a few other tasks. They hypothesize that it’s the lack of in-context learning causing the gap.
Combining SSMs with blocks from dense or MoE models gives the best of both worlds, thus the name hybrid models .
The core idea behind these linear attention mechanisms is to reorder computations so attention no longer costs  ,
which becomes intractable at long context lengths. How does that work? First, recall the attention formulation at inference.
Producing the output for token  looks like:
Now drop the softmax:
Reordering gives:
We define the running state:
with the simple update:
So we can write:
Why is the reordering important? The left form  means “for each past token  , take a dot  (a scalar),
use it to scale  , and add up those  vectors”—that’s about  work at step  . The right form rewrites this as
 . You keep a single running state matrix  that already summarizes all past
 . Each new token updates it with one outer product  at cost  ; then the output is just one matrix–vector
multiply,  (also  ). So, generating  tokens from scratch with the left form is  , while maintaining  and
using the right form is  . Intuitively: left = “many small dot-scale-adds each step”; right = “one pre-summarized
matrix times the query,” trading dependence on sequence length for dependence on dimension. We focus on inference and
recurring form here, but it’s also more efficient in training, where the reordering is as simple as the following equation:
This now looks very similar to an RNN-like structure. We’ve solved our issue, right? Almost. In practice, the softmax plays an
important stabilizing role, and the naive linear attention can be unstable without some normalization. To address this, a few
practical variants have been proposed.
Lightning Attention
O(nd)2
t
o =t
  
j=1
∑
t
 exp(q  k )∑l=1t t⊤ l
exp(q  k )v 
t⊤ j j
o =t
 (q  k  )v
j=1
∑
t
t⊤ j j
 (q  k  )v  =
j=1
∑
t
t⊤ j j (  v  k  )q .
j=1
∑
t
j j⊤ t
S ≜t
 k  v  =
j=1
∑
t
j j⊤ K  V  ∈1:t⊤ 1:t Rd×d
S =t S  +t−1 k v  
t t⊤
o =t S q   =t t S  q +t−1 t v (k  q )t t⊤t
 (q  k  )v  ∑j≤t t⊤ j j j q  k  
t⊤ j
v  
j t O(td) t(  v  k  )q ∑j≤t j j⊤ t S =t
 v  k  ∈∑j≤t j j⊤ Rd×d
(k  ,v  )j j v k  
t t⊤ O(d)2
S q 
t t O(d)2 T O(Td)2 S 
t
O(Td)2
 V =n×n(QK)⊤ Q  
d×d(KV)⊤

Building on the NormAttention idea (Qin et al., 2022), which replaces softmax normalization with norm-based scaling to
control attention magnitude and improve stability, the Lightning Attention variant focuses on making the implementation fast
and efficient, with a few important architectural tweaks. Here are the formulas for each:
NormAttention:
Lightning Attention:
Step 1: QKV projection + SiLU
Step 2: Decay factor and (  in the previous notation)
Step 3: RMSNorm + gate
Empirically, hybrid models with Lightning Attention match softmax attention on most tasks, according to MiniMax et al.
(2025).
RMSNorm(Q(KV))T
Q=SiLU(Q), K=SiLU(K), V=SiLU(V), G=σ(G)
S =t KV 
t
KV =t λKV  +t−1 k  v 
t⊤ t
o =t q KV 
t t
Y=G⊙RMSNorm(O)

What’s interesting here is that on retrieval tasks like NIAH it can do much much better than full softmax attention, which
might indicate that there is some synergy between the softmax and the linear layer.
MiniMax M2
Surprisingly, the recently released MiniMax-M2 does not use hybrid or linear attention. According to the team’s pretraining
lead, while early MiniMax-M1 experiments with Lightning Attention looked promising at smaller scales on the benchmarks
popular at the time (MMLU, BBH, MATH), they found it had “clear deficits in complex, multi-hop reasoning tasks” at larger
scales. They also cite numerical precision issues during reinforcement learning (RL) training and infrastructure maturity as
key blockers, concluding that designing a new architecture at scale is a multivariable problem that is hard and compute-
intensive due to the sensitivity to other parameters (data distribution, optimizer, etc.). However, they acknowledge that “as
GPU compute growth slows while data length keeps increasing, the benefits of linear and sparse attention will gradually
emerge.” This highlights both the complexity of architecture ablations and the gap between research and production reality.
Now, let’s take a look at some other attention methods and see how they can be understood within a unified framework.
Advanced Linear Attention
A helpful lesson from recurrent models is to let the state occasionally let go of the past. In practice, that means introducing
a gate  for the previous state:G 
t


Almost all recent linear attention mechanisms have this gating component, with different implementations of  . Here’s a
list of variants for the gate and the corresponding architectures from Yang et al., 2024.
One notable variant is Mamba-2 (Dao & Gu, 2024); it’s used in several hybrid models, such as Nemotron-H (NVIDIA, :,
Blakeman, et al., 2025), Falcon H1 (Zuo et al., 2025), and Granite-4.0-h (IBM Research, 2025). However, it’s still early
days, and there’s important nuance to consider when scaling to large hybrid models.
While these attention mechanisms show promise, MiniMax’s experience with M2 highlights that benefits at small scale
don’t always translate to large-scale production systems. That said, hybrid models are moving quickly and remain a solid
choice for frontier training. Qwen3-Next, which includes a gated DeltaNet update (Qwen Team, 2025), reports that they are
faster at inference for long contexts, faster to train, and perform better on standard benchmarks. We’re also looking forward
to Kimi’s next model, which will most likely use their new Kimi Delta Attention. And we should mention Sparse Attention,
which addresses the long-context scaling problem by computing attention only for selected blocks or queries. Some
examples are Native Sparse Attention (Yuan et al., 2025), DeepSeek Sparse Attention (DeepSeek-AI, 2025), and InfLLM v2
(M. Team et al., 2025).
Now that you know what the architecture options are, how do you choose between them? Let’s get that out of the way
before moving on to tokenizers.
TO MOE OR NOT TO MOE: CHOOSING A BASE ARCHITECTURE
The decision of whether to use a dense, MoE, or hybrid model typically depends on a few central factors: where you’ll deploy
the model, your team’s expertise, and your timeline. Let’s briefly go over the pros and cons of each option and build a
simple decision tree for making that choice.
Dense transformers are the standard decoder-only transformer architecture, where every parameter activates for every
token. See the Harvard NLP blog post “The Annotated Transformer” for the math and Jay Alammar’s “The Illustrated
Transformer” to build your intuition.
Pros: Widely supported, well understood, stable training, good performance per parameter.
Cons: Compute scales linearly with size; a 70B model costs ~23× more than 3B.
S =t G ⊙t S  +t−1 v k  
t t⊤
G 
t
Model Parameterization
Mamba (A. Gu & Dao, 2024)
Mamba-2 (Dao & Gu, 2024)
mLSTM (Beck et al., 2025; H. Peng et al., 2021)
Gated Retention (Sun et al., 2024)
DFW (Mao, 2022; Pramanik et al., 2023)
GateLoop (Katsch, 2024)
HGRN-2 (Qin et al., 2024)
RWKV-6 (B. Peng et al., 2024)
Gated Linear Attention (GLA) (Yang et al., 2024)
G =t exp(−(1α )⊙⊤ t exp(A)), αt=softplus(xtWα Wα )1 2 A
G =t γ 11, γ =t ⊤ t exp(−softplus(xtWγ)exp(a)) Wγ
G =t γ 11, γ =t ⊤ t σ(xtWγ) Wγ
G =t γ 11, γ =t ⊤ t σ(xtWγ)
 
τ1 Wγ
G =t α  β , α =t⊤ t t σ(xtWα), β =t σ(xtWβ) Wα
G =t α  1, α =t⊤ t σ(xtWα )exp(xtWα i)1 2 Wα
G =t α  1, α =t⊤ t γ+(1−γ)σ(xtWα) Wα
G =t α  1, α =t⊤ t exp(−exp(xtWα)) Wα
G =t α  1, αt=t⊤ σ(xtWα Wα )1 2
 
τ1 Wα
Gated linear attention formulations of recent models, which vary in their parameterization of $G_t$. The bias terms are omitted.

This is usually the default choice for memory-constrained use cases or new LLM trainers.
Mixture of experts models replace feedforward layers in the transformer with multiple “experts.” A gating network routes
each token to just a few experts. The result is the capacity of a large network at a fraction of the compute. For example,
Kimi K2 has 1T total parameters but only 32B active per token. The catch is that all experts must be loaded in memory. For
a visual guide, check out Maarten Grootendorst’s blog post.
Pros: Better performance per compute for training and inference.
Cons: High memory demands (all experts must be loaded) and more complex training than dense transformers. Framework
support is improving but less mature than for dense models, and distributed training is complex with expert placement, load
balancing, and all-to-all communication challenges.
Use when you’re not memory-constrained and want maximum performance per compute.
Hybrid models combine transformers with state space models like Mamba, offering linear complexity for some operations,
compared with attention’s quadratic scaling. Some useful resources here are Sasha Rush’s blog post and Maarten
Grootendorst’s visual guide to Mamba and SSMs.
Pros: Potentially better long context handling. More efficient for very long sequences.
Cons: Less mature than dense and MoE architectures, with fewer proven training recipes. Limited framework support.
To make your decision, start by asking where your model will be deployed. Then consider your team’s expertise and your
training timeline to assess how much exploration you can afford.
Use if you want to scale to very large contexts while reducing the inference overhead of standard transformers.

For SmolLM3, we wanted to build a strong small model for on-device deployment, we had roughly a three-month timeline,
and we’ve mostly trained dense models in the past. This ruled out MoE (memory constraints) and hybrid (short timeline for
exploration, dense models capable of supporting the target context length of 128k tokens) architectures, so we went for a
dense Llama-style model.
With the model architecture out of the way, let’s now turn our attention to the tokenizer, which forms the bridge between the
data and our model.
THE TOKENIZER
The tokenization scheme is likely one of the most underrated components of any language model. Think of it as the
translator between human language and the mathematical world the model lives in. Though architecture innovations tend to


hog the spotlight, just like with any translator, the quality of the translation matters a lot. So how do we build or choose the
right tokenizer for our needs?
Tokenizer Fundamentals
At its core, a tokenizer converts raw text into sequences of numbers that our model can process, by segmenting a running
text into individual processable units called tokens. Before diving into the technical details, we should first answer some
fundamental questions that will guide our tokenizer design:
What languages do we want to support? If we’re building a multilingual model but our tokenizer has only seen English,
the model will be inefficient when encountering non-English text, which will get split into many more tokens than
necessary. This directly impacts performance, training cost, and inference speed.
Which domains matter to us? Beyond languages, domains like math and code require careful representation of digits.
What is our target data mixture? If we plan to train our tokenizer from scratch, ideally we should train it on a sample that
mirrors our final training mixture.
Vocabulary Size
The vocabulary is essentially a dictionary listing all the tokens (minimal text units, like words, subwords, or symbols) our
model recognizes.
Larger vocabularies typically compress text more efficiently since we generate fewer tokens per sentence, but there’s a
computational trade-off: The vocabulary size directly affects the size of our embedding matrices. If we have vocabulary size
V and hidden dimension h , the input embeddings have V × h parameters, and the output layer has another V × h
parameters. For smaller models, this can amount to a significant chunk of the total parameters (as we saw in “Embedding
Sharing”), but the relative cost shrinks as models scale up.
The sweet spot depends on our target coverage and model size. For English-only models, around 50k tokens usually suffice,
but multilingual models often need 100k+ to efficiently handle diverse writing systems and languages. Modern state-of-the-
art models like Llama 3 have adopted vocabularies in the 128k+ range to improve token efficiency across diverse
languages. Smaller models in the same family apply embedding sharing to reduce the percentage of embedding parameters
while still benefiting from the larger vocabulary.
TokenizationAlgorithm**
Now that we’ve seen the key parameters that define a tokenizer, we face a practical decision: Should we use an existing
tokenizer or train one from scratch? The answer depends on coverage, or whether existing tokenizers with our target
vocabulary size handle our languages and domains well.
Among existing tokenizers, byte-pair encoding (BPE) (Sennrich et al., 2016) remains the most popular choice. Other
algorithms exist, such as WordPiece and SentencePiece, but they are less widely adopted. (There’s also growing research
interest in tokenizer-free approaches that work directly on bytes or characters, potentially eliminating tokenization
altogether.)
As for the question of coverage, consider the following figure, which compares how GPT-2’s English-only BPE tokenizer
(Radford et al., 2019) and Gemma 3’s multilingual SentencePiece tokenizer (G. Team et al., 2025) segment the same
Once we’ve answered these questions, we can examine the main design decisions.
Dagan et al. (2024) analyze the impact of vocabulary size on compression, inference, and memory. They observe that
compression gains from larger vocabularies decrease exponentially, suggesting an optimal size exists. For inference, larger
models benefit from bigger vocabularies because compression saves more on the forward pass than the additional
embedding tokens cost in softmax. For memory, the optimal size depends on sequence length and batch size: Longer
contexts and large batches benefit from larger vocabularies due to KV cache savings from having fewer tokens.

English and Arabic sentences.
While both tokenizers seem to perform similarly on English, the difference becomes striking for Arabic: GPT-2 breaks the
text into over a hundred fragments, while Gemma 3 produces far fewer tokens thanks to its multilingual training data and
larger, more inclusive vocabulary.
But to evaluate a tokenizer’s quality, we can’t just eyeball a few tokenization examples and call it good, the same way we
can’t make architecture changes based on intuition without running ablations. We need concrete metrics.
Measuring Tokenizer Quality
To evaluate how well a tokenizer performs, we can employ two key metrics used in FineWeb2 (Penedo et al., 2025):
Fertility measures the average number of tokens needed to encode a word (the words-to-tokens ratio ). Lower fertility
means better compression, which translates to faster training and inference.


Proportion of continued words tells us what percentage of words get split into multiple pieces. Lower percentages are
better since it means fewer words get fragmented, leading to more efficient tokenization.
The fertility metric is defined around the concept of words because it provides meaningful cross-linguistic comparisons
when appropriate word tokenizers are available, for example in Spacy or Stanza (Penedo et al., 2025). When comparing
tokenizers for a single language, you can use the number of characters or bytes instead of words to get the characters-to-
tokens ratio or bytes-to-tokens ratio (Dagan et al., 2024). However, these metrics have limitations for cross-linguistic
comparison. Bytes can be skewed because characters in different scripts require different byte representations (e.g.,
Chinese characters use 3 bytes in UTF-8 while Latin characters use 1 or 2 bytes). Similarly, using the number of characters
doesn’t account for the fact that words vary dramatically in length across languages. For instance, Chinese words tend to be
much shorter than German compound words.
We can implement these metrics as follows:
Evaluating Tokenizers
To compare tokenizers across different languages, we’ll use the setup from FineWeb2’s tokenizer analysis (Penedo et al.,
2025), using Wikipedia articles as our evaluation corpus. For each language, we’ll sample 100 articles to get a meaningful
sample while keeping computation manageable.
First, let’s install dependencies and define which tokenizers and languages we want to compare:
import numpy as np
1
 
2
def compute_tokenizer_metrics(tokenizer, word_tokenizer, text):
3
    """
4
    Computes fertility and proportion of continued words.
5
    
6
    Returns:
7
        tuple: (fertility, proportion_continued_words)
8
            - fertility: average tokens per word (lower is better)
9
            - proportion_continued_words: percentage of words split into 2+ tokens (lower is 
better)
10
 
11
    """
12
    words = word_tokenizer.word_tokenize(text)
13
    tokens = tokenizer.batch_encode_plus(words, add_special_tokens=False)
14
    tokens_per_word = np.array(list(map(len, tokens["input_ids"])))
15
    
16
    fertility = np.mean(tokens_per_word).item()
17
    proportion_continued_words = (tokens_per_word >= 2).sum() / len(tokens_per_word)
18
    
19
    return fertility, proportion_continued_words
20
For specialized domains like code and math, though, besides fertility we need to dig deeper and look at how well the
tokenizer handles domain-specific patterns. Most modern tokenizers do single-digit splitting, so “123” becomes [“1”, “2”,
“3”] (Chowdhery et al., 2022; DeepSeek-AI et al., 2024). It might seem counterintuitive to break numbers apart, but it
actually helps models learn arithmetic patterns more effectively. If “342792” is encoded as one indivisible token, the model
must memorize what happens when you add, subtract, or multiply that specific token with every other number token, but
when it’s split, the model learns how digit-level operations work. Some tokenizers, like Llama 3’s (Grattafiori et al., 2024),
encode numbers from 1 to 999 as unique tokens, and the rest are composed of these tokens.
pip install transformers datasets sentencepiece 'datatrove[multilingual]'
1
## we need datatrove to load word tokenizers
2

Now let’s load our Wikipedia samples. We use streaming to avoid downloading entire datasets:
With our data ready, we can now evaluate each tokenizer on each language. For each combination, we load the appropriate
word tokenizer from DataTrove and compute both metrics:
tokenizers = [
1
    ("Llama3", "meta-llama/Llama-3.2-1B"),
2
    ("Gemma3", "google/gemma-3-1b-pt"),
3
    ("Mistral (S)", "mistralai/Mistral-Small-24B-Instruct-2501"),
4
    ("Qwen3", "Qwen/Qwen3-4B")
5
]
6
 
7
languages = [
8
    ("English", "eng_Latn", "en"),
9
    ("Chinese", "cmn_Hani", "zh"),
10
    ("French", "fra_Latn", "fr"),
11
    ("Arabic", "arb_Arab", "ar"),
12
]
13
from datasets import load_dataset
1
 
2
wikis = {}
3
for lang_name, lang_code, short_lang_code in languages:
4
wiki_ds = load_dataset("wikimedia/wikipedia", f"20231101.{short_lang_code}", 
streaming=True, split="train")
5
wiki_ds = wiki_ds.shuffle(seed=42, buffer_size=10_000)
6
# Sample 100 articles per language
7
  ds_iter = iter(wiki_ds)
8
  wikis[lang_code] = "\n".join([next(ds_iter)["text"] for _ in range(100)])
9
from transformers import AutoTokenizer
1
from datatrove.utils.word_tokenizers import load_word_tokenizer
2
import pandas as pd
3
 
4
results = []
5
6
for tokenizer_name, tokenizer_path in tokenizers:
7
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
8
    
9
    for lang_name, lang_code, short_lang_code in languages:
10
        word_tokenizer = load_word_tokenizer(lang_code)
11
        
12
        # Compute metrics on Wikipedia
13
        fertility, pcw = compute_tokenizer_metrics(tokenizer, word_tokenizer, wikis[lang_code])
14
        
15
        results.append({
16
            "tokenizer": tokenizer_name,
17
            "language": lang_name,
18
            "fertility": fertility,
19
            "pcw": pcw
20
        })
21
 
22
df = pd.DataFrame(results)
23
print(df)
24

The results reveal some winners and trade-offs, depending on your priorities.
Gemma 3’s tokenizer achieves low fertilities and word-splitting rates across multiple languages—notably English, French,
and Spanish—which can be explained by its multilingual training data and very large vocabulary size (262k tokens, roughly
2× larger than Llama 3’s 128k). Qwen3’s tokenizer excels on Chinese but falls behind Llama 3’s on English, French, and
Spanish. Mistral Small’s tokenizer (Mistral AI, 2025) does best on Arabic but underperforms the others on English and
Chinese.
Choosing Between Existing and Custom Tokenizers
Currently, there’s a good selection of strong tokenizers available. Many recent models start with something like GPT-4’s
tokenizer (OpenAI et al., 2024) and augment it with additional multilingual tokens. Llama 3’s tokenizer performs well on
average across multilingual text and code, while Qwen2.5’s excels particularly on Chinese and some low-resource
languages. How do you decide whether to use one of the available options or train your own? Here are a few guidelines to
help you choose:
When to use existing tokenizers: If your target use case matches the language or domain coverage of the tokenizers
compared above, these are solid choices that have been battle-tested. For SmolLM3 training, we chose Llama 3’s
      tokenizer    language  fertility       pcw
1
0        Llama3     English   1.481715  0.322058
2
1        Llama3     Chinese   1.601615  0.425918
3
2        Llama3      French   1.728040  0.482036
4
3        Llama3     Spanish   1.721480  0.463431
5
4        Llama3  Portuguese   1.865398  0.491938
6
5        Llama3     Italian   1.811955  0.541326
7
6        Llama3      Arabic   2.349994  0.718284
8
7        Gemma3     English   1.412533  0.260423
9
8        Gemma3     Chinese   1.470705  0.330617
10
9        Gemma3      French   1.562824  0.399101
11
10       Gemma3     Spanish   1.586070  0.407092
12
11       Gemma3  Portuguese   1.905458  0.460791
13
12       Gemma3     Italian   1.696459  0.484186
14
13       Gemma3      Arabic   2.253702  0.700607
15
14  Mistral (S)     English   1.590875  0.367867
16
15  Mistral (S)     Chinese   1.782379  0.471219
17
16  Mistral (S)      French   1.686307  0.465154
18
17  Mistral (S)     Spanish   1.702656  0.456864
19
18  Mistral (S)  Portuguese   2.013821  0.496445
20
19  Mistral (S)     Italian   1.816314  0.534061
21
20  Mistral (S)      Arabic   2.148934  0.659853
22
21        Qwen3     English   1.543511  0.328073
23
22        Qwen3     Chinese   1.454369  0.307489
24
23        Qwen3      French   1.749418  0.477866
25
24        Qwen3     Spanish   1.757938  0.468954
26
25        Qwen3  Portuguese   2.064296  0.500651
27
26        Qwen3     Italian   1.883456  0.549402
28
27        Qwen3      Arabic   2.255253  0.660318
29


tokenizer: It offers competitive tokenization quality on our target languages (English, French, Spanish, Portuguese,
German, and Italian) with a modest vocabulary size that made sense for our small model size. For larger models where
embeddings are a smaller fraction of total parameters, Gemma3’s efficiency gains become more attractive.
When to train your own: If you’re training for low-resource languages or have a unique data mixture, you’ll likely need to
train your own tokenizer to ensure good coverage. In this case, it’s important that you train the tokenizer on a dataset
close to what you believe the final training mixture will look like. This creates a bit of a chicken-and-egg problem, since
you need a tokenizer to run data ablations and find the right mixture, but you can retrain the tokenizer before launching
the final run and verify that downstream performance improves and fertilities are still good.
Your choice of tokenizer might seem like a technical detail, but it ripples through every aspect of your model’s performance.
Don’t be afraid to invest time in getting it right.
SMOLLM3
Now that we’ve explored the architectural landscape and run some ablations, let’s see how this all comes together in
practice for a model like SmolLM3.
The SmolLM family is about pushing the boundaries of what’s possible with small models. SmolLM2 delivered three capable
models at 135M, 360M, and 1.7B parameters, all designed to run efficiently on-device. For SmolLM3, we wanted to scale
up performance while staying small enough for phones, and tackle SmolLM2’s weak spots: multilinguality, very long context
handling, and strong reasoning capabilities. We chose 3B parameters as the sweet spot for this balance.
Since we were scaling up a proven recipe, we naturally gravitated toward dense transformers. MoE wasn’t implemented in
Nanotron yet, and we already had the expertise and infrastructure for training strong small dense models. More importantly,
for edge device deployment we’re memory-bound; an MoE with many parameters, even if only a few are active, would be
limiting since we would still need to load all the experts into memory, making dense models the more practical choice.
We started with SmolLM2 1.7B’s architecture as our foundation, then trained a 3B ablation model on 100B tokens using
the Qwen2.5-3B layout. This gave us a solid baseline to test each modification individually. Each architecture change
needed to either improve the loss and downstream performance on English benchmarks or provide measurable benefits,
like inference speed, without quality degradation.
Here’s what we tested before launching the run that made the cut:
Tokenizer: Before diving into architecture modifications, we needed to choose a tokenizer. We found a good set of
candidates that covered our target languages and domains. Based on our fertility analysis, Llama 3.2’s tokenizer gave
us the best trade-off between performance for our six target languages while keeping the vocabulary at 128k—large
enough for multilingual efficiency but not so large that it bloated our 3B parameter count with embedding weights.
Grouped query attention: We reconfirmed our earlier finding that GQA with four groups matches MHA’s performance, but
this time at 3B scale with 100B tokens. The KV cache efficiency gains were too good to pass up, especially for on-
device deployment where memory is precious.
NoPE for long context: We implemented a hybrid positional encoding scheme, with NoPE applied every fourth layer and
RoPE used in the remaining layers. Our 3B ablation confirmed our previous findings: NoPE improved long context
handling without sacrificing short context performance.
Intra-document attention masking: We prevented cross-document attention during training to help with training speed
and stability when training on very large sequences. Again, we found that this didn’t impact downstream performance.
Model layout optimization: We compared layouts from recent 3B models in the literature, some prioritizing depth, others
width. We tested Qwen2.5-3B (3.1B), Llama 3.2-3B (3.2B), and Falcon3-H1-3B (3.1B) layouts on our training setup,
where depth and width varied. The results were interesting: All layouts achieved nearly identical loss and downstream
performance, despite Qwen2.5-3B actually having fewer parameters, but Qwen2.5-3B’s deeper architecture aligned with

"
research showing that network depth benefits generalization (Petty et al., 2024). Therefore, we went with the deeper
layout, betting it would help as training progressed.
Stability improvements: We kept tied embeddings from SmolLM2 but added a new trick inspired by OLMo 2, removing
weight decay from embeddings. Our ablations showed this didn’t hurt performance while lowering embedding norms,
which can help prevent training divergence.
The beauty of the systematic ablations approach is that we could confidently combine all these modifications, knowing each
had been validated.
💡Combining changes in ablations
In practice, we test changes incrementally: Once a feature is validated, it becomes part of the baseline for testing the next
feature. Testing order matters. Start with the battle-tested features first (tied embeddings → GQA → document masking →
NoPE → remove weight decay).
RULES OF ENGAGEMENT
Let your deployment target guide architectural decisions. Consider how and where your model will actually run when
evaluating new architectural innovations.
Strike the right balance between innovation and pragmatism. We can’t afford to ignore major architectural advances—using
MHA when better alternatives (like GQA) exist would be a poor technical choice. Stay informed about the latest research and
adopt techniques that offer clear, validated benefits at scale. But resist the temptation to chase every new paper that
promises marginal gains (unless you have the resources to do so or your goal is architecture research).
Systematic beats intuitive. Validate every architecture change, no matter how promising it looks on paper. Test
modifications individually before combining them to understand their impact.
Scale effects are real—re-ablate at target size when possible. Don’t assume your small-scale ablations will hold perfectly at
your target model size. If you have the compute, try to reconfirm them.
Validate tokenizer efficiency on your actual domains. Fertility metrics across your target languages and domains matter
more than following what the latest model used. A 50k English tokenizer won’t cut it for serious multilingual work, but you
don’t need a 256k vocabulary either if you’re not covering that many languages.
Now that the model architecture is decided, it’s time to tackle the optimizer and hyperparameters that will drive the learning
process.
