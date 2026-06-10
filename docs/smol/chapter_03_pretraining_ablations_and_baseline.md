# Chapter 3: Every Big Model Starts with a Small Ablation

Every Big Model Starts with a Small Ablation,” come in.
📍Learn to identify what's worth testing, not just how to run tests.
Perfect ablations on irrelevant choices waste as much compute as sloppy ablations on important ones.
In the following chapters, you will learn about all the options you have to define your model and how to narrow down the
choices with systematic experiments. Before going there, we want to share a few learnings on how to set up teams and
projects that we’ve gained from training our own models as well as observing other amazing teams building great LLMs.
Superpower: Speed and Data
Of course, there are many ways to get to Rome, but we’ve found that what consistently sets successful LLM training teams
apart is iteration speed . Training LLMs is really a learning-by-doing discipline: The more often you train, the better your
team will become. So, between the teams that train a model per year and the ones that train one per quarter, the latter will
improve much faster. You can look at the teams from Qwen and DeepSeek for examples. Now household names, they have
a long track record of consistently releasing new models on a fast cadence.
Besides iteration speed, by far the most influential aspect of LLM training is data curation . There’s a natural tendency to
dive into architecture choices to improve the model, but the teams that excel in LLM training are the ones that are
obsessed with high-quality data more than anything else.
Another aspect that is tied to iteration speed is team size . For the main pretraining tasks, you only need a handful of
people equipped with enough compute to execute. To pretrain a model like Llama 3 today, you probably only need two or
three people. Once you start to venture into more diverse trainings and downstream tasks (multimodal, multilingual, post-
training, etc.), you will need to add a few more people to excel at each domain.
So, start with a small, well-equipped team and build a new model every two or three months, and within a short amount of
time you’ll climb to the top. The rest of this guide will focus on the technical day-to-day activities of this team!
Every Big Model Starts with a Small Ablation
Before we can start training an LLM, we need to make many decisions that will shape the model’s performance and training
efficiency. Which architecture will best serve our use case? What optimizer and learning rate schedule should we use, and
which data sources should we mix in?
How these decisions are made is a frequently asked question. People sometimes expect that they require deep thought.
And while strategic thinking is essential—as we covered in the previous section—reasoning alone isn’t enough. Things are
not always intuitive with LLMs, and hypotheses about what should work sometimes don’t pan out in practice.
For example, using what seems like “the highest-quality data” doesn’t always yield stronger models. Take arXiv for example,
which is a vast collection of humanity’s scientific knowledge. Intuitively, training on such rich STEM data should produce
superior models, right? In practice, it doesn’t, especially for smaller models, where it can even hurt performance (Shao et

al., 2024). Why? While arXiv papers are full of knowledge, they’re highly specialized and written in a narrow academic style
that’s quite different from the diverse, general text that models learn best from.
Since those experiments will guide many of our crucial decisions, it’s really important to set them up well. There are
essentially two main attributes we want from them:
1. Speed: They should run as fast as possible so we can iterate often. The more ablations we can run, the more
hypotheses we can test.
2. Reliability: They should provide strong discriminative power. If the metrics we look at can’t meaningfully distinguish
between different setups early on, our ablations may reveal little (and if they’re noisy, we risk chasing noise!). For more
details, check out our blog post on scaling FineWeb to 1000+ languages.
But before we can set up our ablations, we need to make some foundational choices about architecture type and model
size. These decisions—guided by our compass—impact which training framework to use, how to allocate our compute
budget, and which baseline to start from.
For SmolLM3, we went with a dense Llama-style architecture at 3B parameters because we were targeting small on-device
models. But as you’ll see in “Designing the Model Architecture,” a mixture of experts (MoE) or hybrid model might be better
suited for your use case, and different model sizes come with different trade-offs. We’ll explore these choices in depth later,
and show you how to make these decisions. For now, let’s start with the most practical first step: choosing your baseline.
Choosing Your Baseline
Every successful model builds on a proven foundation and modifies it for its needs. When Qwen trained their first model
family (Bai et al., 2023), they started from Llama’s architecture. When Meta trained Llama 3, they started from Llama 2.
Kimi K2 started from DeepSeek-V3’s MoE architecture. This applies not only to architectures, but also to training
hyperparameters and optimizers.
Why? Designing good architectures and training setups takes years of iteration across many organizations. The standard
transformer and optimizers like Adam have been refined through thousands of experiments. People have found their failure
modes, debugged instabilities, optimized implementations. Starting from a proven foundation means inheriting all that
accumulated knowledge. Starting fresh means rediscovering every problem yourself.
To make a good starting point, an architecture should:
Match your constraints , aligning with your deployment target and use case
Be proven at scale (demonstrated in multi-trillion-token runs at similar or larger sizes)
Be well-documented , with known hyperparameters that have been proven to work in open models
Have good framework support (ideally, it should be supported in the training frameworks you are considering and the
inference frameworks you are planning to use)
Here’s a non-exhaustive list of strong 2025 baseline options for various architectures and model sizes, at the time of
writing (early 2026):
So, how can we know what works if staring at the problem long and hard doesn’t help? We run a lot of experiments, like
good empiricists! Machine learning is not pure math, but actually very much an experimental science.

Architecture Type Model Family Sizes
Dense Llama 3.1 8B, 70B
Dense Llama 3.2 1B, 3B
Dense Qwen3 0.6B, 1.7B, 4B, 14B, 32B
Dense Gemma 3 12B, 27B
Dense SmolLM2, SmolLM3 135M, 360M, 1.7B, 3B
MoE Qwen3 MoE 30B-A3B, 235B-A122B
MoE gpt-oss 21B-A3B, 117B-A5B
MoE Kimi Moonlight 16B-A3B
MoE Kimi K2 1T-A32B
MoE DeepSeek-V3 671B-A37B
Hybrid Zamba2 1.2B, 2.7B, 7B
Hybrid Falcon-H1 0.5B, 1.5B, 3B, 7B, 34B
MoE + hybrid Qwen3-Next 80B-A3B
MoE + hybrid MiniMax-01 456B-A46B
MoE + hybrid MiMo-V2-Flash 309B-A15B
Find your architecture type, and pick a baseline close to the number of parameters you’d like your model to have. Don’t
overthink it too much, as the architecture you start from is not set in stone. In the next section, you’ll see how to go from a
baseline to a final architecture that is optimal for you.
MODIFYING YOUR BASELINE: THE DISCIPLINE OF DERISKING
You now have a baseline that works and fits your use case. You could stop here, train it, and (assuming your data mixture
is good) likely get a decent model. Many successful projects do exactly that. But baselines aren’t optimized for your specific
constraints; they’re designed for the use cases and deployment targets of whoever built them. This means there are
probably modifications worth making to better align with your goals. However, every architectural change carries risk: It
might boost performance, tank it, or do nothing while wasting your ablation compute.
The discipline that keeps you on track is derisking : Never change anything unless you’ve tested that it helps.
📍What counts as derisked?
A change is derisked when testing shows it either improves performance on your target capabilities or provides a meaningful
benefit (e.g., faster inference, lower memory, better stability) without hurting performance beyond your acceptable limits.
The tricky part is that your baseline and training setup have many components you could modify: attention mechanisms,
positional encodings, activation functions, optimizers, training hyperparameters, normalization schemes, model layout, and
more. Each represents a potential experiment, and these components often interact in nonlinear ways. You have neither the
time nor the compute to test everything or explore every interaction.
So, start by testing promising changes against your current baseline. When something works, integrate it to create a new
baseline, then test the next change against that. If your compute budget allows it, you could test changes individually and

🎯Strategic experimentation
Knowing how to run experiments isn’t enough; you need to know which experiments are worth running. Ask yourself two
questions before testing any modification:
Will this help my specific use case?
Will this optimize my training?
If a modification doesn’t clearly address either goal, skip it.
Now that you know how to identify what’s promising through strategic planning, it’s time to move on to empirical validation.
In the next sections, we’ll show you how to actually test these changes in practice. We’ll cover how to set up reliable
experiments, interpret results, and avoid common pitfalls. Then, in the following chapters, we’ll walk through concrete
examples of testing popular architectural, data, infrastructure, and training decisions.
Picking a Training Framework
This choice involves balancing three key considerations:
1. The framework must support our target architecture, or let us easily extend it.
2. It needs to be stable and production-ready, and not prone to mysteriously breaking midway through training.
3. It should deliver strong throughput so we can iterate quickly and make the most of our compute budget.
In practice, these requirements might pull against each other, creating trade-offs. Let’s look at some of the available
options:
This table summarizes the key trade-offs between popular frameworks. Lines of code for the first three frameworks are from
the TorchTitan technical report (Liang et al., 2025). Let’s discuss each in more detail:
Megatron-LM from NVIDIA has been around for years and is battle-tested. It’s what powers models like Kimi’s K2 (Team
et al., 2025); it delivers solid throughput and has most of the production features we’d want. But that maturity comes
with complexity: The codebase can be hard to navigate and modify when you’re new to it.
DeepSpeed falls into a similar category. It’s the pioneer of ZeRO optimization and powered models like BLOOM and
GLM. Like Megatron-LM, it’s extensively battle-tested and optimized, but it shares the same complexity challenges. The
large codebase (194k total lines) can be intimidating when you’re getting started, particularly for implementing custom
features or debugging unexpected behavior.
run a leave-one-out analysis. Don’t fall into the trap of running exhaustive grid searches over every hyperparameter or
testing every architectural variant that comes out.
The first decision we need to make is which framework to use for training our model and, by extension, for running all our
ablations.
Framework Features Battle-Tested Optimize
Megatron-LM✅  Extensive ✅  Kimi K2, Nemotron ✅  Pioneers of 3D parallelism
DeepSpeed✅  Extensive ✅  BLOOM, GLM ✅  Pioneers of ZeRO & 3D paralle
TorchTitan⚡  Growing feature set ⚠  Newer but tested by PyTorch team⚡ Optimized for dense models, Mo
Nanotron🎯  Minimal, tailored for HF pretraining✅  Yes (StarCoder, SmolLM)✅  Optimized (UltraScale Playbook

On the other side, PyTorch’s recent TorchTitan library is much lighter and simpler to navigate, thanks to its compact and
modular codebase. It has the core features needed for pretraining and is great for rapid experimentation. However, being
newer, it isn’t as battle-tested and can still be a bit unstable as it’s actively developed.
Nanotron is a framework we built from scratch. This gave us full flexibility and a deep understanding of large-scale
pretraining—insights that later evolved into the Ultra-Scale Playbook. Since we open sourced the library, we also got
valuable feedback from the community, though for most cases we had to battle-test features ourselves first. The
framework now supports all the production features we need for training, but we’re still building out areas like MoE
support.
Building from scratch made sense in our case, but it demands major investment in team expertise and time to debug
issues and add missing features. A strong alternative is forking an existing framework and enhancing it for your needs. For
example, Thinking Machines Lab built their internal pretraining library as a fork of TorchTitan.
Ultimately, your choice will depend on your team’s expertise, your target features, and how much time you’re willing to invest
in development versus using the most production-ready option.
If multiple frameworks support your needs, compare their throughput on your specific hardware. For quick experiments and
speed runs, simpler codebases often win.
Ablation Setup
With the framework chosen, we now need to design our ablation setup. We need experiments that are fast enough to iterate
on quickly, but large enough that the results give us signal and transfer to the final model.
SETTING UP OUR ABLATION FRAMEWORK
The goal of ablations is to run experiments at a small scale and get results we can confidently extrapolate to our final
production run.
There are two main approaches. First, we can take our target model size and train it on fewer tokens. For the SmolLM3
ablations, we trained the full 3B model on 100B tokens instead of the final 11T. Second, if our target model is too large, we
can train a smaller proxy model for ablations. For example, when Kimi were developing their 1T parameter K2 model with
32B active parameters, using the full size for all ablations would have been prohibitively expensive, so they ran some
ablations on a 3B MoE with 0.5B active parameters (Team et al., 2025).
One key question is whether these small-scale findings actually transfer. In our experience, if something hurts performance
at small scale, you can confidently rule it out for large scale. But if something works at small scale, you’ll want to make sure
you’ve trained on a reasonable number of tokens to conclude with high probability that these findings will extrapolate to
larger scales. The longer you train and the closer the ablation models are to the final model, the better.
Our baseline 1B config captures all the essential training details in a structured YAML format. Here are the key sections:
We decided to use a baseline vanilla transformer for all ablations. Our main setup is a 1B transformer following the Llama
3.2 1B architecture trained on 45B tokens. This takes about a day and a half to train on a node of eight H100s using this
Nanotron config (42k tokens per second per GPU).

## Datasets and mixing weights
1
data_stages:
2
- data:
3
 
4
    dataset:
5
      dataset_folder:
6
      - fineweb-edu
7
      - stack-edu-python
8
      - finemath-3plus
9
 
10
      dataset_weights:
11
      - 0.7
12
      - 0.2
13
      - 0.1
14
 
15
## Model architecture, Llama 3.2 1B configuration
16
model:
17
  model_config:
18
    hidden_size: 2048
19
    num_hidden_layers: 16
20
    num_attention_heads: 32
21
    num_key_value_heads: 8  
22
    intermediate_size: 8192
23
    max_position_embeddings: 4096
24
    rope_theta: 50000.0
25
    tie_word_embeddings: true
26
 
27
## Training hyperparameters, AdamW with cosine schedule
28
optimizer:
29
  clip_grad: 1.0
30
  learning_rate_scheduler:
31
    learning_rate: 0.0005
32
    lr_decay_starting_step: 2000
33
    lr_decay_steps: 18000
34
    lr_decay_style: cosine
35
    lr_warmup_steps: 2000
36
    lr_warmup_style: linear
37
    min_decay_lr: 5.0e-05
38
  optimizer_factory:
39
    adam_beta1: 0.9
40
    adam_beta2: 0.95
41
    adam_eps: 1.0e-08
42
    name: adamW
43
 
44
## Parallelism, 1 node
45
parallelism:
46
  dp: 8  # Data parallel across 8 GPUs
47
  tp: 1  # No tensor or pipeline parallelism needed at 1B scale
48
  pp: 1 
49
 
50
## Tokenizer
51
tokenizer:
52
  tokenizer_max_length: 4096
53
  tokenizer_name_or_path: HuggingFaceTB/SmolLM3-3B
54
 
55
## Batch size, sequence length and total training for 30B tokens
56
tokens:
57
  batch_accumulation_per_replica: 16
58
  micro_batch_size: 3 # GBS (global batch size)=dp * batch_acc* MBS * sequence=1.5M tokens
59
  sequence_length: 4096
60
  train_steps: 20000 # GBS * 20000 = 30B
61

For our ablations, we’ll modify different sections depending on what we’re testing while keeping everything else constant:
the model section for architecture choices, the optimizer section for optimizer and training hyperparameters, and the
data_stages section for data curation.
☝Modify one thing at a time
Change only one variable per ablation, while keeping everything else constant. If you change multiple things and performance
improves, you won’t know what caused it. Test modifications individually, then combine successful ones and reassess.
When running ablations, some architectural changes can significantly alter the parameter count. For instance, switching
from tied to untied embeddings doubles our embedding parameters, while using grouped query or multi-query attention
instead of multi-head attention decreases our attention parameters substantially (we’ll talk about all of these things
shortly).
To ensure fair comparisons, we need to track parameter counts and occasionally adjust other hyperparameters (like hidden
size or layer count) to keep model sizes roughly the same. Here is a simple function that we use to estimate parameter
counts for different configurations:
We also provide an interactive tool to visualize LLM parameter distributions, in the case of a dense transformer. This can
come in handy when making architecture decisions or setting up configs for ablations.
 
62
...
63
from transformers import LlamaConfig, LlamaForCausalLM
1
 
2
def count_parameters(
3
    tie_embeddings=True,
4
    num_key_value_heads=4,
5
    num_attention_heads=32,
6
    hidden_size=2048,
7
    num_hidden_layers=16,
8
    intermediate_size=8192,
9
    vocab_size=128256,
10
    sequence_length=4096,
11
):
12
    config = LlamaConfig(
13
        hidden_size=hidden_size,
14
        num_hidden_layers=num_hidden_layers,
15
        num_attention_heads=num_attention_heads,
16
        num_key_value_heads=num_key_value_heads,
17
        intermediate_size=intermediate_size,
18
        vocab_size=vocab_size,
19
        max_position_embeddings=sequence_length,
20
        tie_word_embeddings=tie_embeddings,
21
    )
22
    model = LlamaForCausalLM(config)  
23
    return f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B"
24

UNDERSTANDING WHAT WORKS: EVALUATION
Once we launch our ablations, how do we know what works and what doesn’t?
The first instinct of anyone who trains models might be to look at the loss, and yes, that is important. You want to see it
decreasing smoothly, without wild spikes or instability. For many architectural choices, the loss correlates well with
downstream performance and can be sufficient (Y. Chen et al., 2025). However, looking only at the loss is not always
reliable. Taking the example of data ablations, you would find that training on Wikipedia gives a lower loss than training on
web pages (the next token is easier to predict), but that doesn’t mean you’d get a more capable model. Similarly, if we
change the tokenizer between runs, the losses aren’t directly comparable since text gets split differently. Some changes
might also specifically affect certain capabilities, like reasoning and math, but get washed away in the average loss. Last


but not least, models can continue improving on downstream tasks even after pretraining loss has converged (Liu et al.,
2022).
We need more fine-grained evaluation to see the full picture and understand these nuanced effects. A natural approach is
to use downstream evaluations that test knowledge, understanding, reasoning, and whatever other domains matter for us.
For these ablations, it’s useful to focus on tasks that give good early signal and avoid noisy benchmarks. In FineTasks and
FineWeb2, reliable evaluation tasks are defined by four key principles:
Monotonicity: The benchmark scores should consistently improve as models train longer.
Low noise: When we train models with the same setup but different random seeds, the benchmark scores shouldn’t vary
wildly.
Above-random performance: Many capabilities only emerge later in training, so tasks that show random-level
performance for extended periods aren’t useful for ablations. This is the case, for example, for MMLU in multiple choice
format, as we will explain later.
Ranking consistency: If one approach outperforms another at early stages, this ordering should remain stable as training
continues.
The quality of a task also depends on the task formulation (how we ask the model questions) and metric choice (how we
compute the answer score).
Three common task formulations are multiple choice format (MCF) , cloze formulation (CF) , and free-form generation (FG) .
MCF requires models to select an option from a number of choices explicitly presented in the prompt and prefixed with
A/B/C/D (as is done in MMLU, for example). In CF, we compare the likelihood of the different choices to see which one is
more likely without having provided them in the prompt. In FG, we look at the accuracy of the greedy generation for a given
prompt. FG requires a lot of latent knowledge in the model and is usually too difficult a task for the model to be really useful
in short pretraining ablations before a full training run. We thus focus on multiple choice formulations when running small-
sized ablations (MCF or CF).
📍Heads‐up
For post-trained models, FG becomes the primary formulation since we’re evaluating whether the model can actually generate
useful responses. We’ll cover evaluation for these models in “Beyond Base Models—Post-Training.”
Our ablations evaluation suite includes the benchmarks from FineWeb ablations, except for SIQA, which we found to be too
noisy. We add math and code benchmarks like GSM8K and HumanEval and the long context benchmark RULER for long
context ablations. This aggregation of tasks tests world knowledge, reasoning, and common sense across a variety of
formats, as shown in the following table. To speed up evaluations at the expense of some additional noise, we only
evaluate on 1,000 questions from each benchmark (except for GSM8K, HumanEval, and RULER, which we use in full for the
3B SmolLM3 ablations but omit from the 1B experiments). We use CF to evaluate all multiple-choice benchmarks, as
explained previously. Note that for multilingual ablations and actual training, we add more benchmarks to test
multilinguality, which we detail later. These evaluations are run using LightEval. Here’s a summary of the key characteristics
of each benchmark:
Research has also shown that models struggle with MCF early in training, making CF better for early signal (Du et al., 2025;
Gu et al., 2025; J. Li et al., 2025). We thus use CF for small ablations and integrate MCF in the main run (as it gives better
signal at later stages of training). To score a model’s answer in sequence likelihood evaluations like CF, we compute
accuracy as the percentage of questions where the the correct answer has the highest log probability normalized by
character count. This normalization prevents a bias toward shorter answers.

Benchmark Domain Task Type Questions What It Tests
MMLU Knowledge Multiple choice14k Broad academic knowledge across 57 subject
ARC Science & reasoningMultiple choice7k Grade school science reasoning
HellaSwag Common-sense reasoningMultiple choice10k Common-sense reasoning about everyday situ
WinoGrandeCommon-sense reasoningBinary choice 1.7k Pronoun resolution requiring world knowledge
CommonSenseQACommon-sense reasoningMultiple choice1.1k Common-sense reasoning about everyday con
OpenBookQAScience Multiple choice500 Elementary science facts with reasoning
PIQA Physical common senseBinary choice 1.8k Physical common sense about everyday object
GSM8K Math Free-form generation1.3k Grade school math word problems
HumanEval Code Free-form generation164 Python function synthesis from docstrings
Let’s look at a few example questions from each to get a concrete sense of what these evaluations actually test:
↗View interactive version
You can browse through the examples in the interactive version to see the types of questions in each benchmark. Notice
how MMLU and ARC test factual knowledge with multiple choices, GSM8K requires computing numerical answers to math
problems, and HumanEval requires generating complete Python code. This diversity ensures we’re testing different aspects
of model capability throughout our ablations.


"
Which Data Mixture for the Ablations?
For architecture ablations , we train on a fixed mix of high-quality datasets that provide early signal across a wide range of
tasks. We use English (FineWeb-Edu), math (FineMath), and code (Stack-Edu-Python). Architectural findings should
extrapolate well to other datasets and domains, including multilingual data, so we can keep our data mixture simple.
The real value of a solid ablation setup goes beyond just building a good model. When things go wrong during our main
training run (and they will, no matter how much we prepare), we want to be confident in every decision we made and able to
quickly identify which components weren’t tested. This preparation saves debugging time and protects our future mental
sanity.
ESTIMATING ABLATION COST
Ablations are amazing, but they require GPU time. It’s worth understanding the cost of these experiments. The following
table shows our complete compute breakdown for SmolLM3 pretraining: the main run (accounting for occasional downtime),
ablations before and during training, plus compute spent on an unexpected scaling issue that forced a restart and some
debugging (which we’ll detail later).
The numbers reveal an important fact: Ablations and debugging consumed a total of 161,280 GPU hours, more than half
the cost of our main training run (276,480 GPU hours). We ran over 100 ablations across SmolLM3’s development: We
spent 20 days on pretraining ablations, 10 days on mid-training ablations, and 7 days recovering from the aforementioned
unexpected training issue.
Before we move on to designing the model architecutre, let’s establish some ground rules that every person running
experiments should follow.
Rules of Engagement
Validate your evaluation suite. Before training any models, make sure your evaluation suite can reproduce the published
results of models you will compare against. If any benchmarks are generative in nature (e.g., GSM8k), be extra paranoid
and manually inspect a few samples to ensure that the prompt is formatted correctly and that any post-processing is
For data ablations , we take the opposite approach: We fix the architecture and systematically vary the data mixtures to
understand how different data sources affect model performance.
Phase GPUs Days GPU Hours
Main pretraining run 384 30 276,480
Ablations (pretraining) 192 15 69,120
Ablations (mid-training) 192 10 46,080
Training reset / debugging 384 / 192 3 / 4 46,080
Total cost - - 437,760
This highlights why ablation costs must be factored into your compute budget: Plan for training cost plus ablations plus a
buffer for surprises. If you’re targeting SOTA performance, implementing new architecture changes, or don’t already have a
proven recipe, ablations become a substantial cost center rather than minor experiments.
TL;DR: Be paranoid.

extracting the correct information. Since evals will guide every single decision, getting this step right is crucial for the
success of the project!
Change one thing at a time. Keep everything else identical between experiments. Some changes can interact with each
other in unexpected ways, so you want to assess the individual contribution of each change first, then try combining them to
see their overall impact.
Train on enough tokens and use sufficient evaluations. As we mentioned earlier, you need to make sure you have good
coverage in your evaluation suite and train long enough to get reliable signal. Cutting corners here will lead to noisy results
and bad decisions.
Following these rules might feel overly cautious, but the alternative is spending days debugging mysterious performance
drops that turn out to have been caused by an unrelated dependency update from days earlier. The golden principle: Once
you have a good setup, no change should go untested!
