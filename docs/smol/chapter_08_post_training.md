# Chapter 8: Beyond Base Models—Post-Training in 2025

"
Beyond Base Models—Post-Training in 2025
Pretraining gave us SmolLM3’s raw ability, but before the GPUs have cooled down we enter the next frontier of model
capabilities: post-training . This includes supervised fine-tuning, reinforcement learning, model merging, and more—all
designed to bridge the gap between “a model that predicts text” and “a model people can actually use.” If pretraining is
about brute-forcing knowledge into weights, post-training is about sculpting that raw capability into something reliable and
steerable. And once again, the polished post-training papers don’t capture the late-night surprises: GPU meltdowns, finicky
data mixtures, or the way a seemingly minor chat template decision can ripple through downstream benchmarks. In this
chapter, we’ll show how we navigated the messy world of post-training to turn SmolLM3 from a strong base model into a
state-of-the-art hybrid reasoner.
📍What is a hybrid reasoning model?
A hybrid reasoning model operates in two distinct modes: one for concise, direct responses and another for extended step-
by-step reasoning. Typically, the operating mode is set by the user in the system message. Following Qwen3, we make this
explicit with lightweight commands: “/think” invokes extended reasoning, while “/no_think” enforces concise answers. This
way, the user controls whether the model prioritizes depth or speed.
Post-Training Compass: Why → What → How
Once the pre-training finishes, we should have an SFT
baseline within a day.
—Lewis T unstall, optimistic LLM expert
Choose your own (post-training) adventure.

Just like pretraining, post-training benefits from a clear compass to avoid wasted research and engineering cycles. Here’s
how to frame it:
1. Why post-train? The three motivations for training that we outlined in the pretraining compass apply equally to post-
training. For example, you might be exploring whether reinforcement learning can unlock new reasoning capabilities in an
existing model (research), or you might need to distill a large model into a smaller one for latency reasons (production),
or you may have identified a gap where no strong open model exists for a specific use case (strategic open source). The
distinction is that post-training builds on existing capabilities rather than creating them from scratch. But before reaching
for your GPUs, ask yourself:
Do you really need to post-train? Many open-weight models now rival proprietary ones on a wide range of tasks.
Some can even be run locally with quantization and modest compute. If you want a generalist assistant, an off-the-
shelf model from the Hugging Face Hub may already meet your needs.
Do you have access to high-quality domain-specific data? Post-training makes most sense when you are targeting a
specific task or domain where a generalist model underperforms. With the right data, you can tune the model to
produce more accurate outputs for the applications you care most about.
Can you measure success? Without clear evaluation criteria, you won’t know if post-training really helps.
2. What should post-training achieve?  This depends on your priorities. Do you want:
A crisp instruction follower that rarely drifts off topic?
A versatile assistant that can switch tones and roles on demand?
A reasoning engine that can tackle math, code, or agentic problems?
A model that can converse in multiple languages?
3. How will you get there?  This is where recipes matter. We’ll cover:
Supervised fine-tuning (SFT)  to instill core capabilities
Preference optimization (PO) to learn directly from human or AI preferences
Reinforcement learning (RL)  to refine reliability and reasoning beyond supervised data
Data curation  to strike the right balance between diversity and quality
Evaluation  to track progress and catch regressions early
This compass keeps the chaos of post-training grounded. The  why  gives direction, the  what  sets priorities, and the  how
turns ambitions into a practical training loop.
Let’s walk through how we answered these questions for SmolLM3:
Why? For us, the “why” was straightforward as we had a base model that needed post-training before release. At the
same time, hybrid reasoning models like Qwen3 were becoming increasingly popular, yet open recipes showing how to
train them were scarce. SmolLM3 gave us an opportunity to address both goals: Prepare a model for real-world use and
contribute a fully open recipe to sit on the Pareto front alongside Qwen3’s 1.7B and 4B models.
What? We set out to train a hybrid reasoning model that was tailored to SmolLM3’s strengths, chiefly that reasoning
quality should hold up across languages other than English. And since real-world use increasingly involves tool calling
and long context workflows, those became core requirements in our post-training recipe.
How? That’s what you’ll find out in the rest of this chapter!
Just like with pretraining, with post-training we start with the fundamentals, evals and baselines, because every big model
starts with a small ablation. But there’s a key difference in how we ablate. In pretraining, “small” usually means smaller

Let’s start with the topic many model trainers avoid until too late into a project: evals.
First Things First: Evals Before Everything Else
The very first step in post-training—just like in pretraining—is to decide on the right set of evals. Since most LLMs today are
used as assistants, we’ve found that aiming for a model that “works well” is a better goal than chasing abstract benchmarks
of “intelligence” like ARC-AGI. So what does a good assistant need to do? At minimum, it should be able to:
Handle ambiguous instructions
Plan step by step
Write code
Call tools when appropriate
At Hugging Face, we use a layered eval suite, echoing the pretraining principles (monotonicity, low noise, above-random
signal, ranking consistency) that we detailed in the ablations section.
📍Keep your evals current
The list of evals to consider is continuously evolving as models improve. The ones discussed here reflect our focus in mid-
2025. See the LLM Evaluation Guidebook for a comprehensive overview of post-training evals.
There are many ways one can evaluate a post-trained model. These include:
models and datasets. In post-training, “small” means smaller datasets and simpler algorithms . We almost never use a
different base model for ablations because behavior is too model-dependent, and runs are short enough to iterate on the
target model directly.
These behaviors draw on a mix of reasoning, long context handling, and skills with math, code, and tool use. Models as
small as or even smaller than 3B parameters can work well as assistants, though performance usually falls off steeply
below 1B.
1. Capability evals. This class of evals targets fundamental skills:
Knowledge:  We currently use GPQA Diamond (Rein et al., 2024) as the main eval for scientific knowledge. This
benchmark consists of graduate-level, multiple-choice questions. For small models, it’s far from saturated and gives
better signal than MMLU and friends, while being much faster to run. Another good test of factuality is SimpleQA (Wei
et al., 2024), although small models tend to struggle significantly on this benchmark due to their limited knowledge.
Math:  To measure mathematical ability, most models today are evaluated on the latest version of AIME (currently
the 2025 version). MATH-500 (Lightman et al., 2023) remains a useful sanity test for small models, but it’s largely
saturated by reasoning models. For a more comprehensive set of math evals, we recommend those from MathArena.
Code:  We use the latest version of LiveCodeBench to track coding competency. Although targeted toward
competitive programming problems, we’ve found that improvements on LiveCodeBench do translate into better
coding models, albeit limited to Python. SWE-bench Verified is a more sophisticated measure of coding skill, but it
tends to be too hard for small models and thus is not one we usually consider.
Multilinguality: Unfortunately, there are not many options when it comes to testing the multilingual capabilities of
models. We currently rely on Global MMLU (Singh et al., 2025) to target the main languages our models should
perform well in, with MGSM (Shi et al., 2022) included as a test of multilingual mathematical ability.
2. Integrated task evals. These evals test capabilities that are similar to those we were looking to ship:

Instructionfollowing:**  IFEval (J. Zhou et al., 2023) is currently the most popular eval to measure instruction following;
it uses automatic scoring against “verifiable instructions.” IFBench (Pyatkin et al., 2025) is an extension from Ai2 that
includes a more diverse set of constraints than IFEval and mitigates some benchmaxxing1 that has occurred in recent
model releases. For multi-turn instruction following, we recommend Multi-IF (He et al., 2024) or MultiChallenge
(Sirdeshmukh et al., 2025).
Alignment: Measuring how well models align to user intent is typically done by human annotators or through public
leaderboards like LMArena. This is because qualities such as free-form generation, style, and overall helpfulness are
difficult to measure quantitatively with automated metrics. However, it’s very expensive to run these evaluations, which
is why the community has resorted to using LLMs as a proxy for human preferences. The most popular benchmarks of
this flavor include AlpacaEval (Dubois et al., 2025), ArenaHard (T. Li et al., 2024), and MixEval (Ni et al., 2024), with the
latter having the strongest correlation with human Elo ratings on LMArena.
Tool calling:  The Berkeley Function Calling Leaderboard (BFCL) provides a comprehensive test of tool calling, albeit one
that is often saturated quite quickly. TAU-Bench (Barres et al., 2025), which provides a test of a model’s ability to use
tools and resolve user problems in simulated customer service settings, has also become a popular benchmark to
report on.
1. Vibe evals and arenas. Similarly, we have found that “vibe testing” intermediate checkpoints (aka interacting with the
model) is essential for uncovering subtle quirks in model behavior that are not captured by eval scores. As we discuss
later, vibe testing uncovered a bug in our data processing code where all system messages were deleted from the
corpus! This is also something that can be done at scale to measure human preference, like on the popular LMArena.
However, crowdsourced human evaluation tends to be brittle (favoring sycophancy and flowery speech over actual
usefulness), so it’s important to see it as a low-signal feedback.
☝Decontaminate your training data
One risk with relying on public benchmarks is that models can easily overfit to them, especially when synthetic data is used
to generate prompts and responses that are similar to the target benchmarks. For this reason, it is essential to
decontaminate your training data against the evals you will use to guide model development. You can do this with n -gram
matching using scripts like those in Open-R1.
For SmolLM3, we wanted a hybrid reasoning model that could reliably follow instructions and reason well in popular
domains like mathematics and code. We also wanted to ensure we preserved the base model’s capabilities of
multilinguality and long context retrieval.
This led us to the following set of evals:
Long context use:  The most commonly used test for long context retrieval is Needle in a Haystack (NIAH) (Kamradt,
2023), where a random fact (“needle”) is placed in somewhere within a long document (“haystack”) and the model
has to retrieve it. However, this benchmark is too superficial to discriminate long context understanding, so the
community has developed more comprehensive evals, like RULER (Hsieh et al., 2024) and HELMET (Yen et al.,
2025). More recently, OpenAI has released the MRCR and GraphWalks benchmarks, which extend the difficulty of
long context evals.
1. Overfitting prevention evals. To test whether our models are overfitting to a specific skill, we include some robustness or
adaptability evals in our set, like GSMPlus (Q. Li et al., 2024), which perturbs problems from GSM8k (Cobbe et al.,
2021) to test whether models can still solve problems of similar difficulty.
2. Internal evals.** Although public benchmarks can provide some useful signal during model development, they are no
substitute for implementing your own internal evals to target specific capabilities, or asking internal experts to interact
with your model. For example, for SmolLM3 we needed a benchmark to evaluate whether the model was capable of
multi-turn reasoning, so we implemented a variant of Multi-IF to measure this.

Benchmark Category Number of PromptsMetric
AIME25 Competitive mathematics30 avg@64
LiveCodeBench v4 (v5 for final release)Competitive programming100 (268) avg@16
GPQA Diamond Graduate-level reasoning198 avg@8
IFEval Instruction following 541 Accuracy
MixEval Hard Alignment 1000 Accuracy
BFCL v3 Tool use 4441 Mixed
Global MMLU (lite for validation)Multilingual Q&A 590,000 (6,400)Accuracy
GSMPlus (mini for validation) Robustness 10,000 (2,400) Accuracy
RULER Long context 6,500 Accuracy
Let’s look at a few example questions to get a concrete sense of what these evaluations actually test.

↗View interactive version
You can browse through the examples in the interactive viewer to see the types of questions in each benchmark. Notice
how the diversity of domains ensures we’re testing different aspects of model capability throughout our ablations.
For the 3B model scale we were working with, we felt these evals, which would run faster than training itself, would give us
actionable signal and confidence that any improvements were real and not just noise from sampling. We also tracked our
pretraining evals (see the ablation section for a full list) to make sure we weren’t regressing too much on the base model
performance.
☝Prioritize your evals
The story above suggests that we got together as a team, converged on the set of evals, and had them ready to go before
we started training. The reality was far messier: We had a tight deadline and rushed ahead with model training before many
of the above evals were implemented (e.g., RULER was not available until a few days before the model release 🙈 ). In
hindsight, this was a mistake; we should have discussed with the pretraining team which core evals should be preserved
across post-training and prioritized implementing them long before the base model was finished training. In other words,
prioritize your evals before all else!


RULES OF ENGAGEMENT
Let’s summarize this section with a few lessons we’ve learned the hard way, through evaluating thousands of models:
Use small subsets to accelerate evals during model development. For example, LiveCodeBench v4 is highly correlated
with v5 but runs in half the time. Alternatively, use methods like those from tinyBenchmarks (Polo et al., 2024), which
seek to find the smallest subset of prompts that reliably match the full evaluation.
For  reasoning models , strip the chain of thought from the  scored  output. This eliminates false positives and also
directly impacts benchmarks like IFEval which penalize responses that violate constraints like “write a poem in under 50
words.”
If an eval uses LLM judges,  pin the judge and version for apples-to-apples comparisons over time. Even better, use an
open-weight model so that the eval is reproducible even if a provider deprecates the judge model.
Be wary of  contamination  in the base models. For example, most models released before AIME 2025 performed
substantially worse on that than on AIME 2024, suggesting some benchmaxxing was at play.
If possible, treat anything used during ablations as  validation , not  test. This means keeping a set of held-out
benchmarks for the final model reports, similar to the Tulu 3 evaluation framework (Lambert et al., 2025).
Always include a small set of  “vibe evals”  on your own data and tasks to catch overfitting to public suites.
For evals with a small number of problems (typically less than ~2k), sample k times and report the avg@ k accuracy.
This is important to mitigate noise, which can lead to incorrect decisions during development.
When implementing a new eval, make sure you can replicate the published results of a few models (within some error).
Failing to do this can lead to wasting a lot of time later if you need to fix the implementation and reevaluate many
checkpoints.
When in doubt, always go back to the evaluation data , and inspect what you are prompting your models with.
With the evals in hand, it’s time to train some models! But before doing that, we need to pick a post-training framework.
Tools of the Trade
Behind every post-training recipe lies a toolbox of frameworks and libraries that enable large-scale experimentation. Each
frameworks brings its own set of supported algorithms, fine-tuning methods, and scalability features. The following table
summarizes the main areas of support, from supervised fine-tuning to preference optimization and reinforcement learning:

Framework SFT PO RL Multi-modal FullFT LoRA Distributed
TRL ✅ ✅✅✅ ✅ ✅ ✅
Axolotl ✅ ✅✅✅ ✅ ✅ ✅
OpenInstruct ✅ ✅✅❌ ✅ ✅ ✅
Unsloth ✅ ✅✅✅ ✅ ✅ ✅
vERL ✅ ❌✅✅ ✅ ✅ ✅
Prime RL ✅ ❌✅❌ ✅ ✅ ✅
PipelineRL ❌ ❌✅❌ ✅ ✅ ✅
ART ❌ ❌✅❌ ❌ ✅ ❌
TorchForge ✅ ❌✅❌ ✅ ❌ ✅
NemoRL ✅ ✅✅❌ ✅ ❌ ✅
OpenRLHF ✅ ✅✅❌ ✅ ✅ ✅
Here, FullFT  refers to  full fine-tuning , where all model parameters are updated during training.  LoRA  stands for  Low-Rank
Adaptation , a parameter-efficient approach that updates only small low-rank matrices while keeping the base model frozen.
Multi-modal refers to whether support for training on modalities beyond text (e.g., images) is supported, and Distributed
indicates whether training models on more than one GPU is possible.
At Hugging Face, we develop and maintain TRL, so it’s our framework of choice and the one we used to post-train SmolLM3.
📍Fork your frameworks
Given the fast-moving pace of the field, we’ve found it quite effective to run our experiments on an internal fork of TRL. This
allows us to add new features very quickly, which are later upstreamed to the main library. If you’re comfortable working with
your framework’s internals, adopting a similar workflow can be a powerful approach for rapid iteration.
WHY BOTHER WITH FRAMEWORKS AT ALL?
There is a class of researchers that love to bemoan the use of training frameworks and argue that you should implement
everything from scratch, all the time. The implicit claim here is that “real” understanding only comes from reimplementing
every RL algorithm, manually coding every distributed training primitive, or hacking together a one-off eval harness.
But this position ignores the reality of modern research and production. Take RL, for example. Algorithms like PPO and
GRPO are notoriously tricky to implement correctly (Huang et al., 2024), and tiny mistakes in normalization or Kullback–
Leibler (KL) penalties can lead to days of wasted compute and effort.
Similarly, although it may be tempting to write a single-file implementation of some algorithm, can that same script scale
from 1B to 100B+ parameters?
Frameworks exist precisely because the basics are already well understood, and endlessly reinventing them is a poor use of
time. That’s not to say there’s no value in low-level tinkering. Implementing PPO from scratch once is an excellent learning
exercise. Writing a toy transformer without a framework teaches you how attention really works. But in most cases, you’re
better off just picking a framework you like and hacking it for your purposes.
With that rant out of the way, let’s take a look at where we often start our training runs.

Why (Almost) Every Post-Training Pipeline Starts with SFT
RL isn’t new, of course. OpenAI and other labs relied heavily on RL from human feedback (RLHF) (Lambert et al., 2022) to
align their early models, but it wasn’t until the release of DeepSeek-R1 (DeepSeek-AI, Guo, et al., 2025) that RL-based
post-training really caught on in the open source ecosystem.
One thing hasn’t changed, though: Almost every effective post-training pipeline still begins with supervised fine-tuning. The
reasons are straightforward:
It’s cheap: SFT requires modest compute compared to RL. You can usually get meaningful gains without needing to burn
a bonfire of silicon, and in fraction of the time required for RL.
It’s stable: Unlike RL, which is notoriously sensitive to reward design and hyperparameters, SFT “just works.”
It’s the right baseline: A good SFT checkpoint usually gives most of the gains you’re after, and it makes later methods
like DPO or RLHF far more effective.
In practice, this means SFT isn’t just the first step because it’s easy; it’s the step that consistently improves performance
and makes the most sense before anything more complex is attempted. This is especially true when you’re working with
base models, which, with a few exceptions, are too unrefined to benefit from advanced post-training methods.
📍What about DeepSeek-R1-Zero?
At the frontier, the usual reasons for starting with SFT don’t always apply. There’s no stronger model to distill from, and
human annotations are too noisy for complex behaviors like long chain-of-thought reasoning. That’s why DeepSeek skipped
SFT and went straight to RL with R1-Zero; to discover reasoning behaviors that couldn’t be taught with standard supervision.
If you’re in that regime, starting with RL can make sense. But if you’re operating there, you probably aren’t reading this
anyway.
So, if SFT is where most pipelines begin, the next question is:  What  should you fine-tune? That starts with choosing the
right base model.
PICKING A BASE MODEL
When choosing a base model for post-training, a few practical dimensions matter most:
Model size: Although smol models have dramatically improved over time, it is still the case today that larger models
generalize better, and often with fewer samples. Pick a model size that is representative of how you plan to use or
deploy the model after training. On the Hugging Face Hub, you can filter models by modality and size to find suitable
candidates.
If you spend any time on X these days, you probably think reinforcement learning is the only game in town. Every day brings
new acronyms, algorithmic tweaks, and heated debates (Chu et al., 2025; Yue et al., 2025) about whether RL can elicit
new capabilities or not.

In our experience, the base models from Qwen, Mistral, and DeepSeek are the most amenable to post-training, with Qwen
being a clear favorite since each model series typically covers a large parameter range (Qwen3 models range in size from
0.6B to 235B!). This feature makes scaling far more straightforward.
Once you’ve chosen a base model that matches your deployment needs, the next step is to establish a simple, fast SFT
baseline to probe its core skills.
Architecture (MoE vs. dense): MoE models activate a subset of parameters per token and offer higher capacity per unit
of compute. They’re great for large-scale serving but trickier to fine-tune, in our experience. By contrast, dense models
are simpler to train and often outperform MoEs at smaller scales.
Post-training track record: Benchmarks are useful, but it’s even better if the base model has already spawned a
collection of strong post-trained models that resonate with the community. This provides a proxy for whether the model
trains well.

TRAINING SIMPLE BASELINES
This led us to create the Everyday Conversations dataset, which turned out to be crucial for instilling basic chat capabilities
in small models.
For SmolLM3, we set out to train a hybrid reasoning model and initially picked a small set of datasets to target reasoning,
instruction following, and steerabilty. The following table shows the statistics of each dataset:2
As we learned throughout the development of SmolLM3, training hybrid reasoning models is trickier than standard SFT
because you can’t just mix datasets together; you need to  pair  data across modes. Each example has to clearly indicate
whether the model should engage in extended reasoning or give a concise answer, and ideally you want parallel examples
that teach it when to switch modes. Another thing to note from this table is that you should balance your data mixture in
terms of tokens , not examples : for instance, the s1k-1.1 dataset is ~1% of the total examples but accounts for ~11% of
the total tokens due to the long reasoning responses.
This gave us basic coverage across the skills we cared about most, but also introduced a new challenge: Each dataset had
to be formatted differently, depending on whether it should enable extended thinking or not. To unify these formats, we
needed a consistent chat template.
PICKING A GOOD CHAT TEMPLATE
When it comes to choosing or designing a chat template, there isn’t a one-size-fits-all solution. In practice, we’ve found
there are a few questions worth asking up front:
Can users customize the system role?  If users should be able to define their own system prompts (e.g., “act like a
pirate”), the template needs to handle that cleanly.
Does the model need tools? If your model needs to call APIs, the template needs to accommodate structured outputs
for tool calls and responses.
Is it a reasoning model? Reasoning models use templates like <think> ... </think> to separate the model’s
“thoughts” from its final answer. Some models discard the reasoning tokens across turns in a conversation, and the
For SFT, a good baseline should be fast to train, focused on the model’s core skills, and simple to extend with more data
when a particular capability isn’t up to scratch. Choosing which datasets to use for an initial baseline involves some taste
and familiarity with those that are likely to be of high quality. In general, avoid over-indexing on public datasets that report
high scores on academic benchmarks and instead focus on those that have been used to train great models, like
OpenHermes. For example, in the development of SmolLM, we initially ran SFT on WebInstruct, which is a great dataset on
paper. However, during our vibe tests, we discovered it was too science-focused—the model would respond with equations
to simple greetings like “How are you?”
Dataset Reasoning Mode# of Examples% of Examples# of Tokens (M)% of TokensAvg. #
Everyday Conversations/no_think2,260 2.3 0.6 0.8 260.2
SystemChats 30k /no_think33,997 35.2 21.5 28.2 631.9
Tulu 3 SFT Personas IF/no_think29,970 31.0 13.3 17.5 444.5
Everyday Conversations (Qwen3-32B)/think 2,057 2.1 3.1 4.1 1,522
SystemChats 30k (Qwen3-32B)/think 27,436 28.4 29.4 38.6 1070
s1k-1.1 /think 835 0.9 8.2 10.8 8,859
Total - 96,555 100.0 76.1 100.0 2,131
Data mixture for hybrid reasoning baselines

chat template needs to handle that logic.
Will it work with inference engines? Inference engines like vLLM and SGLang have dedicated parsers for reasoning and
tools.3 Compatibility with these parsers saves a lot of pain later, especially in complex agent benchmarks where
consistent tool calls are essential.4
The following table shows a few popular chat templates and how they compare across the key considerations:
In most cases, we’ve found that ChatML or Qwen’s chat templates are an excellent place to start. For SmolLM3, we needed
a template for hybrid reasoning and found that Qwen3 was one of the few that struck a good balance across the
dimensions we cared about. However, it had one quirk that we weren’t entirely happy with: The reasoning content is
discarded for all but the final turn in a conversation. As shown in the following figure, this is similar to how OpenAI’s
reasoning models work:
Although this makes sense for inference (to avoid blowing up the context), we concluded that for training it is important to
retain the reasoning tokens across all turns in order to condition the model appropriately.
So, we decided to craft our own chat template with the following features:
A structured system prompt, like Llama 3’s and those jailbroken from proprietary models. We also wanted to offer the
flexibility to override the system prompt entirely.
Support for code agents, which execute arbitrary Python code instead of making JSON tool calls.
Explicit control of the reasoning mode via the system message.
To iterate on the design of the chat template, we used the Chat Template Playground. This handy application, developed by
our colleagues at Hugging Face, makes it easy to preview how messages are rendered and debug formatting issues. Here’s
what it looks like:
Chat Tem plate System  Role Custom ization Tools Reasoning Inference Com patibility N
ChatML ✅ ✅❌ ✅ Simple and good for most use
Qwen3 ✅ ✅✅ ✅ Hybrid reasoning template
DeepSeek-R1 ❌ ❌✅ ✅ Prefills reasoning content wit
Llama 3 ✅ ✅❌ ✅ Has built-in tools like a Pytho
Gemma 3 ✅ ❌❌ ❌ System role customization de
Command A Reasoning✅ ✅✅ ❌ Multiple chat templates per m
gpt-oss ✅ ✅✅ ✅ Based on the Harmony respo
Turn 3
Turn 2Turn 1
CONTEXT WINDOW ✂
INPUT
OUTPUT
INPUT
OUTPUT
INPUTOUTPUTREASONINGOUTPUT TRUNCATED OUTPUT

You can try it out yourself. Select different examples from the drop-down to see how the chat template works for multi-turn
dialogues, reasoning, or tool use. You can even change the JSON input manually to enable different behavior. For example,
see what happens if you provide enable_thinking: false or append /no_think to the system message.
Once you’ve settled on some initial datasets and a chat template, it’s time to train some baselines!
BABY BASELINES
Before we dive into optimization and squeezing out every point of performance, we need to establish some “baby
baselines.” These baselines aren’t about reaching state of the art (yet), but aim to validate that the chat template does
what we want and that the initial set of hyperparameters produce stable training. Only after we have this foundation do we
start heavily tuning hyperparameters and training mixtures.
When it comes to training SFT baselines, here are the main things to consider:
Will you use full fine-tuning or parameter-efficient methods like LoRA or QLoRA (Quantized LoRA)? As described in the
wonderful blog post by Thinking Machines, LoRA can match FullFT under certain conditions (usually determined by the
size of the dataset).
What type of parallelism do you need? For small models or those trained with LoRA, you can usually get by with data
parallelism. For larger models, you will need FSDP2 or DeepSpeed ZeRO-3 to shard the model weights and optimizer
states. For models trained with long context, use methods like context parallelism.
Use kernels like FlashAttention and Liger if your hardware supports them. Many of these kernels are hosted on the
Hugging Face Hub and can be set via a simple argument in TRL to dramatically lower the VRAM usage.
Mask the loss to train only on assistant tokens. As we discuss later, this can be achieved by wrapping the assistant
turns of your chat template with a special {% generation %} keyword.
↗View interactive version


Tune the learning rate. Aside from the data, this is the most important factor that determines whether your model is
“meh” or “great.”
Pack the training samples and tune the sequence length to match the distribution of your data. This will dramatically
speed up training. TRL has a handy application to do this for you.
Let’s look at how some of these choices panned out for SmolLM3. For our first baseline experiments, we wanted a simple
sanity check: Does the chat template actually elicit hybrid reasoning? To test this, we compared three data mixtures from
our table:
Instruct: Train on the non-reasoning examples.
Thinking: Train on the reasoning examples.
Hybrid: Train on all examples.
Since these are small datasets, we did not use packing, capping sequences at 8,192 tokens for the Instruct subset and
32,768 tokens for the rest. On one node of eight H100s, these experiments were quick to run, taking between 30–90
minutes depending on the subset. The following figures compare the performance of each subset for the corresponding
reasoning mode.
These results quickly showed us that hybrid models exhibit a type of “split brain,” where the data mixture for one reasoning
mode has little effect on the other. This is evident by most evals having similar scores across the Instruct, Thinking, and
Hybrid subsets, with LiveCodeBench v4 and IFEval being the exceptions where hybrid data boosts the overall performance.
For each mixture, we ran SFT on SmolLM3-3B-Base using FullFT with a learning rate of 1e-5 and an effective batch size of
128, and trained for 1 epoch.


VIBE-TEST YOUR BASELINES
Although the evals looked OK, when we tried getting the hybrid model to act in different personas (e.g., like a pirate), it
consistently ignored anything we placed in the system message. After a bit of digging, we found the reason was due to the
way we had formatted the data:
In the design of our chat template, we’d exposed a custom_instructions argument to store the system prompts. For
example, here’s how we set a persona in a dialogue:
from transformers import AutoTokenizer
1
 
2
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
3
 
4
messages = [
5
    {
6
        "content": "I'm trying to set up my iPhone, can you help?",
7
        "role": "user",
8
    },
9
    {
10
        "content": "Of course, even as a vampire, technology can be a bit of a challenge sometimes 
[TRUNCATED]",
11
        "role": "assistant",
12
    },
13
]
14
chat_template_kwargs = {
15
    "custom_instructions": "You are a vampire technologist",
16
    "enable_thinking": False,
17
}
18
rendered_input = tok.apply_chat_template(
19
    messages, tokenize=False, **chat_template_kwargs
20
)
21
print(rendered_input)
22
## <|im_start|>system
23
### Metadata
24
 
25
## Knowledge Cutoff Date: June 2025
26
## Today Date: 28 October 2025
27
## Reasoning Mode: /no_think
28
 
29
### Custom Instructions
30
 
31
## You are a vampire technologist
32
 
33
## <|im_start|>user
34
## I'm trying to set up my iPhone, can you help?<|im_end|>
35
## <|im_start|>assistant
36
## <think>
37
 
38
## </think>
39
## Of course, even as a vampire, technology can be a bit of a challenge sometimes # [TRUNCATED]
<|im_end|>
40


The issue was that our data samples looked like this:
A bug in our processing code had set custom_instructions to None , which effectively removed the system message
from every single training sample 🙈 ! So instead of getting a nice persona for these training samples, we ended up with the
SmolLM3 default system prompt:
This was especially problematic for the SystemChats subset, where all the personas are defined via
custom_instructions and thus the model had a tendency to randomly switch character mid-conversation. This brings us
to the following rule:
☝Rule
Always vibe-test your models, even if the evals look fine. More often than not, you will uncover subtle bugs in your training
data.
{
1
    "messages": [
2
        {
3
            "content": "I'm trying to set up my iPhone, can you help?",
4
            "role": "user",
5
        },
6
        {
7
            "content": "Of course, even as a vampire, technology can be a bit of a challenge 
sometimes [TRUNCATED]",
8
            "role": "assistant",
9
        },
10
    ],
11
    "chat_template_kwargs": {
12
        "custom_instructions": None,
13
        "enable_thinking": False,
14
        "python_tools": None,
15
        "xml_tools": None,
16
    },
17
}
18
chat_template_kwargs = {"custom_instructions": None, "enable_thinking": False}
1
rendered_input = tok.apply_chat_template(messages, tokenize=False, **chat_template_kwargs)
2
print(rendered_input)
3
## <|im_start|>system
4
#### Metadata
5
 
6
## Knowledge Cutoff Date: June 2025
7
## Today Date: 28 October 2025
8
## Reasoning Mode: /no_think
9
 
10
#### Custom Instructions
11
 
12
## You are a helpful AI assistant named SmolLM, trained by Hugging Face.
13
 
14
## <|im_start|>user
15
## I'm trying to set up my iPhone, can you help?<|im_end|>
16
## <|im_start|>assistant
17
## <think>
18
 
19
## </think>
20
## Of course, even as a vampire, technology can be a bit of a challenge sometimes [TRUNCATED]
<|im_end|>
21

Fixing this bug had no impact on the evals, but finally we were confident the chat template and dataset formatting were
working.
Once your setup is stable and your data pipeline checks out, the next step is to focus on developing specific capabilities.
TARGETING SPECIFIC CAPABILITIES
During the development of Open-R1, we noticed that training a base model entirely on single-turn reasoning data would fail
to generalize to multi-turn reasoning. This is not a surprise; absent such examples, the model is being tested outside its
training distribution.
To measure this quantitatively for SmolLM3, we took inspiration from the Qwen3 team, who developed an internal eval
called ThinkFollow that randomly inserts /think or /no_think tags to test whether the model can consistently switch
between reasoning modes. In our implementation, we took the prompts from Multi-IF and then checked if the model
generated empty or non-empty think blocks enclosed in the <think> and </think> tags. As expected, the results from
our hybrid baseline showed the model failing abysmally to enable the reasoning mode beyond the first turn:
The method is illustrated below:
To fix this capability, we constructed a new dataset called IFThink. Based on the Multi-IF pipeline, we used single-turn
instructions from Tulu 3’s instruction-following subset and expanded them into multi-turn exchanges using Qwen3-32B to
generate both verifiable instructions and reasoning traces.


Including this data in our baseline mix produced a dramatic improvement:
Multi-turn prompts
Tulu 3 IF dataset
Set of instruction types
Single-turn prompt Generate instructions with 
Qwen3-32B
Prompt @ turn 1 Prompt @ turn 2 Prompt @ turn 3
Generate reasoning traces 
with Qwen3-32B
IFThink

After fixing the multi-turn reasoning issue with IFThink, our baseline finally behaved as intended; it stayed consistent across
turns, followed instructions, and used the chat template correctly. With that foundation in place, we turned back to the
basics: tuning the training setup itself.
WHICH HYPERPARAMETERS ACTUALLY MATTER?
In SFT, there are only a few hyperparameters that actually matter. Learning rate, batch size, and packing determine almost
everything about how efficiently your model trains and how well it generalizes. In our baby baselines, we picked reasonable
defaults just to validate the data and chat template. Now that the setup was stable, we revisited these choices to see how
much impact they had on our baseline.
Masking User Turns
One subtle design choice for the chat template is whether to mask the user turns during training. In most chat-style
datasets, each training example consists of alternating user and assistant messages (possibly with interleaved tool calls). If
we train the model to predict all tokens, it effectively learns to autocomplete user queries, rather than focusing on producing
high-quality assistant responses.
As shown in the following figure, masking user turns prevents this by ensuring the model’s loss is computed only on
assistant outputs, not user messages:


In TRL, masking is applied for chat templates that can return the assistant tokens mask. In practice, this involves including
a {% generation %} keyword in the template, as follows:
Then, when apply_chat_template() is used with return_assistant_tokens_mask=True , the chat template will
indicate which parts of the dialogue should be masked. Here’s a simple example, which shows how the assistant tokens
are given ID 1 while the user tokens are masked with ID 0:
{%- for message in messages -%}
1
  {%- if message.role == "user" -%}
2
    {{ "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n" }}
3
  {%- elif message.role == "assistant" -%}
4
{% generation %}
5
{{ "<|im_start|>assistant" + "\n" + message.content + "<|im_end|>\n" }}
6
{% endgeneration %}
7
  {%- endif %}
8
{%- endfor %}
9
{%- if add_generation_prompt %}
10
  {{ "<|im_start|>assistant\n" }}
11
{%- endif %}
12


In practice, masking doesn’t have a huge impact on downstream evals in most cases, providing just a few points of
improvement. With SmolLM3, we found it had the most impact on IFEval, likely because the model is less inclined to
restate the prompt and follow the various constraints more closely. The following figures show how user masking affected
each eval and reasoning mode.
To Pack or Not to Pack?
chat_template = '''
1
{%- for message in messages -%}
2
  {%- if message.role == "user" -%}
3
    {{ "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n" }}
4
  {%- elif message.role == "assistant" %}
5
    {% generation %}
6
    {{ "<|im_start|>assistant" + "\n" + message.content + "<|im_end|>\n" }}
7
    {% endgeneration %}
8
  {%- endif %}
9
{%- endfor %}
10
{%- if add_generation_prompt %}
11
  {{ "<|im_start|>assistant\n" }}
12
{%- endif %}
13
'''
14
rendered_input = tok.apply_chat_template(messages, chat_template=chat_template, 
return_assistant_tokens_mask=True, return_dict=True)
15
print(rendered_input)
16
## {'input_ids': [128011, 882, 198, 40, 2846, 4560, 311, 743, 709, 856, 12443, 11, 649, 499, 1520, 
30, 128012, 198, 257, 128011, 78191, 198, 2173, 3388, 11, 1524, 439, 264, 51587, 11, 5557, 649, 
387, 264, 2766, 315, 264, 8815, 7170, 510, 2434, 12921, 9182, 60, 128012, 271], 'attention_mask': 
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'assistant_masks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1]}
17


Sequence packing is one of the training details that make a huge difference to training efficiency. In SFT, most datasets
contain samples of variable length, which means each batch contains a large number of padding tokens that waste
compute and slow convergence.
Packing solves this by concatenating multiple sequences together until a desired maximum token length is achieved. There
are various ways to perform the concatenation, with TRL adopting a “best-fit decreasing” strategy (Ding et al., 2024), where
the ordering of sequences to pack is determined by their length. As shown here, this strategy minimizes truncation of
documents across batch boundaries while also reducing the amount of padding tokens:
Packing in post-training vs. pretraining
In pretraining, this isn’t really a question. When training on trillions of tokens, packing is essential to avoid wasting
significant amounts of compute on padding. Pretraining frameworks like Megatron-LM and Nanotron implement packing by
default. Post-training is different. Because the runs are shorter, the trade-offs change.
To get a sense of how efficient packing is for training, let’s compare the runtimes between packing and no-packing over one
epoch of our baseline dataset:


Depending on the batch size, we see that packing improves throughput by a factor of 3–5×! So, should you always use
packing? To some extent, the answer depends on how large your dataset is, because packing reduces the number of
optimization steps per epoch by fitting more tokens into each step. You can see this in the following figure, where we plot
the average number of non-padding tokens per batch:


With packing, the number of tokens per batch scales linearly with the batch size, and compared to training without packing,
it can include up to 33× more tokens per optimization step! However, packing can slightly alter the training dynamics: While
you process more data overall, you make fewer gradient updates, which can influence final performance. This is especially
true on small datasets, where each sample matters more. For example, if we compare packing versus no packing at the
same effective batch size of 128, we see that some evals, like IFEval, take a significant performance hit:
More generally, we see that once the effective batch size is large than 32, there is an average drop in performance for this
particular model and dataset:


In practice, for large-scale SFT where the dataset is massive, packing is almost always beneficial since the compute
savings far outweigh any minor differences in gradient frequency. However, for smaller or more diverse datasets—like
domain-specific fine-tuning or instruction tuning on limited human-curated data—it might be worth disabling packing to
preserve sample granularity and ensure every example contributes cleanly to optimization.
Ultimately, the best strategy is empirical: Start with packing enabled, monitor both throughput and downstream evals, and
adjust based on whether the speed gains translate into equivalent or improved model quality.
Tuning the Learning Rate
We now come to the last important hyperparameter: the learning rate. Set it too high and training may diverge; too low and
convergence is painfully slow.
In SFT, the optimal learning rate is typically an order of magnitude (or more) smaller than the one used during pretraining.
This is because we’re initializing from a model with rich representations, and aggressive updates can lead to catastrophic
forgetting.
Tuning the learning rate in post-training vs. pretraining
Unlike in pretraining, where hyperparameter sweeps on the full run are prohibitively expensive, post-training runs are short
enough that we can actually do full learning rate sweeps.
In our experiments, we’ve found that the “best” learning rate varies with model family, size, and the use of packing. Since a
high learning rate can lead to exploding gradients, we find it’s often safer to slightly decrease the learning rate when packing
is enabled. Our results show that using a small LR of 3e-6 or 1e-5 gives better overall performance than large values:


Although a few points on average may not seem like much, if you look at individual benchmarks like AIME25, you’ll see the
performance drop dramatically when the learning rate is larger than 1e-5.
Scaling the Number of Epochs
In our ablations, we usually train for a single epoch to iterate quickly. Once you’ve identified a good data mixture and tuned
key parameters like the learning rate, the next step is to increase the number of epochs for final training.
For example, if we take our baseline data mixture and train for five epochs, we see it is possible to squeeze out a few more
percentage points of performance on average:


As we saw with the learning rate scan, the average performance obscures the impact that scaling the number of epochs has
on individual evals: In the case of LiveCodeBench v4 with extended thinking, we nearly double the performance over one
epoch!
Once you’ve iterated on your SFT data mixture and the model has reached a reasonable level of performance, the next step
is often to explore more advanced methods, like [preference optimization](#from-sft-to-preference-optimization-teaching-
models-what- better- means) or reinforcement learning. However, before diving into those, it’s worth considering whether the
additional compute would be better spent on strengthening the base model through continued pretraining.
📍Optimizers in post-training
Another important component we mentioned in the pretraining section is the optimizer. AdamW remains the default choice
for post-training as well. An open question is whether models pretrained with alternative optimizers, like Muon, should be
post-trained with the same optimizer. The Kimi team found that using the same optimizer for pre- and post-training yielded
the best performance for their Moonlight model.
BOOSTING REASONING THROUGH CONTINUED PRETRAINING
Continued pretraining—or mid-training , if you want to sound fancy—means taking a base model and training it further on
large amounts of domain-specific tokens before doing SFT. Mid-training is useful when your target capabilities for SFT share
a common core skill, such as coding or reasoning. In practice, this shifts the model toward a distribution that better
supports reasoning, a specific language, or any other capability you care about. Starting SFT from a model that has already
integrated that core skill allows your model to better focus on the specific topics in your SFT data rather than using compute
to learn the core skill from scratch.
The mid-training approach traces back to ULMFit (Howard & Ruder, 2018), which pioneered the three-stage pipeline of
general pretraining → mid-training → post-training that is now common in modern LLMs like FAIR’s Code World Model (team


et al., 2025):
This approach was also used in the training of Phi-4-Mini-Reasoning (Xu et al., 2025), but with a twist: Instead of doing
continued pretraining on web data, the authors used distilled reasoning tokens from DeepSeek-R1 for the mid-training
corpus. The results were compelling, showing consistent and large gains through multi-stage training:
Model AIME24 MATH-500 GPQA Diamond
Phi-4-Mini 10.0 71.8 36.9
+ Distill mid-training 30.0 82.9 42.6
+ Distill fine-tuning 43.3 89.3 48.3
+ Roll out DPO 50.0 93.6 49.0
+ RL (Phi-4-Mini-Reasoning) 57.5 94.6 52.0
These results prompted us to try a similar approach. From our prior experience with creating and evaluating reasoning
datasets in Open-R1, we had three main candidates to work with:
Mixture of Thoughts : 350k reasoning samples distilled from DeepSeek-R1 across math, code, and science.
Llama-Nemotron-Post-Training-Dataset: NVIDIA’s large-scale dataset distilled from a wide variety of models, such as
Llama 3 and DeepSeek-R1. We filtered the dataset for the DeepSeek-R1 outputs, which resulted in about 3.64M
samples, or 18.7B tokens.
OpenThoughts3-1.2M: One of the highest-quality reasoning datasets, with 1.2M samples distilled from QwQ-32B,
comprising 16.5B tokens.
Since we planned to include reasoning data in the final SFT mix, we decided to keep Mixture of Thoughts for that stage and
use the others for mid-training. We used ChatML as the chat template to avoid “burning in” the SmolLM3 one too early on.
We trained for five epochs with a learning rate of 2e-5, using eight nodes to accelerate training with an effective batch size
of 128.
📍When to mid-train?
You might wonder why we’re discussing mid-training after we did some SFT runs. Chronologically, mid-training happens
before SFT on the base model, but whether it will be beneficial often only becomes clear after you’ve run initial SFT
experiments and identified performance gaps. In practice, you’ll often iterate: Run SFT to identify weak areas, then do
targeted mid-training, then run SFT again. Think of this section as “what to do when SFT alone isn’t enough.”
The Mystery of the Melting GPUs
Running these experiments turned out to be a surprising challenge on our cluster: The aging GPUs would get throttled at
various points, which would lead to hardware failures and forced restarts of each run. To give you a taste of what it was like,


here are the logs from one of the runs, where each color change represents a restart:
We initially thought DeepSpeed might be the culprit, since the accelerator is highly optimized for throughput. To test this, we
switched to data parallelism, which helped somewhat, but then the loss was dramatically different!


So, we switched back to DeepSpeed and added aggressive checkpointing to minimize the time lost from GPUs overheating
and “falling off the bus.” This strategy proved successful and is something we recommend more generally:
☝Rule
As we emphasized in the pretraining section, save model checkpoints frequently during a training run and push them to
remote storage (e.g., the Hugging Face Hub) to avoid accidental overwrites. Also, make your training framework robust to
failures and capable of automatic restarts. Both of these strategies will save you time, especially for long-running jobs like
mid-training.
After babysitting the runs for a week or so, we finally had our results:
Overall, we found that NVIDIA’s post-training dataset gave better performance than OpenThoughts, but the combination was
best overall.
Now let’s take a look at the effect of taking one of these checkpoints and applying our same baseline data mixture:
As we later discovered, a bug with data parallelism in Hugging Face Accelerate meant that the weights and gradients were
stored in the model’s native precision (BF16 in this case), which led to numerical instability and loss of gradient accuracy
during accumulation and optimization.


The effect of using a mid-trained reasoning model instead of the pretrained one was dramatic: With extended thinking, we
nearly tripled the performance on AIME25 and LiveCodeBench v4, while GPQA-D received a full 10-point gain. Somewhat
surprisingly, the reasoning core also translated partially to the /no_think reasoning mode, with about 4- to 6-point
improvements on the reasoning benchmarks. These results gave us clear evidence that for reasoning models, it almost
always makes sense to perform some amount of mid-training if your base model hasn’t already seen a lot of reasoning data
during pretraining.
📍When not to mid-train
Mid-training shines when your model must learn a new core skill. It’s less useful when the base model already has the skill
or if you’re trying to elicit shallow capabilities such as style or conversational chit-chat. In these cases, we recommend
skipping mid-training and allocating your compute to other methods, like preference optimization or reinforcement learning.
Once you’re confident in your SFT data mixture and the model’s broad capabilities, the focus naturally shifts from learning
skills to refining them. In most cases, the most effective way forward is preference optimization.
From SFT to Preference Optimization: Teaching Models What  Better  Means
Although you can keep scaling SFT with more data, at some point you’ll observe diminishing gains or failure modes like your
model being unable to fix its own buggy code. Why? Because SFT is a form of imitation learning , so the model only learns
to reproduce patterns in the data it was trained on. If the data doesn’t already contain good fixes, or if the desired behavior
is hard to elicit with distillation, the model has no clear signal for what counts as “better.”


This is where preference optimization comes in. Instead of just copying demonstrations, we give the model comparative
feedback like “response A is better than response B.” These preferences provide a more direct training signal for quality and
enable model performance to scale beyond the limits of SFT alone.
Another benefit of preference optimization is that it typically requires far less data than SFT, since the starting point is
already a pretty good model that can follow instructions and has knowledge from previous training stages.
Let’s take a look at how these datasets are created.
CREATING PREFERENCE DATASETS
Historically, preference datasets were created by providing human annotators with pairs of model responses and asking
them to grade which one is better (possibly on a scale). This approach is still used by LLM providers to collect  human
preference  labels, but it is extremely expensive and scales poorly. Recently, LLMs have become capable of producing high-
quality responses, and often in a cost-effective way. These advances make it practical for LLMs to generate preferences for
many applications. In practice, there are two common approaches.
Strong vs. Weak
1. Take a fixed set of prompts  (often curated for coverage and difficulty).
2. Generate one response from a weaker or baseline model, and another from a high-performing model.
3. Label the stronger model’s output as the chosen response  and the weaker one as rejected  .
This produces a dataset of “stronger vs. weaker” comparisons  , which is simple to construct because we
assume the stronger model’s output is reliably better.
Here’s a popular example from Intel, who took an SFT dataset with responses from GPT-3.5 and GPT-4 and converted it into
a preference dataset by selecting the GPT-4 responses as chosen and the GPT-3.5 ones as rejected:
x
y 
c y 
r
({x,y ,y })c r

↗View interactive version
On-Policy with Grading
1. Use the  same model  you will train to generate multiple candidate responses to the same prompt. This creates data
that is “on-policy” because it reflects the distribution of outputs the model would naturally produce.
2. Instead of relying on a stronger model as the reference, introduce an  external grader : either a verifier or a reward model
that scores responses along one or more quality axes (e.g., helpfulness, factual accuracy).
3. The grader then assigns preference labels among the candidate responses, producing a more nuanced and flexible
preference dataset.
This method allows ongoing bootstrapping of preference data as the model improves, but its quality depends heavily on the
evaluator’s reliability and calibration.
SnorkelAI provide a nice example of such a dataset. They took the prompts from a popular preference dataset called
UltraFeedback, partitioned them into three sets, and then applied the above recipe iteratively to improve their model:


↗View interactive version
At the time of SmolLM3’s development, there did not exist any preference data with reasoning traces, so we decided to
generate some of our own using the “strong vs. weak” approach. We used the prompts from Ai2’s Tulu 3 preference
mixture to generate responses from Qwen3-0.6B and Qwen3-32B in the  /think  mode. The result was a large-scale
dataset of 250k+ LLM-generated preferences, ready to simultaneously improve our SFT checkpoint across multiple axes
using preference optimization algorithms.
WHICH ALGORITHM SHOULD YOU USE?
Its appeal came from being simple to implement, stable in practice, and effective even with modest amounts of preference
data. As a result, DPO has become the default method to improve SFT models before reaching for more complex techniques
like RL.
Direct preference optimization (DPO) (Rafailov et al., 2024) was the first preference optimization algorithm to gain
widespread adoption in open source.


But researchers quickly discovered there are many ways to improve upon DPO, and nowadays there are a wide variety of
alternatives to explore. Here are a few of the ones we’ve found most effective:
Kahneman–Tversky optimization (KTO) (Ethayarajh et al., 2024): Instead of relying on preference pairs, KTO models
whether an individual response is “desirable” or not, using ideas from human decision making. This is a good choice if
you don’t have access to paired preference data (e.g., raw responses like 👍  or 👎  collected from end users).
Odds ratio preference optimization (ORPO) (Hong et al., 2024): This integrates preference optimization directly into SFT
by adding an odds ratio to the cross-entropy loss. As a result, there is no need for a reference model or SFT stage,
which makes this method more computationally efficient.
Anchored preference optimization (APO) (D’Oosterlinck et al., 2024): This is a more controllable objective that explicitly
regularizes how much the model’s likelihoods for chosen versus rejected outputs should shift, rather than just optimizing
their difference. There are two variants (APO-zero and APO-down). Which to use depends on the relationship between
your model and the preference data; i.e., whether the chosen outputs are better than the model’s or worse.
Luckily, we can switch between many of these with just a one-line change in TRL’s DPOTrainer , so for our initial baseline
we did the following:
1. Use the prompts and completions from Ai2’s Tulu 3 Preference Personas IF dataset to measure the improvements for
instruction following on IFEval with the /no_think reasoning mode.
2. Reuse the prompts from the above dataset, but now generate “strong vs. weak” preference pairs with Qwen3-32B and
Qwen3-0.6B. This gave us preference data for the /think reasoning mode.
3. Train for one epoch and measure the in-domain improvements on IFEval, along with the out-of-domain impact on other
evals, like AIME25, which are directly correlated with instruction following.
As shown in the following figure, the in-domain improvements for both reasoning modes were significant: On IFEval, APO-
zero improved over the SFT checkpoint by 15–20 percentage points!

Since APO-zero also had the best overall out-of-domain performance, we settled on using it for the remainder of our
ablations.
📍Preference optimization works for reasoning
As our results show, preference optimization doesn’t just make models more helpful or aligned; it teaches them to reason
better . If you need a quick way to improve your reasoning model, try generating “strong vs. weak” preferences and ablate
different loss functions. You may find significant gains over vanilla DPO!
WHICH HYPERPARAMETERS MATTER MOST FOR PREFERENCE OPTIMIZATION?
For preference optimization, there are typically only three hyperparameters that impact the training dynamics:
The learning rate, typically a factor of 10–100× smaller than the one used for SFT
The β parameter, which typically controls the size of the margin between preference pairs
The batch size


Let’s take a look at how these played out for SmolLM3, starting from the SFT checkpoint we trained over the whole of
smoltalk2 .
Use Small Learning Rates for Best Performance
The first ablation we ran was to check the influence of the learning rate on model performance. We ran experiments to
determine the influence of learning rates between ~200× smaller (1e-7) and ~2× smaller (1e-5) than the SFT learning rate
(2e-5). Previous projects, like Zephyr 7B, had taught us that the best learning rate for preference optimization methods is
around 10× smaller than the one used for SFT, and the ablations we ran for SmolLM3 confirmed this rule of thumb.
As shown in the following figure, learning rates approximately 10× smaller improved the performance of the SFT model in
both reasoning modes, but all learning rates beyond that 10× limit resulted in worse performance for the extended thinking
mode:
The trend for the /no_think reasoning mode is more stable, with the best learning rate at 5e-6. This was mostly driven by
a single benchmark (LiveCodeBench v4), so we opted for 1e-6 in our SmolLM3 runs.
Our recommendation for your training runs is to run scans of your learning rate at a range of 5× to 20× smaller than your
SFT learning rate. It is highly likely that you will find your optimal performance within that range!
Tune Your β
We ran experiments for the ß parameter at a broad range of values (from 0.01 to 0.99) to explore different degrees of
alignment to the reference model. As a reminder, lower values encourage staying close to the reference model while higher
values allow the model to match the preference data more closely. Performance remained stable across multiple ß values
without extended thinking. The model performance for β=0.1 was the highest for both reasoning modes and showed


improvement compared to the metrics from the SFT checkpoint. Using a lower ß value hurt model performance and resulted
in a worse model than the SFT checkpoint.
These results suggest that values greater than 0.1 are preferable for preference optimization, and that aligning the model
with the preference data is more beneficial than staying close to the reference model. However, we suggest exploring ß
values in the range 0.01–0.5. Higher values may erase capabilities from the SFT checkpoint that we might not be capturing
in the evals shown on the plot.
Scaling the Preference Data
We also ran experiments to determine how dataset size influences results, testing values from 2k to 340k preference pairs.
Across this range, performance remained stable. Performance drops in the extended thinking mode occurred for datasets
beyond 100k preference pairs, but the drop was not as pronounced as we saw with different learning rate values. The
dataset we used for the SmolLM3 training run was 169k preference pairs, but the results showed that smaller datasets
also resulted in improvements over the SFT checkpoint. For future projects, we know we can experiment with smaller
datasets during the iteration phase, when it is important to try multiple ideas and quickly identify the most promising
configurations.


Bringing It All Together
Bringing all these threads together produced the final SmolLM3-3B model: best-in-class for its size and sitting on the Pareto
front with Qwen’s own hybrid reasoning models.


Not too shabby for a few weeks of work!
RULES OF ENGAGEMENT
To summarize our findings about preference optimization that could be useful for your future projects:
Don’t be afraid to create your own preference data! With inference becoming “too cheap to meter,” these days it’s simple
and cost-effective to generate LLM preferences from various inference providers.
Pick DPO as your initial baseline and iterate from there. We’ve found that depending on the type of preference data,
other algorithms, like ORPO, KTO, or APO, can provide significant gains over DPO.
Use a learning rate that’s around 10× smaller than the one used for SFT.
Scan over β, usually in the range 0.01 to 0.5
Since most preference algorithms overfit after one epoch, partition your data and train iteratively for best performance.
Preference optimization is often the sweet spot between simplicity and performance, but it inherits a key limitation: It’s only
as good as the offline preference data you can collect. At some point, static datasets run out of signal and you need
methods that can generate fresh training feedback online as the model interacts with prompts and the environment. That’s
where preference optimization meets the broader family of on-policy and RL-based methods.
Going On-Policy and Beyond Supervised Labels
If you want your model to consistently solve math problems, generate executable code, or plan across multiple steps, you
often need a  reward signal  rather than just “A is better than B.”
Instruct models without reasoning

This is where RL starts to make sense. Instead of supervising the model with preferences, you let it interact with an
environment (which could be a math verifier, a code executor, or even real user feedback) and learn directly from the
outcomes. RL shines when:
You can check correctness automatically , e.g., with unit tests, mathematical proofs, or API calls, or have access to a
high-quality verifier or reward model.
The task requires multi-step reasoning or planning , where local preferences may not capture long-term success.
You want to optimize for objectives beyond preference labels , like passing unit tests for code or maximizing some
objective.
When it comes to LLMs, there are two main flavors of RL:
Reinforcement learning with verifiable rewards (RLVR): This approach, popularized by DeepSeek-R1, involves the use of
verifiers that check whether a model’s output meets some clearly defined correctness criteria (e.g., does the code
compile and pass all tests, or is the mathematical answer correct). The policy is then fine-tuned with RL to produce
more verifiably correct outputs.
Both RLHF and RLVR define what the model is being optimized for, but they don’t tell us how that optimization should be
carried out. In practice, the efficiency and stability of RL-based training depends heavily on whether the learning algorithm is
on-policy or off-policy .
Methods such as GRPO typically fall into the category of on-policy optimization algorithms, where the model (the policy) that
generates the completions is the same as the one being optimized. While it is broadly the case that GRPO is an on-policy
algorithm, there are a few caveats. First, to optimize the generation step, several batches of generations may be sampled
and then  updates are made to the model, with the first batch being on-policy and the next few batches being slightly off-
policy.
As autoregressive generation from LLMs is slow, many frameworks, like verl and PipelineRL, have added asynchronous
generation of completions and “in-flight” updates of model weights to maximize training throughput. These approaches
require more complex and careful implementation but can achieve training speeds 4–5× higher than synchronous training
methods. As we’ll see later, these improvements in training efficiency are especially pronounced for reasoning models,
which have long-tail token distributions.
For SmolLM3, we skipped RL altogether, mostly due to time constraints and having a model that was already best-in-class
with offline preference optimization. However, since the release, we have revisited the topic, and in the next section we’ll
share some of the lessons we’ve learned.
APPLYING RLVR TO HYBRID REASONING MODELS
Hybrid reasoning models pose additional complexity for RLVR because generation lengths vary considerably depending on
the reasoning mode. For example, in the following figure, we plot the token length distributions on AIME25 for the final APO
checkpoint from SmolLM3:
Reinforcement learning from human feedback (RLHF): This approach, popularized by OpenAI’s InstructGPT paper (Ouyang
et al., 2022), was the basis for GPT-3.5 and many modern LLMs. Here, human annotators compare model outputs (e.g.,
“A is better than B”) and a reward model is trained to predict those preferences. The policy is then fine-tuned with RL to
maximize the learned reward.
k
To account for policy lag between the model used for generation and the current model being optimized, importance
sampling and clipping are used to reweight the token probabilities and restrict the size of the updates.

As you can see, the /no_think mode generates solutions with a median length of around 2k tokens, while the median
length for the /think mode is much larger (16k tokens) with a fat-tailed distribution. Ideally, we would like to improve the
overall performance of both modes with RLVR, without changing their respective length distributions too radically.
To explore this, we focused on optimizing the /no_think mode first using a subset of prompts from Big-Math, a dataset of
over 250k math problems with verified answers.
To our surprise, naively applying GRPO led to a form of reward hacking: Despite never being prompted to emit a long chain
of thought (CoT), the model learned to exploit its base reasoning capabilities to increase the reward. As shown in the
following figure, not only did the reward go up, but so too did the length of the completions:


In other words, RLVR with GRPO converted the /no_think mode of our reasoning model into one that looked very similar
to the /think mode!
When we reviewed the model’s completions, we saw that it was now generating long CoTs and even including some of the
cognitive behaviors (Gandhi et al., 2025) like “Wait, …” that are associated with reasoning models. An example of this is
shown here:
Naively applying RLVR to SmolLM3 achieves high reward and improved downstream performance, but comes at the cost of exploding
completion lengths.

Mitigating Reward Hacking with Overlong Penalties
This issue can be mitigated by including an overlong completion penalty that penalizes completions over a certain length.
The penalty is parameterized by two arguments: max completion length,  , and soft punishment cache,  . This
penalty was one of the improvements proposed in the DAPO paper (Yu et al., 2025) and amounts to applying a reward
function as follows:
Using this penalty, we can directly control the model’s output distribution and measure the trade-off between increasing
response length and performance. An example is shown in the following figure, where we vary the overlong penalty from
1.5k to 4k in steps of 512 tokens:
However, since the jogger and the train are moving in the same direction, the relative speed 
between them is the difference of their speeds:
1
\\( v_{\\text{rel}} = v_t - v_j = 12.7778 - 2.7778 = 10 \\) m/s
2
3
 **Wait,**  let me double-check the calculation:
4
5
 **Wait,**  46 km/hr to m/s:
6
46 * 1000 m / 3600 s = (46 * 10)/36 = 460/36 ≈ 12.7778 m/s (correct)
7
10 km/hr is:
8
10 * 1000 / 3600 = 10000 / 3600 ≈ 2.7778 m/s (correct)
9
Difference: 12.7778 - 2.7778 = 10 m/s (exact, since 46-10=36 km/hr → 36*(1000/3600)=10 m/s 
exactly)
10
11
The jogger is 340 meters ahead of the train's engine. When the train passes the jogger, the 
distance the train needs to cover relative to the jogger is 340 meters plus the length of the 
train? Wait, actually, let me think about the problem again.
12
13
 **Wait,**  actually, when the problem says the train takes 46 seconds to pass the jogger, that 
usually means the time from the front of the train to the end passing the jogger. The length of 
the train would then be equal to the relative speed multiplied by the time.
14
15
But first, the initial position: the jogger is 340 meters ahead of the engine. So the engine is 
340 meters ahead of the jogger. Wait no, the wording says "340 m ahead of the engine", so the 
jogger is ahead by 340 meters. So when the jogger is running alongside, the engine of the train is 
340 m away behind the jogger? Wait, maybe I need to clarify.
16
17
 **Wait**  the problem says: "A jogger running at 10 km/hr alongside a railway track is 340 m 
ahead of the engine of a train running at 46 km/hr in the same direction. The train takes 46 
seconds to pass the jogger. How long is the train in meters?"
18
19
Hmm, so the jogger is 340 m ahead of the engine along the track. Since they're moving in the same 
direction, the train is behind the jogger by 340 meters. To pass the jogger, the train has to 
cover that 340 meters plus the length of the train itself?
20
21
L  
max L  
cache
R  (y)=length
   
⎩⎨
⎧0,
 ,L  
cache
(L  −L  −∣y∣)max cache
−1,
∣y∣≤L  −L  
max cache
L  −L <∣y∣≤L  
max cache max
L  <∣y∣max

The trade-off between response length and performance is clearer when we examine the improvements on AIME25:
Applying an overlong penalty constrains the length of each rollout, while also reducing the average reward.

Now we can clearly see how the overlong penalty impacts downstream performance, with penalties in the range 2–4k
producing significant improvements while keeping the token distribution in check. As shown in the next figure, if we take the
checkpoints from step 400, we can compare the output token distributions between the initial policy and the final model
across a range of different penalties:
Downstream performance of SmolLM3 with RLVR on AIME25.

Bringing It All Together
We found that applying a length penalty in the range 2.5–3k gave the best trade-off between performance and response
length, with the following figure showing that GRPO nearly doubles the performance on AIME 2025 over offline methods like
APO:


Now that we know how to improve performance in the /no_think reasoning mode, the next step in the RL training pipeline
would be joint training of the model in both reasoning modes at once. However, we have found this to be quite a tough nut
to crack because each mode requires its own length penalty and the interplay has thus far produced unstable training. This
highlights the main challenge with trying to apply RL on hybrid reasoning models, and we can see it reflected in a new trend
from model developers like Qwen to release the instruct and reasoning variants separately.
Our experiments show that RLVR can steer reasoning behavior effectively, but only with careful reward shaping and stability
mechanisms. Given this complexity, it’s worth asking whether reinforcement learning is the only viable path forward. In fact,
several lighter-weight on-policy optimization strategies have been proposed in recent literature, although they remain
surprisingly underexplored by the open source community. Let’s close out this chapter by taking a look at some of them.
IS RL THE ONLY GAME IN TOWN?
Other approaches to on-policy learning extend preference optimization and distillation into iterative loops that refresh the
training signal as the model evolves. These include:
Online DPO:  Rather than training once on a fixed preference dataset, the model continually samples new responses,
collects fresh preference labels (from reward models or LLM graders), and updates itself. This keeps the
optimization on-policy and reduces drift between training data and the model’s current behavior (Guo et al., 2024).
On-policy distillation:  Instead of preferences, the signal comes from a stronger teacher model. The student samples
responses at every training step and the KL divergence between the student and teacher logits on these samples
provides the learning signal. This allows the student to continuously absorb the teacher’s capabilities, without needing
explicit preference labels or verifiers (Agarwal et al., 2024).
These methods blur the line between static preference optimization and full RL: You still get the benefits of adapting to the
model’s current distribution, but without the full complexity of designing and stabilizing a reinforcement learning loop.


WHICH METHOD SHOULD YOU USE?
Although there are a gazillion research papers about which on-policy method is “best,” in practice the decision depends on a
few factors shown in the following table:
In the open source ecosystem, reinforcement learning methods like GRPO and REINFORCE tend to be the most widely used,
although the Qwen3 tech report (A. Yang, Li, et al., 2025) highlighted the use of on-policy distillation to train the models
under 32B parameters:
One interesting property of on-policy distillation with small models is that it typically outperforms RL-based methods at a
fraction of the compute cost. This is because instead of generating multiple rollouts per prompt, we only sample one, which
is then graded by the teacher in a single forward/backward pass. As the Qwen3 tech report shows, the gains over GRPO
can be significant:
More recently, Thinking Machines have shown that on-policy distillation is also effective at mitigating catastrophic forgetting
, where a post-trained model is further trained on a new domain and its prior performance regresses. In the following table,
they show that although the chat performance of Qwen3-8b (IFEval) tanks when it’s fine-tuned on internal data, the behavior
can be restored with cheap distillation:
We’re quite excited by on-policy distillation, as there’s a huge diversity of capable, open-weight LLMs that can be distilled
into smaller, task-specific models. However, one weakness with all on-policy distillation methods is that the teacher and
Algorithm
Online DPO You can get preference labels cheaply. Best for aligning behavior with evolving distributions.
On-policy distillationYou have access to a stronger teacher model and want to transfer capabilities efficiently.
Reinforcement learningYou have verifiable rewards or tasks requiring multi-step reasoning/planning. Can be used with reward mo
Lightweight Models
Flagship Models
Base ModelsStage 1: Long CoT Cold Start Stage 2: Reasoning RLStage 3: Thinking Mode Fusion Stage 4: General RLQwen3-235B-A22BQwen3-32B
Base ModelsStrong-to-Weak DistillationQwen3-30B-A3B 14B/8B/4B/1.7B/0.6B
Method AIME24AIME25MATH500LiveCodeBench v5MMLU-ReduxGPQ DiamondGPU Hours
Off-policy distillation55.0 42.8 92.4 42.0 86.4 55.6 -
+ Reinforcement learning67.6 55.5 94.8 52.9 86.9 61.3 17,920
+ On-policy distillation74.4 65.5 97.0 60.3 88.3 63.3 1,800


student must share the same tokenizer. To address that, we’ve developed a new method called General On-Policy Logit
Distillation (GOLD) , which allows any teacher to be distilled into any student. We recommend checking out our technical
write-up if you’re interested in these topics.
Similarly, researchers at FAIR have compared the effect of being fully off-policy to on-policy for DPO and shown that it’s
possible to match the performance of GRPO using far less compute (Lanchantin et al., 2025):
As shown in their paper, online DPO works well for math tasks, and even the semi-on-policy variant achieves comparable
performance despite being many steps off policy:
Training Method Math500 NuminaMath AMC23
Seed (Llama-3.1-8B-Instruct) 47.4 33.9 23.7
Offline DPO (s = ∞) 53.7 36.4 28.8
Semi-online DPO (s = 100) 58.9 39.3 35.1
Semi-online DPO (s = 10) 57.2 39.4 31.4
Online DPO (s = 1) 58.7 39.6 32.9
GRPO 58.1 38.8 33.6
Overall, we feel that there still remains much to be done with both scaling RL effectively (Khatri et al., 2025) and exploring
other methods for computational efficiency. Exciting times indeed!
Wrapping Up Post-Training
If you’ve made it this far, congrats: You now have all the core ingredients needed for success with post-training. You’re
ready to run experiments and test different algorithms to get SOTA results.
But as you’ve probably realized, knowing how to train great models is only half the story. To actually bring those models to
life, you need the right infrastructure. Let’s finish this opus with a look at the unsung hero of LLM training.
