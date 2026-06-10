# Chapter 2: Training Compass: Why → What → How

Training Compass: Why → What → How
The field of machine learning has an obsessive relationship with optimization. We fixate on loss curves, model
architectures, and throughput; after all, machine learning is fundamentally about optimizing the loss function of a model.
Yet before diving into these technical details, there’s a more fundamental question that often goes unasked: Should we
even be training this model?
As shown in the following heatmap, the open source AI ecosystem releases world-class models on a nearly daily basis:
Qwen, Gemma, DeepSeek, Kimi, Llama 🪦 , OLMo, the list grows longer every month. These aren’t just research prototypes
or toy examples: They’re production-grade models covering an astonishing breadth of use cases, from multilingual
understanding to code generation and reasoning. Most of them come with permissive licenses and active communities
ready to help you use them.


↗View interactive version
Which raises an uncomfortable point: Maybe you don’t need to train your own model .
This might seem like an odd way to start an “LLM training guide.” But many failed training projects didn’t fail because of bad
hyperparameters or buggy code; they failed because someone decided to train a model they didn’t need. So before you
commit to training, and dive into how to execute it, you need to answer two questions: Why are you training this model?
What model should you train? Without clear answers, you may waste months of compute and engineering time building
something the world already has, or worse, something nobody needs.
Let’s start with the why , because without understanding your purpose, you can’t make coherent decisions about anything
that follows.
📍About this section


This section is different from the rest of this guide: It’s less about experiments and technical details, more about strategic
planning. We’ll walk you through deciding whether you need to train from scratch and what model to build. If you’ve already
thought deeply about your why and what, feel free to jump to “Every Big Model Starts with a Small Ablation” for the technical
deep dive. But if you’re uncertain, investing time here may well save you a lot of effort later.
Why: The Question Nobody Wants to Answer
Let’s be blunt about what happens in practice. Someone (if they’re lucky) gets access to a GPU cluster, maybe through a
research grant, maybe through a company’s spare capacity, and the thought process goes roughly like this: “We have 100
H100s for three months. Let’s train a model!” The model size gets picked arbitrarily, the dataset gets assembled from
whatever’s available. Training starts. And six months later, after burning through compute budget and team morale, the
resulting model sits unused because nobody ever asked why.
Here are some reasons why you shouldn’t train a model:
The allure of “we trained our own model” is powerful, but before investing a lot of time and resources, it makes sense to
ask: Why do we need to train this model?
The following flowchart guides the thought process one should go through before starting a big pretraining project. From a
technical perspective, you should essentially first find out if there’s an existing model that you can either prompt or fine-
tune to do the job.


There are basically three common areas where custom pretraining can make sense: you want to do novel research, you
have very specific needs for your production use case, or you want to fill a gap in the open model ecosystem. Let’s have a
quick look at each.
RESEARCH: WHAT DO YOU WANT TO UNDERSTAND?
There’s plenty of research one can do in the LLM space. What LLM research projects have in common is that you normally
start with a clearly defined question:
Can we scale training on this new optimizer to a 10B+ model? (See “Muon is Scalable for LLM Training.”)
Can reinforcement learning alone, without supervised fine-tuning (SFT), produce reasoning capabilities? (See “DeepSeek-
R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.”


Can we train good small models on purely synthetic textbook data? (See “Textbooks Are All You Need.”)
Can we achieve competitive performance by training on only openly licensed data? (See “The Common Pile v0.1: An 8TB
Dataset of Public Domain and Openly Licensed Text.”)
Making the hypothesis as concrete as possible and thinking about the necessary experiment scale increases the chances
of success.
PRODUCTION: WHY CAN’T YOU USE AN EXISTING MODEL?
There are three main reasons why companies can’t use off-the-shelf models for their use cases. Two of them are technical,
and the other is due to governance.
The first reason to train your own model is domain specificity : when your data or tasks involve highly specialized vocabulary
or structure that existing models can’t handle well. For example, you might want to train:
A DNA model with a unique vocabulary and long-range dependencies
A legal or financial model requiring deep familiarity with domain-specific jargon and logic
A second, related reason is deployment constraints : when you need a model tailored to your hardware, latency, or privacy
requirements. Examples here might be an LLM running on a drone or an on-prem system with custom hardware, like FPGAs.
Here’s a simple test: Spend a few days building on top of Qwen3, Gemma 3, or another current state-of-the-art (SOTA)
model. Can you reach your performance goals through prompting, tool use, or post-training? If not, it’s probably time to train
your own.
The third reason to build your own in-house language model is safety and governance : You need complete control over
training data, model behavior, and update cycles because you’re in a regulated industry or high-stakes application. You
need to know exactly what went into the model and be able to prove it to regulators. In some cases, you might have no
other option than building your own model.
These are the main reasons companies train in-house models—but what about companies or organizations that release
open models?
STRATEGIC OPEN SOURCE: DO YOU SEE A GAP YOU CAN FILL?
One of the most common reasons experienced AI labs release new open models is that they’ve identified a specific gap or
new AI use case in the open source ecosystem.
The pattern typically looks like this: You notice an underexplored area—maybe there are no strong on-device models with
very long context, or multilingual models exist but they’re weak on low-resource languages, or the field is moving toward
interactive world models like Genie 3 and no good open-weight model exists.
You have reason to believe you can do better; perhaps you’ve curated better training data, developed better training recipes,
or have the compute to overtrain where others couldn’t. Your goal is concrete: not “the best model ever,” but “the best 3B
model for on-device use” or “the first small model with a 1M-token context window.”
This is a real goal, and success creates value: Developers adopt your model, it becomes infrastructure for others, or it
establishes technical credibility. But success requires experience. You need to know what’s actually feasible and how to
execute reliably in a competitive space. To make this concrete, let’s look at how we think about this question at Hugging
Face.
Even if the post-training budget needed to meet your requirements is immense, it might still be cheaper than starting from
scratch. Fine-tuning your model for 1T tokens is more economical than starting from scratch to train for 10T+ tokens.

HUGGING FACE’S JOURNEY
This includes datasets, tooling, and training models. Every LLM training project we’ve started began with noticing a gap and
believing we could contribute something meaningful.
We started our first LLM project after GPT-3 (Brown et al., 2020) was released. At the time, it felt like no one else was
building an open alternative, and we were worried that the knowledge would end up locked away in just a few industry labs.
So we launched the BigScience workshop to train an open version of GPT-3. The resulting model was BLOOM, created by
dozens of contributors who worked for a year to build the training stack, tokenizer, and pretraining corpus to pretrain a
175B-parameter model.
The successor of BLOOM was StarCoder, in 2022 (Li et al., 2023). OpenAI had developed Codex for GitHub Copilot (Chen et
al., 2021), but it was closed source. Building an open source alternative clearly would provide value to the ecosystem. So,
in collaboration with ServiceNow, under the BigCode umbrella, we built The Stack dataset, and we trained StarCoder 15B to
reproduce Codex. StarCoder2 (Lozhkov et al., 2024) came from learning we could have trained longer, and recognizing that
smaller models trained for longer might be more valuable than one large model. We trained a family (3B/7B/15B) on
multiple trillions of tokens, far beyond what anyone had done for open code models at the time.
The SmolLM family followed a similar pattern. We noticed there were very few strong small models, and we had just built
FineWeb-Edu (Penedo et al., 2024), which was a strong pretraining dataset. SmolLM (135M/360M/1.7B) was our first
version. SmolLM2 (Allal et al., 2025) focused on better data and training longer, reaching SOTA performance on multiple
fronts. SmolLM3 scaled to 3B parameters while adding hybrid reasoning, multilinguality, and long context, features that the
community values in 2026.
Hopefully, this section has convinced you that there is value in thinking deeply about why you want to train a model.
For the rest of this guide, we’ll assume you’ve done this soul searching and have a legitimate reason to train.
What: Translating Goals into Decisions
Now that you know why you’re training, what should you train? By “what,” we mean the model type (dense, mixture of
experts, hybrid, something new), model size, architecture details, and data mixture. Once you’ve settled on the why, you can
derive the what. For example:
Fast model for on-device → small efficient model
Multilingual model → large tokenizer vocab
Super long context → hybrid architecture
Besides decisions driven by the use case, there are also some choices that optimize the training itself, by being more
stable, more sample efficient, or faster. These decisions are not always so clear-cut, but you can divide the decision
process into roughly these phases:
Planning: Before running experiments, map your use case to the components you need to decide on. Your deployment
environment determines model size constraints. Your timeline determines which architectural risks you can take. Your
So why does Hugging Face train open models? The answer is simple: We build things that are useful to the open source
ecosystem and fill gaps that few others are filling.
This pattern extends beyond pretraining: We trained Zephyr (Tunstall et al., 2023) to show Direct Preference Optimization
(DPO) works at scale, started Open-R1 to reproduce DeepSeek-R1’s distillation pipeline, and released OlympicCoder for
competitive programming, with SOTA performance in the International Olympiad in Informatics. We’ve also explored other
modalities with SmolVLM (Marafioti et al., 2025) for vision and SmolVLA (Shukor et al., 2025) for robotics.

target capabilities determine dataset requirements. This phase is about connecting each constraint from your “why” to
concrete specifications in your “what.”
Validation: Once you have a starting point and a list of potential modifications, test systematically. Since testing is
expensive, focus on changes that could meaningfully improve performance for your use case or optimize your training.
This is where ablations, covered in “