# Chapter 6: The Art of Data Curation

The Art of Data Curation
Picture this: You’ve spent weeks perfecting your architecture, tuning hyperparameters, and setting up the most robust
training infrastructure. Your model converges beautifully, and then… it can’t write coherent code, struggles with basic math,
and maybe even switches languages mid-sentence. What went wrong?
The answer usually lies in the data. While we obsess over fancy architectural innovations and hyperparameter sweeps, data
curation often determines whether our model becomes genuinely useful or is just another expensive experiment.
If model architecture defines how your model learns, then data defines what it learns, and no amount of compute or
optimizer tuning can compensate for training on the wrong content. It’s the difference between random web crawls versus
carefully curated, high-quality datasets that actually teach the skills we want our model to learn. But getting the training
data right isn’t just about having good datasets. It’s about assembling the right mixture : balancing conflicting objectives
(like strong English vs. robust multilinguality) and tuning data proportions to align with our performance goals. This process
is less about finding a universal “best” mix and more about asking the right questions and devising concrete plans to
answer them:
What do we want our model to be good at?
Which datasets are best for each domain, and how do we mix them?
Do we have enough high-quality data for our target training scale?
This chapter will hep you navigate these questions, using a mix of principled methods, ablation experiments, and a little bit
of alchemy to turn a pile of great datasets into a great training mixture.
What’s a Good Data Mixture, and Why Does It Matter So Much?
We expect a lot from our language models. They should be able to help us write code, give us advice, answer questions
about pretty much anything, complete tasks using tools, and more. Plentiful pretraining data sources, like the web, don’t
cover the full range of knowledge and capabilities needed for these tasks. As a result, recent models additionally rely on
more specialized pretraining datasets that target specific domains, such as math and coding. We’ve done a lot of work in
the past on curating datasets, but for SmolLM3 we primarily made use of preexisting datasets. (To learn more about
dataset curation, check out our reports on building FineWeb and FineWeb-Edu, FineWeb2, and Stack-Edu and FineMath.)
THE UNINTUITIVE NATURE OF DATA MIXTURES

If you’re new to training language models, finding a good data mixture might seem straightforward: Identify your target
capabilities, gather high-quality datasets for each domain, and combine them. The reality is more complex, since some
domains might compete with each other for your training budget. When focusing on some particular capability, like coding, it
can be tempting to upweight task-relevant data, like source code. However, upweighting one source implicitly downweights
all of the other sources, which can harm the language model’s capabilities in other settings. Training on a collection of
different sources therefore involves striking some kind of balance between downstream capabilities.
Additionally, across all these sources and domains, there’s often a subset of “high-quality” data that is especially helpful at
improving the language model’s capabilities. Why not just throw out all the lower-quality data and train on only the highest-
quality data? For a model with a large training budget, like SmolLM3’s 11T tokens, doing such extreme filtering would result
in repeating data many times. Prior work has shown that this kind of repetition can be harmful (Muennighoff et al., 2025),
so we should ideally be able to make use of higher- and lower-quality data while still maximizing model performance.
To balance data across sources and ensure we have enough high-quality data, we need to carefully design the mixture : the
relative proportion of training documents from each source. Since a language model’s performance on any given task or
domain depends heavily on the amount of data it has seen that is relevant to that task, tuning the mixing weights provides
a direct way of balancing the model’s capabilities across domains. Because these trade-offs are model-dependent and
difficult to predict, ablations are essential.
In addition, the mixture doesn’t have to stay fixed throughout training. By adjusting the mixture as training progresses—what
we call multi-stage or curriculum training —we can make better use of both high-quality and lower-quality data.
THE EVOLUTION OF TRAINING CURRICULA
In the early days of large language model training, the standard approach was to fix a single data mixture for the entire
training run. Models like GPT-3 and early versions of Llama trained on a static mixture from start to finish. More recently,
the field has shifted toward a multi-stage approach (Allal et al., 2025), where the data mixture changes over the course of
training. The main motivation is that a language model’s final behavior is strongly influenced by data seen toward the end of
training (Y. Chen et al., 2025b). This insight enables a practical strategy: upweighting more plentiful sources early in training
and mixing in smaller, higher-quality sources toward the end.
A common question is: How do you decide when to change the mixture? There’s no universal rule, but we typically follow
these principles:
1. Performance-driven interventions: Monitor evaluation metrics on key benchmarks and adapt dataset mixtures to address
specific capability bottlenecks. For example, if math performance plateaus while other capabilities continue improving,
that’s a signal to introduce higher-quality math data.
2. Reserve high-quality data for late stages: Small, high-quality math and code datasets are most impactful when
introduced during the annealing phase (the final stage, with learning rate decay).
Now that we’ve established why mixtures matter and how curricula work, let’s discuss how to tune both.
Ablation Setup: How to Systematically Test Data Recipes
When testing data mixtures, our approach is similar to how we run architecture ablations, with one difference: We try to run
them at the target model scale. Small and large models have different capacities. For example, a very small model might
struggle to handle many languages, while a larger one can absorb them without sacrificing performance elsewhere.
Therefore, running data ablations at too small a scale risks drawing the wrong conclusions about the optimal mix.
For SmolLM3, we ran our main data ablations directly on the 3B model, using shorter training runs of 50B and 100B
tokens. We also used another type of ablation setup: annealing experiments . Instead of training from scratch with different

mixtures, we took an intermediate checkpoint from the main run (for example, at 7T tokens) and continued training with
modified data compositions. This approach, used in recent work such as SmolLM2, Llama 3, and OLMo 2, allows us to test
data mixture changes for multi-stage training. For evaluation, we expanded our benchmark suite to include multilingual
tasks alongside our standard English evaluations, ensuring we could properly assess the trade-offs between different
language ratios.
Recent work has proposed automated approaches for finding optimal data proportions. For example:
DoReMi (Xie et al., 2023) uses a small proxy model to learn domain weights that minimize validation loss.
RHO-LOSS (Mindermann et al., 2022) selects individual training points based on a holdout loss, prioritizing samples that
are learnable, task-relevant, and not yet learned by the model.
RegMix (Q. Liu et al., 2025) determines optimal data mixture proportions through regularized regression that balances
performance across multiple evaluation objectives and data domains.
We experimented with DoReMi and RHO-LOSS in past projects but found they tended to converge toward distributions that
roughly mirror the natural distribution of dataset sizes, essentially suggesting to use more of what we have more of. While
theoretically appealing, they didn’t outperform careful manual ablations in our setting. Recent SOTA models still rely on
manual mixture tuning through systematic ablations and annealing experiments, which is the approach we adopted for
SmolLM3.
SmolLM3: Curating the Data Mixture
For SmolLM3, we wanted a model that could handle English and multilingual content and excel at math and code. These
content types are common in most LLMs, but the process we’ll describe here applies equally if you’re training for a low-
resource language or a specific domain such as finance or healthcare. The method is the same: Identify good candidate
datasets, run ablations, and design a mixture that balances all the target domains.
We won’t cover how to build high-quality datasets here, since we’ve already detailed that extensively in earlier work. Instead,
this section focuses on how we combine those datasets into an effective pretraining mixture.
BUILDING ON PROVEN FOUNDATIONS
When it comes to pretraining data, the good news is that we rarely have to start from scratch. The open source community
has already built strong datasets for most common domains. Sometimes we need to create something new, as we did with
the Fine series (FineWeb, FineMath, etc.), but more often the challenge is in selecting and combining existing sources
rather than reinventing them.


That was our situation with SmolLM3. SmolLM2 had already established a strong recipe at 1.7B parameters for English web
data and identified the best math and code datasets we had access to. Our goal was to scale that success to 3B
parameters while adding capabilities such as robust multilinguality, stronger math reasoning, and better code generation.
ENGLISH WEB DATA: THE FOUNDATION LAYER
Web text forms the backbone of any general-purpose LLM, but quality matters as much as quantity.
From SmolLM2, we knew that FineWeb-Edu and DCLM were the strongest open English web datasets at the time of training.
Together, they gave us 5.1T tokens of high-quality English web data. The challenge was determining the optimal mixing
ratio: FineWeb-Edu helps on educational and STEM benchmarks, while DCLM improves common-sense reasoning.
Following the SmolLM2 methodology, we ran a sweep on our 3B model over 100B tokens, testing FineWeb-Edu/DCLM
ratios of 20/80, 40/60, 50/50, 60/40, and 80/20. We found that mixing them at a ratio of 60/40 or 50/50 provided the
best balance across benchmarks, matching our SmolLM2 findings. We decided to use a 50/50 ratio for stage 1. We also
added a few other datasets, like pes2o, Wikipedia & Wikibooks, and StackExchange. This didn’t have any impact on the
performance, but we included them to improve diversity.
MULTILINGUAL WEB DATA
For multilingual capability, we targeted five other languages: French, Spanish, German, Italian, and Portuguese. We selected
them from FineWeb2-HQ, which gave us a totall of 628B tokens. We also included 10 other languages at smaller ratios,
such as Chinese, Arabic, and Russian, not to target state-of-the-art performance for them but to allow people to easily do
continual pretraining of SmolLM3 on these languages. We used FineWeb2 for the languages not supported in FineWeb2-HQ.
The key question was: How much of our web data should be non-English? We know that the more data a model sees in a
language or domain, the better it gets at that language or domain. However, our fixed compute budget meant that
increasing data for one language required reducing data for the other languages, including English.
Through ablations on the 3B model, we found that 12% multilingual content (about 14% of the overall web mix) struck the
right balance, improving multilingual performance without degrading English benchmarks. This fit SmolLM3’s expected
usage, where English would remain the primary language. It’s also worth noting that with only 628B tokens of non-English
data versus 5.1T English tokens, going much higher would have required more repetition of the multilingual data.
CODE DATA
Our code sources for stage 1 were extracted from the StarCoder2 and The Stack v2 training corpus. We included:
The Stack v2 (16 languages) as our basis, filtered as StarCoder2Data
StarCoder2 GitHub pull requests for real-world code review reasoning
Jupyter and Kaggle notebooks for executable, step-by-step workflows
GitHub issues and Stack Exchange threads for contextual discussions around code
Aryabumi et al. (2024) highlight that code improves language models’ performance beyond coding, for example on natural
language reasoning and world knowledge, and recommend using 25% code in the training mixture. Motivated by this, we
started our ablations with 25% code in the mixture. However, we observed significant degradation on English benchmarks
(HellaSwag, ARC-C, MMLU). Reducing to 10% code, we didn’t see improvements on our English benchmark suite compared
to 0% code, but we included it anyway since code generation was a very important capability to have in the model.
We delayed adding Stack-Edu—our educationally filtered subset of StarCoder2Data—until later stages, following the
principle of staging high-quality data for maximum late-training impact.

MATH DATA
We followed a similar philosophy for math as for code. Early on, we used the larger, more general sets FineMath3+ and
InfiWebMath3+. Later, we upsampled FineMath4+ and InfiWebMath4+ and introduced a few new high-quality datasets:
MegaMath (Zhou et al., 2025)
Instruction and reasoning datasets like OpenMathInstruct (Toshniwal et al., 2024) and OpenMathReasoning (Moshkov et
al., 2025)
We used 3% math data in stage 1, equally split between FineMath3+ and InfiWebMath3+. With only 54B tokens available
and an estimated 8T- to 9T-token stage 1, using more than this would have required more than five epochs on the dataset.
FINDING THE RIGHT MIXTURE FOR NEW STAGES
While we ran ablations from scratch to determine the best stage 1 mixture, to test new datasets for the next stages (in our
case, two of them) we used annealing ablations. We took a checkpoint at around 7T tokens (late in stage 1) and ran 50B-
token annealing experiments with the following setup:
40% baseline mixture: The exact stage 1 mixture we’d been training on.
60% new dataset: The candidate dataset we wanted to evaluate.
For example, to test whether MegaMath would improve our math performance, we ran an annealed mixture consisting of
40% stage 1 data (maintaining the 75/12/10/3 domain split) and 60% MegaMath data.
You’ll find details on the composition of all three stages in the following chapter.
With our data carefully curated and our mixture validated through ablations, we were ready to embark on the actual training
journey. What follows is the story of SmolLM3’s month-long training run: the preparation, the unexpected challenges, and
the lessons learned along the way.
