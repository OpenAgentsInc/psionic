# Chapter 7: The Training Marathon

The Training Marathon
Well done on making it this far—the real fun is about to begin!
At this point, we have everything in place: a validated architecture, a finalized data mixture, and tuned hyperparameters. The
only thing left to do is set up the infrastructure and hit “train.”
For SmolLM3, we trained on 384 H100 GPUs (48 nodes) for nearly a month, processing 11 trillion tokens. This section
walks you through what actually happens during a long training run: the preflight checks, the inevitable surprises, and how
we kept things stable. You’ll see firsthand why both solid ablation practices and reliable infrastructure matter. We cover the
technical infrastructure details of GPU hardware, storage systems, and optimizing throughput in the final chapter.
Our team has been through this many times: from StarCoder and StarCoder2 to SmolLM, SmolLM2, and now SmolLM3.
Every single run is different. Even if you’ve trained a dozen models, each new run finds a fresh way to surprise you. Our aim
here is to show you how to stack the odds in your favor so you’re ready for those surprises.
Preflight Checklist: What to Verify Before Training Starts
Before hitting “train,” we go through a checklist to ensure everything works end-to-end. This includes:

Infrastructure readiness:
If your cluster supports Slurm reservations, use them. For SmolLM3, we had a fixed 48-node reservation for the
entire run. That meant no queueing delays, consistent throughput, and the ability to track node health over time.
Stress-test GPUs before launch (we use GPU Fryer and DCGM Diagnostics) to catch throttling or performance
degradation. For SmolLM3, we found two GPUs throttling and replaced them before starting the run.
Avoid storage bloat. Our system uploads each checkpoint to S3, then deletes the local copy right after saving the
next one, so we never store more than one on the fast local SSDs.
Evaluation setup: Evaluations are deceptively time-consuming. Running them manually, logging results, and making plots
can eat up hours. Automate them completely if you can, and ensure they are running and logging correctly before the run
starts. For SmolLM3, every saved checkpoint automatically triggered an evaluation job on the cluster that got logged to
Wandb and Trackio.
Checkpoint & auto-resume system: Verify that checkpoints are saved correctly and that the training job can resume from
the latest one without manual intervention. On Slurm, we use the --requeue option so a failed job gets automatically
relaunched, resuming from the most recent checkpoint.
Metrics logging: Confirm that you’re logging all the metrics you care about: evaluation scores, throughput (tokens/sec),
training loss, gradient norm, node health (GPU utilization, temperature, memory usage), and any custom debug metrics
specific to your run.
Training configuration sanity check: Double-check your training config, launch scripts, and Slurm submission commands.
Infrastructure deep dive
For detailed guidance on GPU testing, storage benchmarking, monitoring setup, and building resilient training systems, see
the Infrastructure chapter.
Scaling Surprises
We were ready for the big one: 11T tokens. That’s when reality started throwing curveballs.
MYSTERY #1—THE VANISHING THROUGHPUT
Within hours of launch, throughput plummeted. It was a steep fall, with repeated sharp drops.
📍Why throughput matters
Throughput measures how many tokens per second our system processes during training. It directly impacts training time—a
50% drop in throughput means our month-long run becomes a two-month run. In the Infrastructure chapter, we’ll show how
we optimized throughput for SmolLM3 before starting the run.
After running extensive ablations for SmolLM3, we were ready for the full-scale run. Our 3B ablations on 100B tokens
looked promising. The architectural changes compared to SmolLM2 (detailed in “Architecture Choices”: GQA, NoPE,
document masking, tokenizer) either improved or maintained performance, and we found a good data mixture that balanced
English, multilingual, code, and math performance (see “SmolLM3: Curating the Data Mixture”). We optimized our
configuration for around 30% model flops utilization (MFU) on 384 GPUs (48 nodes).

This didn’t happen in any ablation run, so what had changed? Three things:
1. The size of the training datasets. We were now using the full ~24 TB training dataset instead of the smaller subsets
used for ablations, though the data sources themselves were the same.
2. The number of training steps. We set the real step count for 11T tokens instead of the short 100B-token ablation
horizon.
3. Potentially, hardware state. GPUs that worked fine in ablations might fail and network connections might degrade under
sustained load.
Everything else remained exactly the same as in the throughput ablations: number of nodes, dataloader configuration,
model layout, parallelism setup…
Intuitively, neither dataset size nor step count should cause throughput drops, so we naturally suspected hardware issues
first. We checked our node-monitoring metrics, which showed that the big throughput fall correlated with spikes in disk read
latency. That pointed us straight at our data storage.
📍Storage options in our cluster
Our cluster has three storage tiers for training data:
FSx: Network-attached storage that uses Weka, a “keep-hot” caching model that stores frequently accessed files locally
and evicts inactive “cold” files to S3 as capacity fills up.
Scratch (Local NVMe RAID): Fast local storage on each node (8 × 3.5 TB NVMe drives in RAID), which is faster than FSx
but limited to local node access.
S3: Remote object storage for cold data and backups.
You can find more details in the Infrastructure chapter.


For SmolLM3’s 24 TB dataset, we initially stored the data in FSx (Weka). But with 24 TB of training data, on top of storage
already used by several other teams, we were pushing Weka’s storage to the limit. It started evicting dataset shards mid-
training, which meant we had to fetch them back, creating stalls, which explained the big throughput dropoff.
Fix #1—Changing Data Storage
We couldn’t find a way to pin our dataset folders as hot for the full training in Weka, so we tried to change the storage
method. Streaming directly from S3 was slow, so we decided to store the data on each node in its local scratch storage.
This came with a catch: If a node died and was replaced, the new replacement GPUs had no data. Downloading 24 TB from
S3 with s5cmd took 3 hours. We cut that to 1.5 hours by copying from another healthy node using fpsync instead of
going through S3. This was faster because all the nodes were in the same datacenter.
Still, 1.5 hours of downtime per node failure, and the need to manually copy the data to the new node immediately, was
painful. The hack that finally made it bearable was reserving a spare node in our Slurm reservation with the dataset
preloaded. If a node died, we swapped it instantly with the spare node, so there was zero recovery delay. While idle, the
spare ran evals or dev jobs, so it wasn’t wasted.
This fixed Mystery #1… or so we thought.
MYSTERY #2—THE PERSISTENT THROUGHPUT DROPS
Even after moving the data to scratch, the individual throughput drops kept happening, although we didn’t find any
anomalies in the hardware monitoring metrics. The following chart compares the throughput we got after fixing the storage
issue (in orange) to the throughput we were getting during the ablations (in blue). As you can see, the drops actually
became much sharper.


Still suspecting hardware, we decided to test on fewer nodes. With 384 GPUs, there’s a high chance something could be
failing. Surprisingly, we were able to reproduce the exact same throughput drops on a single node, no matter which specific
node we tested. This ruled out hardware issues.
Remember the three things that had changed from our ablations? We had already addressed the data storage issue by
moving to local node storage. Hardware was now eliminated as well. That left only one variable: the step count. We tested
this by rolling back to smaller step counts (from 3M to 32k), and the throughput drops became smaller! Larger step counts
produced sharper, more frequent drops.
Here are the exact configs we used, with everything else remaining unchanged:
The results, as shown in the following figure, were clear: Shorter runs produced small throughput drops, while longer step
counts produced sharper, more frequent drops. So the issue was not the hardware, but a software bottleneck—likely in the
dataloader, given that most other training components process each batch identically regardless of step count.
## Short run (32k steps)
1
- "lr_decay_starting_step": 2560000
2
- "lr_decay_steps": 640000
3
- "train_steps": 3200000
4
 
5
## Long run (3.2M steps)
6
+ "lr_decay_starting_step": 26000
7
+ "lr_decay_steps": 6000
8
+ "train_steps": 32000
9
 
10


That’s when we realized we’d never actually done large-scale pretraining with Nanotron’s dataloader. SmolLM2 had been
trained with steady throughput using a Megatron-LM-derived dataloader ( TokenizedBytes ) through an internal wrapper
around Nanotron. For SmolLM3, we’d switched to Nanotron’s built-in dataloader ( nanosets ).
After a deep dive into its implementation, we found that it was naively building one giant index that grew with each training
step. For very large steps, this caused higher shared memory usage, which triggered throughput drops.
Fix #2—Bringing in the TokenizedBytes Dataloader
To confirm that the dataloader was indeed the culprit, we launched the same configuration with our internal SmolLM2
framework using the TokenizedBytes dataloader. We got no drops on 48 nodes using the same datasets.
Fastest path forward: Copy this dataloader into Nanotron. The drops were gone and the throughput back to target.
We were ready to relaunch… until the next curveball.
MYSTERY #3—THE NOISY LOSS
With the new dataloader, we didn’t have throughput drops, but the loss curve looked noisier.
nanosets had been producing smoother loss, and the difference rang a bell from an old debugging war: A few years ago,
we’d found a shuffling bug in our pretraining code where documents were shuffled but sequences inside a batch were not,
leading to small spikes.
Checking our new dataloader confirmed it: It was reading sequences sequentially from each document. That’s fine for short
files, but with domains like code, a single long low-quality file can fill an entire batch and cause loss spikes.
Fix #3—Shuffling at the Sequence Level
We had two options:
1. Change the dataloader to do random access (risk: higher memory usage).
2. Pre-shuffle tokenized sequences offline.
With the time pressure to start the run and our cluster reservation running, we went with option #2 as the safer, faster fix.
Tokenized data was already on each node, so reshuffling locally was cheap (~1 h). We generated shuffled sequences for
each epoch with different seeds to avoid repeating shuffling patterns across epochs.
Know when to patch vs. fix
When facing urgent deadlines, it might be faster to adopt a proven solution or quick workaround than to debug your own
broken implementation. Earlier, we plugged in the TokenizedBytes dataloader rather than fixing the index implementation
in nanosets . Here, we chose offline pre-shuffling over dataloader changes. Be careful about taking too many shortcuts,
though, or you’ll end up with a patchwork system that’s hard to maintain or optimize.
LAUNCH, TAKE TWO
By now we had:
Stable throughput (scratch storage + spare node strategy)
No step count–induced drops ( TokenizedBytes dataloader)
Clean, sequence-level shuffling (offline pre-shuffle per epoch)

We relaunched. This time, everything held. The loss curve was smooth, throughput was consistent, and we could finally
focus on training instead of firefighting.
MYSTERY #4—UNSATISFACTORY PERFORMANCE
We trained smoothly for the first two days. Nothing in the logs suggested any problems. At around the 1T token mark,
however, the evaluations revealed something unexpected.
As part of our monitoring, we evaluate intermediate checkpoints and compare them to historical runs. For instance, we had
the intermediate checkpoints from SmolLM2 (1.7B) trained on a similar recipe, so we could track how both models
progressed at the same stages of training. The results were puzzling: Despite having more parameters and a better data
mixture, the 3B model was performing worse than the 1.7B model at the same training point. Loss was still decreasing, and
benchmark scores were improving, but the improvement rate was clearly below expectations.
Given that we had thoroughly tested every architecture and data change introduced in SmolLM3 compared to SmolLM2, we
turned our attention to the training framework. There were only a few remaining untested differences between the two
training setups. The most obvious was tensor parallelism (TP): SmolLM2 could fit on a single GPU and was trained without
TP, while SmolLM3 required TP=2 to fit in memory. We hadn’t thought of testing it before, since TP was used in the 3B
ablations and their results made sense.
Fix #4—The Final Fix
To test the TP bug hypothesis, we trained a 1.7B model with the exact same setup as SmolLM3—same architecture
changes (document masking, NoPE), same data mixture, same hyperparameters—both with and without TP. The difference
was immediately clear: The TP version consistently had a higher loss and lower downstream performance than the non-TP
version. That confirmed we were looking at a TP-related bug.
We then examined the TP implementation in detail, comparing weights from TP and non-TP runs. The problem turned out to
be subtle but significant: We were using identical random seeds across all TP ranks, when each rank should have been
initialized with a different seed. This caused correlated weight initialization across shards, which affected convergence. The
effect was not catastrophic—the model still trained and improved—but it introduced enough inefficiency to explain the gap
we observed at scale.
Here’s the bug fix:
diff --git a/src/nanotron/trainer.py b/src/nanotron/trainer.py
1
index 1234567..abcdefg 100644
2
--- a/src/nanotron/trainer.py
3
+++ b/src/nanotron/trainer.py
4
@@ -185,7 +185,10 @@ class DistributedTrainer:
5
     ):
6
         # Set random states
7
-        set_random_seed(self.config.general.seed)
8
+        # Set different random seed for each TP rank to ensure diversity
9
+        tp_rank = dist.get_rank(self.parallel_context.tp_pg)
10
+        set_random_seed(self.config.general.seed + tp_rank)
11
 
12
+
13

Once we had fixed the seeds so that each TP rank used a different seed, we repeated the ablation experiments and
confirmed that TP and non-TP runs now matched in both loss curves and downstream performance. To make sure there
were no other hidden issues, we ran additional sanity checks: a SmolLM2-style (architecture and data-wise) run at 3B
parameters, and a separate SmolLM3 run at 3B parameters, comparing both to SmolLM2’s checkpoints. The results now
aligned with expectations: The 1.7B SmolLM2 performed worse than the 3B SmolLM2 variant, which in turn was
outperformed by SmolLM3-3B.


This debugging process reinforced one of the core principles we outlined earlier: “The real value of a solid ablation setup
goes beyond just building a good model. When things go wrong during our main training run (and they will, no matter how
much we prepare), we want to be confident in every decision we made and able to quickly identify which components
weren’t properly tested and could be causing the issues. This preparation saves debugging time and protects our future
mental sanity.”
There’s nothing worse than staring at a mysterious training failure with no idea where the bug could be hiding. Because
every other component in our training setup had been validated, we were able to pinpoint TP as the only plausible cause
and fix the bug within a single day of detecting the performance gap.
With that, we had resolved the last in a series of unexpected issues that had surfaced since launch. Third time’s the charm:
From that point on, the rest of the month of training was relatively uneventful—just the steady work of turning trillions of
tokens into a finished model, interrupted by occasional restarts due to node failures.
Staying the Course
As the previous section showed, scaling from ablations to full pretraining wasn’t just “plug and play.” It brought unexpected
challenges, but we successfully identified and resolved each issue. This section covers the essential monitoring setup and
considerations for large-scale training runs. We’ll address critical questions: When should you restart training after
encountering problems? How do you handle issues that surface deep into a run? Which metrics truly matter? Should you
maintain a fixed data mixture throughout training?
TRAINING MONITORING: BEYOND LOSS CURVES


The reason we caught the tensor parallelism bug was not the loss curve, which looked fine, but the fact that downstream
evaluations were lagging behind expectations. Additionally, having evaluations from SmolLM2’s intermediate checkpoints
was critical, as they gave us an early indication that the 3B model wasn’t on the right track. So, if you’re training large
models, start running downstream evaluations early, and if you’re comparing to an open source model, ask whether the
authors can provide intermediate checkpoints. Those can be invaluable as reference points.
On the infrastructure side, the most important metric is throughput, measured in tokens per second. For SmolLM3, we
expected stable throughput between 13,500–14,000 tokens/sec across the run, and any sustained deviation was a red
flag. But throughput alone is not enough: You also need continuous hardware health monitoring to anticipate and detect
hardware failures. Some of the key metrics we tracked included GPU temperatures, memory usage, and compute utilization.
We logged them into Grafana dashboards and set up real-time Slack alerts for hardware anomalies.
FIX AND RESTART VS. FIX ON THE FLY
Given that we restarted our run after 1T tokens, an important question arises: Do you always need to restart when
something goes wrong? The answer depends on the severity and root cause of the issue.
In our case, the TP seeding bug meant we were starting on the wrong foot; half our weights weren’t properly initialized. The
model was showing performance similar to SmolLM2’s and plateauing at similar points, meaning we’d likely end up with a
model that performed the same but cost almost twice as much to train. Restarting made sense. However, many issues can
be course-corrected mid-run to avoid wasting compute. The most common issue involves loss spikes, those sudden jumps
in training loss that can signal either minor hiccups or divergence.
As Stas Bekman nicely puts it in the Machine Learning Engineering Open Book, “Training loss plots are similar to heartbeat
patterns—there’s the good, the bad, and the you-should-worry ones.”
↗View interactive version
Loss spikes fall into two categories:
Recoverable spikes: These may recover either quickly (immediately after the spike) or slowly (requiring several more
training steps to return to the pre-spike trajectory). You can usually continue training through them. If recovery is very
slow, you can try rewinding to a previous checkpoint to skip problematic batches.


Non-recoverable spikes: With these, the model either diverges or plateaus at worse performance than before the spike.
They require more significant intervention than simply rewinding to a previous checkpoint.
While we don’t fully understand training instabilities, we know they become more frequent at scale. Common culprits,
assuming a conservative architecture and optimizer, include:
High learning rates: These cause instability early in training and can be fixed by reducing the learning rate.
Bad data: This is usually the main cause of recoverable spikes, though recovery may be slow. Issues can arise deep into
training, when the model encounters low-quality data.
Data–parameter state interactions: PaLM (Chowdhery et al., 2022) observed that spikes often result from specific
combinations of data batches and model parameter states, rather than “bad data” alone. Training on the same
problematic batches from a different checkpoint didn’t reproduce the spikes.
Poor initialization: Recent work by OLMo 2 (OLMo et al., 2025) showed that switching from scaled initialization to a
simple normal distribution (mean=0, std=0.02) improved stability.
Precision issues: While no one trains with FP16 anymore, BLOOM found it highly unstable compared to BF16.
What can you do?
Before Spikes Happen, Build in Stability
Small models with conservative learning rates and good data rarely spike, but larger models require proactive stability
measures. As more teams have trained at scale, we’ve accumulated a toolkit of techniques that help prevent training
instability. These include:
Data filtering and shuffling: By this point, you’ve noticed how often we circle back to data. Making sure your data is clean
and well-shuffled can prevent spikes. For instance, OLMo 2 found that removing documents with repeated n -grams (32+
repetitions of 1- to 13-token spans) significantly reduced spike frequency.
Training modifications: Z-loss regularization keeps output logits from growing too large without affecting performance.
Excluding embeddings from weight decay also helps.
Architectural changes: QK-norm (normalizing query and key projections before attention) has proven effective. OLMo2
and other teams found it helps with stability, and interestingly, the Marin team found that it can even be applied mid-run
to fix divergence issues.
When Spikes Happen Anyway—Damage Control
Even with these precautions, spikes can still occur. Here are some options for fixing them:
Skip problematic batches: Rewind to before the spike and skip the problematic batches. This is the most common fix for
spikes. The Falcon team (Almazrouei et al., 2023) skipped 1B tokens to resolve their spikes, while the PaLM team
(Chowdhery et al., 2022) found that skipping 200–500 batches around the spike location prevented recurrence.
Tighten gradient clipping: Reduce the gradient norm threshold temporarily.
Apply architectural fixes: As mentioned above, QK-norm may be effective.
We’ve walked through the scaling challenges we encountered (from throughput drops to the TP bug), monitoring practices to
catch problems early, and strategies for preventing and fixing loss spikes. We’ll finish this chapter by discussing how multi-
stage training can enhance your model’s final performance.
Mid-Training

Modern LLM pretraining typically involves multiple stages with different data mixtures, often followed by a final phase to
extend context length. For example, Qwen3 (A. Yang, Li, et al., 2025) uses a three-stage approach: a general stage on 30T
tokens at 4k context length, a reasoning stage with 5T higher-quality tokens emphasising STEM and coding, and finally a
long context stage on hundreds of billions of tokens at 32k context length. SmolLM3 follows a similar philosophy, with
planned interventions to introduce higher-quality datasets and extend context alongside reactive adjustments based on
performance monitoring.
As we explained in the previous chapter, the data mixture doesn’t have to stay fixed throughout training. Multi-stage training
allows us to strategically shift dataset proportions as training progresses. Some interventions are planned from the start:
For SmolLM3, we knew we’d introduce higher-quality FineMath4+ and Stack-Edu in stage 2, then add curated Q&A and
reasoning data during the final decay phase. Other interventions are reactive, driven by monitoring performance during
training. For example, in SmolLM2, when we found math and code performance lagging behind our targets, we curated
entirely new datasets (FineMath and Stack-Edu) and introduced them mid-training. This flexibility—whether following a
planned curriculum or adapting to emerging gaps—is what allows us to maximize the value of our compute budget.
STAGE 2 AND STAGE 3 MIXTURES
The following chart shows our three training stages and the progression of our web/code/math ratios during training. The
SmolLM3 training configs for each stage are available in the SmolLM GitHub repository, with exact data weights. For more
details on the rationale behind the composition of each stage, refer to “The Art of Data Curation.”
Stage 1—Base training (8T tokens, 4k context): The foundation stage uses our core pretraining mixture of web data
(FineWeb-Edu, DCLM, FineWeb2, FineWeb2-HQ), code from The Stack v2 and StarCoder2, and math from FineMath3+ and
InfiWebMath3+. All training happens at 4k context length.
Stage 2—High-quality injection (2T tokens, 4k context): In this stage, we introduce higher-quality filtered datasets, including
Stack-Edu for code, FineMath4+ and InfiWebMath4+ for math, and MegaMath for advanced mathematical reasoning (we
add the Qwen Q&A data, synthetic rewrites, and text–code interleaved blocks).
Stage 3—LR decay with reasoning and Q&A data (1.1T tokens, 4k context): During the learning rate decay phase, we further
upsample high-quality code and math datasets while introducing instruction and reasoning data like OpenMathReasoning,
OpenCodeReasoning, and OpenMathInstruct. The Q&A samples are simply concatenated and separated by newlines.
LONG CONTEXT EXTENSION: FROM 4K TO 128K TOKENS
Context length determines how much text your model can process. It’s crucial for tasks like analyzing long documents,
maintaining coherent multi-turn conversations, and processing entire codebases. SmolLM3 started training at 4k tokens,


but we needed to scale to 128k for real-world applications.
Why Extend Context Mid-Training?
Training on long contexts from the start is computationally expensive because attention mechanisms scale quadratically
with sequence length. Moreover, research shows that extending context with a few dozen to a hundred billion tokens toward
the end of training, or during continual pretraining, is enough to reach good long context performance (Gao et al., 2025).
Sequential Scaling: 4k → 32k → 64k
We didn’t jump straight to 128k. Instead, we gradually extended the context in stages, giving the model time to adapt at
each length before pushing further. We ran two long context stages: first increasing from 4k to 32k, then from 32k to 64k
(the 128k capability comes from inference-time extrapolation, not training). We found that starting a fresh learning rate
schedule for each stage over 50B tokens worked better than extending context during the last 100B tokens of the main
decay phase. At each stage, we ran ablations to find a good long context data mix and RoPE theta value, and evaluated on
the RULER benchmark.
💡Long context evals on the base model
During the long context ablations, we found the HELMET benchmark to be very noisy on base models (the same training with
different seeds gives variable results). Gao et al. (2025) recommend doing supervised fine-tuning on top to reduce variance
on the benchmarks’ tasks. We instead we opted for RULER, which we found to give more reliable signal at the base model
level.
RoPE with Adjusted Base Frequency (ABF)
When extending the context length from 4k to 32k, we increased the RoPE base frequency (theta) to 2M, and when
extending the context length from 32k to 64k we increased it to 5M. We found that using larger values, like 10M, slightly
improved the RULER score but hurt some short context tasks, such as GSM8k, so we kept it at 5M, which didn’t impact
those tasks.
YARN Extrapolation: Reaching 128k
We wanted SmolLM3 to handle 128k at inference, but training on 128k sequences would have been prohibitively expensive.
Instead, we used YARN (Yet Another RoPE extensioN method) (B. Peng et al., 2023), which allows the model to extrapolate
beyond its training length.
In theory, YARN allows a four-fold increase in sequence length. We found that using the 64k checkpoint gave better
performance at 128k than using the 32k checkpoint, confirming the benefit of training closer to the target inference length.
However, pushing to 256k (4 × 64k) showed degraded RULER performance, so we recommend only using the model up to
128k.
During this phase, it’s common to upsample long context documents such as lengthy web pages and books to improve long
context performance (Gao et al., 2025). We ran several ablations upsampling books, articles, and even synthetically
generated documents for tasks like retrieval and fill-in-the-middle, following Qwen2.5-1M’s approach (A. Yang, Yu, et al.,
2025) with FineWeb-Edu and Python-Edu. Surprisingly, we didn’t observe any improvement over just using the baseline
mixture from stage 3, which was already competitive with other state-of-the-art models like Llama 3.2 3B and Qwen2.5-3B
on RULER. We hypothesize that this is because the baseline mixture naturally includes long documents from web data and
code (estimated at 10% of tokens), and that using NoPE helped.
During this context extension phase, we also took the opportunity to further upsample math, code, and reasoning Q&A data,
and we added few hundred thousand samples in ChatML format.

And with that, we’ve walked through the full pretraining journey for SmolLM3, from planning and ablations to the final
training run, with all the behind-the-scenes challenges along the way.
Wrapping Up Pretraining
We’ve covered a lot of ground, from the training compass that helps decide why and what to train, through strategic
planning and systematic ablations that validate every architectural choice, to the actual training marathon where, in our
case, surprises emerged at scale (throughput mysteriously collapsing, dataloader bottlenecks, and a subtle tensor
parallelism bug that forced a restart at 1T tokens).
The messy reality behind all those polished technical reports is now visible: Training LLMs is as much about disciplined
experimentation and rapid debugging as it is about architectural innovations and data curation. Planning identifies what’s
worth testing. Ablations validate each decision. Monitoring catches problems early. And when things inevitably break,
systematic derisking tells you exactly where to look.
For SmolLM3 specifically, this process delivered what we set out to build: a 3B model trained on 11T tokens that’s
competitive on math, code, multilingual understanding, and long context tasks, in the Pareto frontier of Qwen3 models.
With our base model checkpoint saved and training complete, we might be tempted to call it done. After all, we have a
model that predicts text well, achieves strong benchmark scores, and demonstrates the capabilities we targeted. All set,
right?
Not quite. Because what people want today are assistants and coding agents, not raw next-token predictors.
This is where post-training comes in. And just like with pretraining, the reality is messier than you might think.
The win rate of base models evaluated on HellaSwag, ARC, WinoGrande, CommonsenseQA, MMLU-CF, MMLU Pro CF, PIQA, OpenBookQA,
GSM8K, MATH, HumanEval+, and MBPP+.