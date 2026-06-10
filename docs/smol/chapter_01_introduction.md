# Chapter 1: Introduction

The Sm ol Training Playbook:The Secrets to Building World-Class LLMs
A practical journey through the challenges, decisions, and messy reality behind training state-of-the-art
language models
AUTHORS
Loubna Ben Allal, Lewis Tunstall, Nouamane Tazi, 
Elie Bakouch, Ed Beeching, Carlos Miguel Patiño, 
Clémentine Fourrier, Thibaud Frere, Anton Lozhkov, 
Colin Raffel, Leandro von Werra, Thomas Wolf
AFFILIATION
Hugging Face
PUBLISHED
Oct. 30, 2025
Introduction
Published research makes it look straightforward: strategic architecture choices, carefully curated datasets, and sufficient
compute. The results are polished, the ablations are structured and clean. Every decision seems obvious in hindsight. But
those reports only show what worked and apply a bit of rosy retrospection—they don’t capture the 2 a.m. dataloader
debugging sessions, the loss spikes, or the subtle tensor parallelism bug (see later!) that quietly sabotages your training.
The reality is messier, more iterative, and full of decisions that don’t make it into the final paper.
Join us as we look behind the scenes of training SmolLM3, a 3B-parameter multilingual reasoning model trained on 11T
tokens. This is not an ordinary guide, but rather the untangling of a spiderweb of decisions, discoveries, and dead ends that
led to deep insights into what it takes to build world-class language models.
It is also the final opus in our long-form model training series. We’ve worked through building datasets at scale (FineWeb),
orchestrating thousands of GPUs to sing in unison (The Ultra-Scale Playbook), and selecting the best evaluations at each
step of the process (The LLM Evaluation Guidebook). Now we’re putting it all together to build a strong AI model. We’ll walk
you through the complete journey—not just the final recipe that worked, but the failures, infrastructure breakdowns, and
debugging processes that shaped every decision. You’ll see how promising small-scale ablations sometimes don’t translate
at scale; why we restarted a training run after 1T tokens; how we balanced the competing objectives of multilinguality, math,
and code while maintaining strong English-language performance; and finally how we post-trained a hybrid reasoning model.
What does it actually take to train a high-performance LLM today?


We’ve tried to avoid just listing everything we did in favor of presenting an organized story about our adventure. Think of this
as a guide for anyone trying to go from “we have a great dataset and GPUs” to “we built a really strong model.” We hope
being this open will help close the gap between research and production, and make your next training run a little less
chaotic.
How to Read This Guide
You don’t need to read the whole guide from start to finish, and at this point it’s too long to realistically read end to end in
one sitting anyway. It’s structured into several distinct pieces, any of which can be skipped or read individually:
Training compass: A high-level discussion about whether you should pretrain your own model. We walk you through
fundamental questions to ask yourself before burning through all your VC money, and how to think systematically
through the decision process. This is a high-level section; if you want to skip straight to the technical content, that’s
fine.
Pretraining: The sections following the training compass cover everything you need to know to build a solid recipe for
your own pretraining run: how to run ablations, select evaluations, mix data sources, make architecture choices, tune
hyperparameters, and finally endure the training marathon. This section is also relevant if you’re not planning to pretrain
from scratch but are interested in continued pretraining (aka mid-training).
Post-training: In this part of the guide you’ll learn all the tricks needed to get most out of your pretrained models. We’ll
cover the whole post-training alphabet, starting with SFT, DPO, and GRPO, as well as the dark arts and alchemy of model
merging. Most of the knowledge about making these algorithms work well is learned through painful lessons, and we’ll
share our experience here to hopefully spare you some of them.
Infrastructure: If pretraining is the cake and post-training is the icing and cherry on top, then infrastructure is the
industrial-grade oven. Without it, nothing happens, and if it’s broken, your happy Sunday baking session turns into a fire
hazard. Knowledge about how to understand, analyze, and debug GPU clusters is scattered across the internet in
various libraries, docs, and forums. This section walks through GPU layout, communication patterns between
CPU/GPU/nodes/storage, and how to identify and overcome bottlenecks.
So where do we even start? Pick the section that you find most exciting and let’s go!