# Chapter 10: Conclusion and References

Conclusion
We started this journey with a simple question: What does it actually take to train a high-performance LLM in 2026? To
answer that question, we’ve walked through the complete pipeline, from pretraining to post-training, showing you not just
the techniques we use but the methodology that makes them work.
We started by presenting the training compass, a framework for deciding whether to train at all, then showed you how to
translate your goals into concrete architectural decisions. You saw how to set up reliable ablation pipelines, test changes
individually, and scale from few-billion-token experiments to multi-trillion-token runs. We documented the infrastructure
challenges that can emerge at scale (throughput collapses, dataloader bottlenecks, subtle bugs) and how monitoring and
systematic derisking help you catch them early and debug quickly.
We also showed that going from a base model to a helpful assistant requires its own systematic approach: establishing
evals before training anything, iterating on SFT data mixtures, applying preference optimization, and optionally pushing
further with RL. You saw how vibe testing catches bugs that metrics miss, how chat templates can silently break instruction
following, and why the data mixture balance matters as much in post-training as it does in pretraining.
Throughout both phases—pretraining and post-training—we kept coming back to the same core insights: Validate everything
through experiments, change one thing at a time, expect scale to break things in new ways, and let your use case drive
decisions rather than chasing every new paper. Following this process is how we trained SmolLM3, a competitive 3B
multilingual reasoner with long context. Along the way, we learned a lot about what works, what breaks, and how to debug
when things go wrong. We’ve tried to document it all here, the successes and failures alike.
What’s Next?
This guide covers the fundamentals of modern LLM training, but the field is evolving rapidly. Here are some ways to go
deeper:
Run experiments yourself. Reading about ablations is useful; running your own teaches you what actually matters. Pick a
small model, set up evals, and start experimenting.
Read the source code. Training frameworks like Nanotron, TRL, and others are open source. Understanding their
implementations reveals details that papers gloss over.
Follow recent work. Papers on recent state-of-the-art models show where the field is heading. The references section
contains our curated list of impactful papers and resources.
We hope this guide helps you approach your next training project with clarity and confidence, whether you’re at a large lab
pushing the frontier or part of a small team solving a specific problem.
Now go train something. And when your loss spikes mysteriously at 2 a.m., remember: Every great model has debugging
stories behind it. May the force of open source and open science always be with you!
ACKNOWLEDGMENTS
We thank Guilherme, Hugo, and Mario for their valuable feedback, and Abubakar for his help with Trackio features.
The following is a curated list of papers, books, and blog posts that have informed us on our LLM training journey.
LLM ARCHITECTURE
Dense models: Llama 3, OLMo 2, MobileLLM

MoEs: DeepSeek-V2, DeepSeek-V3, Scaling Laws of Efficient MoEs
Hybrid models: MiniMax-01, Mamba2
OPTIMIZERS & TRAINING PARAMETERS
Muon is Scalable for LLM Training, Fantastic Pretraining Optimizers
Large Batch Training, DeepSeek LLM
DATA CURATION
Web: FineWeb & FineWeb-Edu, FineWeb2, DCLM
Code: The Stack v2, To Code or Not to Code
Math: DeepSeekMath, FineMath, MegaMath
Data mixtures: SmolLM2, Does Your Data Spark Joy?
SCALING LAWS
Kaplan, Chinchilla, Scaling Data-Constrained Language Models
POST-TRAINING
InstructGPT: OpenAI’s foundational paper to turn base models into helpful assistants. The precursor to ChatGPT and a
key step on humanity’s path up the Kardashev scale.
Llama 2 & 3: Extremely detailed tech reports from Meta on the training behind their Llama models (may they rest in
peace). They each contain many insights into human data collection, both for human preferences and for model
evaluation.
Secrets of RLHF in LLMs, Part I & II: These papers contain lots of goodies on the nuts and bolts of RLHF, specifically on
how to train strong reward models.
Direct Preference Optimization: The breakthrough paper from 2023 that convinced everyone to stop doing RL with LLMs.
DeepSeek-R1: The breakthrough paper from 2025 that convinced everyone to start doing RL with LLMs.
Understanding R1-Zero-Like Training (Dr. GRPO): One of the most important papers on understanding the baked-in
biases with GRPO and how to fix them.
DAPO: Bytedance shares many implementation details to unlock stable R1-Zero-like training for the community.
ScaleRL: A massive flex from Meta to derive scaling laws for RL. Burns over 400k GPU hours to establish a training
recipe that scales reliably over many orders of compute.
LoRA Without Regret: A beautifully written blog post which finds that RL with low-rank LoRA can match full fine-tuning (a
most surprising result).
Command A: A remarkably detailed tech report from Cohere on various strategies to post-train LLMs effectively.
INFRASTRUCTURE
Ultra-Scale Playbook

Jax scaling book
Modal GPU Glossary
TRAINING FRAMEWORKS
Megatron-LM
DeepSpeed
TorchTitan
Nanotron
NanoChat
TRL
EVALUATION
LLM Evaluation Guidebook
OLMES
FineTasks
Lessons from the Trenches
Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Ramos, S., Geist, M., & Bachem, O. (2024). On-Policy Distillation of
Language Models: Learning from Self-Generated Mistakes. https://arxiv.org/abs/2306.13649
Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-
Query Transformer Models from Multi-Head Checkpoints. https://arxiv.org/abs/2305.13245
Allal, L. B., Lozhkov, A., Bakouch, E., Blázquez, G. M., Penedo, G., Tunstall, L., Marafioti, A., Kydlí č ek, H., Lajarín, A. P.,
Srivastav, V., Lochner, J., Fahlgren, C., Nguyen, X.-S., Fourrier, C., Burtenshaw, B., Larcher, H., Zhao, H., Zakka, C., Morlon,
M., … Wolf, T. (2025). SmolLM2: When Smol Goes Big – Data-Centric Training of a Small Language Model.
https://arxiv.org/abs/2502.02737
Almazrouei, E., Alobeidli, H., Alshamsi, A., Cappelli, A., Cojocaru, R., Debbah, M., Goffinet, É., Hesslow, D., Launay, J.,
Malartic, Q., Mazzotta, D., Noune, B., Pannier, B., & Penedo, G. (2023). The Falcon Series of Open Language Models.
https://arxiv.org/abs/2311.16867
An, C., Huang, F., Zhang, J., Gong, S., Qiu, X., Zhou, C., & Kong, L. (2024). Training-Free Long-Context Scaling of Large
Language Models. https://arxiv.org/abs/2402.17463
Aryabumi, V., Su, Y., Ma, R., Morisot, A., Zhang, I., Locatelli, A., Fadaee, M., Üstün, A., & Hooker, S. (2024). To Code, or
Not To Code? Exploring Impact of Code in Pre-training. https://arxiv.org/abs/2408.10914
Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., Hui, B., Ji, L., Li, M., Lin, J., Lin, R.,
Liu, D., Liu, G., Lu, C., Lu, K., … Zhu, T. (2023). Qwen Technical Report. https://arxiv.org/abs/2309.16609
Barres, V., Dong, H., Ray, S., Si, X., & Narasimhan, K. (2025). τ2-Bench: Evaluating Conversational Agents in a Dual-Control
Environment. https://arxiv.org/abs/2506.07982
Beck, M., Pöppel, K., Lippe, P., & Hochreiter, S. (2025). Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM
Kernels. https://arxiv.org/abs/2503.14376
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A.,
Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., … Amodei,
D. (2020). Language Models are Few-Shot Learners. https://arxiv.org/abs/2005.14165
Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman,
G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., … Zaremba, W. (2021).
Evaluating Large Language Models Trained on Code. https://arxiv.org/abs/2107.03374

Chen, Y., Huang, B., Gao, Y., Wang, Z., Yang, J., & Ji, H. (2025a). Scaling Laws for Predicting Downstream Performance in
LLMs. https://arxiv.org/abs/2410.08527
Chen, Y., Huang, B., Gao, Y., Wang, Z., Yang, J., & Ji, H. (2025b). Scaling Laws for Predicting Downstream Performance in
LLMs. https://arxiv.org/abs/2410.08527
Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating long sequences with sparse transformers. arXiv Preprint
arXiv:1904.10509.
Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann,
S., Schuh, P., Shi, K., Tsvyashchenko, S., Maynez, J., Rao, A., Barnes, P., Tay, Y., Shazeer, N., Prabhakaran, V., … Fiedel,
N. (2022). PaLM: Scaling Language Modeling with Pathways. https://arxiv.org/abs/2204.02311
Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., Le, Q. V., Levine, S., & Ma, Y. (2025). SFT Memorizes, RL
Generalizes: A Comparative Study of Foundation Model Post-training. https://arxiv.org/abs/2501.17161
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse,
C., & Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. https://arxiv.org/abs/2110.14168
Cohere, T., :, Aakanksha, Ahmadian, A., Ahmed, M., Alammar, J., Alizadeh, M., Alnumay, Y., Althammer, S.,
Arkhangorodsky, A., Aryabumi, V., Aumiller, D., Avalos, R., Aviv, Z., Bae, S., Baji, S., Barbet, A., Bartolo, M., Bebensee, B.,
… Zhao, Z. (2025). Command A: An Enterprise-Ready Large Language Model. https://arxiv.org/abs/2504.00698
Dagan, G., Synnaeve, G., & Rozière, B. (2024). Getting the most out of your tokenizer for pre-training and domain
adaptation. https://arxiv.org/abs/2402.01035
Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State
Space Duality. https://arxiv.org/abs/2405.21060
DeepSeek-AI. (2025). DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention. DeepSeek.
https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
DeepSeek-AI, :, Bi, X., Chen, D., Chen, G., Chen, S., Dai, D., Deng, C., Ding, H., Dong, K., Du, Q., Fu, Z., Gao, H., Gao, K.,
Gao, W., Ge, R., Guan, K., Guo, D., Guo, J., … Zou, Y. (2024). DeepSeek LLM: Scaling Open-Source Language Models with
Longtermism. https://arxiv.org/abs/2401.02954
DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X.,
Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., … Zhang, Z. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in
LLMs via Reinforcement Learning. https://arxiv.org/abs/2501.12948
DeepSeek-AI, Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D., Yang, D., Chen,
D., Ji, D., Li, E., Lin, F., Luo, F., Hao, G., Chen, G., … Xie, Z. (2024). DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model. https://arxiv.org/abs/2405.04434
DeepSeek-AI, Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Guo, D.,
Yang, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., … Pan, Z. (2025). DeepSeek-V3 Technical Report.
https://arxiv.org/abs/2412.19437
Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A., Caron, M., Geirhos, R.,
Alabdulmohsin, I., Jenatton, R., Beyer, L., Tschannen, M., Arnab, A., Wang, X., Riquelme, C., Minderer, M., Puigcerver, J.,
Evci, U., … Houlsby, N. (2023). Scaling Vision Transformers to 22 Billion Parameters. https://arxiv.org/abs/2302.05442
Ding, H., Wang, Z., Paolini, G., Kumar, V., Deoras, A., Roth, D., & Soatto, S. (2024). Fewer Truncations Improve Language
Modeling. https://arxiv.org/abs/2404.10830
D’Oosterlinck, K., Xu, W., Develder, C., Demeester, T., Singh, A., Potts, C., Kiela, D., & Mehri, S. (2024). Anchored
Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment.
https://arxiv.org/abs/2408.06266
Du, Z., Zeng, A., Dong, Y., & Tang, J. (2025). Understanding Emergent Abilities of Language Models from the Loss
Perspective. https://arxiv.org/abs/2403.15796
Dubois, Y., Galambosi, B., Liang, P., & Hashimoto, T. B. (2025). Length-Controlled AlpacaEval: A Simple Way to Debias
Automatic Evaluators. https://arxiv.org/abs/2404.04475
Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., & Kiela, D. (2024). KTO: Model Alignment as Prospect Theoretic
Optimization. https://arxiv.org/abs/2402.01306
Gandhi, K., Chakravarthy, A., Singh, A., Lile, N., & Goodman, N. D. (2025). Cognitive Behaviors that Enable Self-Improving
Reasoners, or, Four Habits of Highly Effective STaRs. https://arxiv.org/abs/2503.01307

Gao, T., Wettig, A., Yen, H., & Chen, D. (2025). How to Train Long-Context Language Models (Effectively).
https://arxiv.org/abs/2410.02660
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A.,
Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., … Ma, Z. (2024).
The Llama 3 Herd of Models. https://arxiv.org/abs/2407.21783
Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
https://arxiv.org/abs/2312.00752
Gu, Y., Tafjord, O., Kuehl, B., Haddad, D., Dodge, J., & Hajishirzi, H. (2025). OLMES: A Standard for Language Model
Evaluations. https://arxiv.org/abs/2406.08446
Guo, S., Zhang, B., Liu, T., Liu, T., Khalman, M., Llinares, F., Rame, A., Mesnard, T., Zhao, Y., Piot, B., Ferret, J., & Blondel,
M. (2024). Direct Language Model Alignment from Online AI Feedback. https://arxiv.org/abs/2402.04792
Hägele, A., Bakouch, E., Kosson, A., Allal, L. B., Werra, L. V., & Jaggi, M. (2024). Scaling Laws and Compute-Optimal
Training Beyond Fixed Training Durations. https://arxiv.org/abs/2405.18392
He, Y., Jin, D., Wang, C., Bi, C., Mandyam, K., Zhang, H., Zhu, C., Li, N., Xu, T., Lv, H., Bhosale, S., Zhu, C., Sankararaman,
K. A., Helenowski, E., Kambadur, M., Tayade, A., Ma, H., Fang, H., & Wang, S. (2024). Multi-IF: Benchmarking LLMs on
Multi-Turn and Multilingual Instructions Following. https://arxiv.org/abs/2410.15553
Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl,
J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K.,
Elsen, E., … Sifre, L. (2022). Training Compute-Optimal Large Language Models. https://arxiv.org/abs/2203.15556
Hong, J., Lee, N., & Thorne, J. (2024). ORPO: Monolithic Preference Optimization without Reference Model.
https://arxiv.org/abs/2403.07691
Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification.
https://arxiv.org/abs/1801.06146
Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., & Ginsburg, B. (2024). RULER: What’s the Real
Context Size of Your Long-Context Language Models? https://arxiv.org/abs/2404.06654
Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhao, W., Zhang, X., Thai, Z. L., Zhang, K.,
Wang, C., Yao, Y., Zhao, C., Zhou, J., Cai, J., Zhai, Z., … Sun, M. (2024). MiniCPM: Unveiling the Potential of Small
Language Models with Scalable Training Strategies. https://arxiv.org/abs/2404.06395
Huang, S., Noukhovitch, M., Hosseini, A., Rasul, K., Wang, W., & Tunstall, L. (2024). The N+ Implementation Details of
RLHF with PPO: A Case Study on TL;DR Summarization. https://arxiv.org/abs/2403.17031
IBM Research. (2025). IBM Granite 4.0: Hyper-efficient, High Performance Hybrid Models for Enterprise.
https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample,
G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Scao, T. L., Lavril, T., Wang, T., Lacroix, T., & Sayed, W. E. (2023).
Mistral 7B. https://arxiv.org/abs/2310.06825
Kamradt, G. (2023). Needle In A Haystack - pressure testing LLMs. In GitHub repository. GitHub.
https://github.com/gkamradt/LLMTest_NeedleInAHaystack
Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D.
(2020). Scaling Laws for Neural Language Models. https://arxiv.org/abs/2001.08361
Katsch, T. (2024). GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling.
https://arxiv.org/abs/2311.01927
Kazemnejad, A., Padhi, I., Ramamurthy, K. N., Das, P., & Reddy, S. (2023). The Impact of Positional Encoding on Length
Generalization in Transformers. https://arxiv.org/abs/2305.19466
Khatri, D., Madaan, L., Tiwari, R., Bansal, R., Duvvuri, S. S., Zaheer, M., Dhillon, I. S., Brandfonbrener, D., & Agarwal, R.
(2025). The Art of Scaling Reinforcement Learning Compute for LLMs. https://arxiv.org/abs/2510.13786
Kingma, D. P. (2014). Adam: A method for stochastic optimization. arXiv Preprint arXiv:1412.6980.
Krajewski, J., Ludziejewski, J., Adamczewski, K., Pióro, M., Krutul, M., Antoniak, S., Ciebiera, K., Król, K., Odrzygó ź d ź , T.,
Sankowski, P., Cygan, M., & Jaszczur, S. (2024). Scaling Laws for Fine-Grained Mixture of Experts.
https://arxiv.org/abs/2402.07871

Lambert, N., Castricato, L., von Werra, L., & Havrilla, A. (2022). Illustrating Reinforcement Learning from Human Feedback
(RLHF). Hugging Face Blog.
Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N., Lyu, S., Gu, Y.,
Malik, S., Graf, V., Hwang, J. D., Yang, J., Bras, R. L., Tafjord, O., Wilhelm, C., Soldaini, L., … Hajishirzi, H. (2025). Tulu 3:
Pushing Frontiers in Open Language Model Post-Training. https://arxiv.org/abs/2411.15124
Lanchantin, J., Chen, A., Lan, J., Li, X., Saha, S., Wang, T., Xu, J., Yu, P., Yuan, W., Weston, J. E., Sukhbaatar, S., &
Kulikov, I. (2025). Bridging Offline and Online Reinforcement Learning for LLMs. https://arxiv.org/abs/2506.21495
Li, J., Fang, A., Smyrnis, G., Ivgi, M., Jordan, M., Gadre, S., Bansal, H., Guha, E., Keh, S., Arora, K., Garg, S., Xin, R.,
Muennighoff, N., Heckel, R., Mercat, J., Chen, M., Gururangan, S., Wortsman, M., Albalak, A., … Shankar, V. (2025).
DataComp-LM: In search of the next generation of training sets for language models. https://arxiv.org/abs/2406.11794
Li, Q., Cui, L., Zhao, X., Kong, L., & Bi, W. (2024). GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of
LLMs as Mathematical Problem Solvers. https://arxiv.org/abs/2402.19255
Li, R., Allal, L. B., Zi, Y., Muennighoff, N., Kocetkov, D., Mou, C., Marone, M., Akiki, C., Li, J., Chim, J., Liu, Q.,
Zheltonozhskii, E., Zhuo, T. Y., Wang, T., Dehaene, O., Davaadorj, M., Lamy-Poirier, J., Monteiro, J., Shliazhko, O., … de
Vries, H. (2023). StarCoder: may the source be with you! https://arxiv.org/abs/2305.06161
Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E., & Stoica, I. (2024). From Crowdsourced Data to
High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline. https://arxiv.org/abs/2406.11939
Liang, W., Liu, T., Wright, L., Constable, W., Gu, A., Huang, C.-C., Zhang, I., Feng, W., Huang, H., Wang, J., Purandare, S.,
Nadathur, G., & Idreos, S. (2025). TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training.
https://arxiv.org/abs/2410.06511
Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K.
(2023). Let’s Verify Step by Step. https://arxiv.org/abs/2305.20050
Liu, H., Xie, S. M., Li, Z., & Ma, T. (2022). Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language
Models. https://arxiv.org/abs/2210.14199
Liu, Q., Zheng, X., Muennighoff, N., Zeng, G., Dou, L., Pang, T., Jiang, J., & Lin, M. (2025). RegMix: Data Mixture as
Regression for Language Model Pre-training. https://arxiv.org/abs/2407.01492
Liu, Z., Zhao, C., Iandola, F., Lai, C., Tian, Y., Fedorov, I., Xiong, Y., Chang, E., Shi, Y., Krishnamoorthi, R., Lai, L., &
Chandra, V. (2024). MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.
https://arxiv.org/abs/2402.14905
Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts.
https://arxiv.org/abs/1608.03983
Lozhkov, A., Li, R., Allal, L. B., Cassano, F., Lamy-Poirier, J., Tazi, N., Tang, A., Pykhtar, D., Liu, J., Wei, Y., Liu, T., Tian, M.,
Kocetkov, D., Zucker, A., Belkada, Y., Wang, Z., Liu, Q., Abulkhanov, D., Paul, I., … de Vries, H. (2024). StarCoder 2 and
The Stack v2: The Next Generation. https://arxiv.org/abs/2402.19173
Marafioti, A., Zohar, O., Farré, M., Noyan, M., Bakouch, E., Cuenca, P., Zakka, C., Allal, L. B., Lozhkov, A., Tazi, N.,
Srivastav, V., Lochner, J., Larcher, H., Morlon, M., Tunstall, L., von Werra, L., & Wolf, T. (2025). SmolVLM: Redefining small
and efficient multimodal models. https://arxiv.org/abs/2504.05299
McCandlish, S., Kaplan, J., Amodei, D., & Team, O. D. (2018). An Empirical Model of Large-Batch Training.
https://arxiv.org/abs/1812.06162
Merrill, W., Arora, S., Groeneveld, D., & Hajishirzi, H. (2025). Critical Batch Size Revisited: A Simple Empirical Approach to
Large-Batch Language Model Training. https://arxiv.org/abs/2505.23971
Meta AI. (2025). The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation.
https://ai.meta.com/blog/llama-4-multimodal-intelligence/
Mindermann, S., Brauner, J., Razzak, M., Sharma, M., Kirsch, A., Xu, W., Höltgen, B., Gomez, A. N., Morisot, A., Farquhar,
S., & Gal, Y. (2022). Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt.
https://arxiv.org/abs/2206.07137
MiniMax, Li, A., Gong, B., Yang, B., Shan, B., Liu, C., Zhu, C., Zhang, C., Guo, C., Chen, D., Li, D., Jiao, E., Li, G., Zhang, G.,
Sun, H., Dong, H., Zhu, J., Zhuang, J., Song, J., … Wu, Z. (2025). MiniMax-01: Scaling Foundation Models with Lightning
Attention. https://arxiv.org/abs/2501.08313
Mistral AI. (2025). Mistral Small 3.1. https://mistral.ai/news/mistral-small-3-1

Moshkov, I., Hanley, D., Sorokin, I., Toshniwal, S., Henkel, C., Schifferer, B., Du, W., & Gitman, I. (2025). AIMO-2 Winning
Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset.
https://arxiv.org/abs/2504.16891
Muennighoff, N., Rush, A. M., Barak, B., Scao, T. L., Piktus, A., Tazi, N., Pyysalo, S., Wolf, T., & Raffel, C. (2025). Scaling
Data-Constrained Language Models. https://arxiv.org/abs/2305.16264
Ni, J., Xue, F., Yue, X., Deng, Y., Shah, M., Jain, K., Neubig, G., & You, Y. (2024). MixEval: Deriving Wisdom of the Crowd
from LLM Benchmark Mixtures. https://arxiv.org/abs/2406.06565
Nrusimha, A., Brandon, W., Mishra, M., Shen, Y., Panda, R., Ragan-Kelley, J., & Kim, Y. (2025). FlashFormer: Whole-Model
Kernels for Efficient Low-Batch Inference. https://arxiv.org/abs/2505.22758
Nvidia, :, Adler, B., Agarwal, N., Aithal, A., Anh, D. H., Bhattacharya, P., Brundyn, A., Casper, J., Catanzaro, B., Clay, S.,
Cohen, J., Das, S., Dattagupta, A., Delalleau, O., Derczynski, L., Dong, Y., Egert, D., Evans, E., … Zhu, C. (2024).
Nemotron-4 340B Technical Report. https://arxiv.org/abs/2406.11704
NVIDIA, :, Basant, A., Khairnar, A., Paithankar, A., Khattar, A., Renduchintala, A., Malte, A., Bercovich, A., Hazare, A., Rico,
A., Ficek, A., Kondratenko, A., Shaposhnikov, A., Bukharin, A., Taghibakhshi, A., Barton, A., Mahabaleshwarkar, A. S., Shen,
A., … Chen, Z. (2025). NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model.
https://arxiv.org/abs/2508.14444
NVIDIA, :, Blakeman, A., Basant, A., Khattar, A., Renduchintala, A., Bercovich, A., Ficek, A., Bjorlin, A., Taghibakhshi, A.,
Deshmukh, A. S., Mahabaleshwarkar, A. S., Tao, A., Shors, A., Aithal, A., Poojary, A., Dattagupta, A., Buddharaju, B., Chen,
B., … Chen, Z. (2025). Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models.
https://arxiv.org/abs/2504.03624
OLMo, T., Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., Bhagia, A., Gu, Y., Huang, S., Jordan, M., Lambert, N.,
Schwenk, D., Tafjord, O., Anderson, T., Atkinson, D., Brahman, F., Clark, C., Dasigi, P., Dziri, N., … Hajishirzi, H. (2025). 2
OLMo 2 Furious. https://arxiv.org/abs/2501.00656
OpenAI, Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S.,
Anadkat, S., Avila, R., Babuschkin, I., Balaji, S., Balcom, V., Baltescu, P., Bao, H., Bavarian, M., Belgum, J., … Zoph, B.
(2024). GPT-4 Technical Report. https://arxiv.org/abs/2303.08774
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A.,
Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R.
(2022). Training language models to follow instructions with human feedback. https://arxiv.org/abs/2203.02155
Penedo, G., Kydlí č ek, H., allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C., Werra, L. V., & Wolf, T. (2024). The FineWeb
Datasets: Decanting the Web for the Finest Text Data at Scale. https://arxiv.org/abs/2406.17557
Penedo, G., Kydlí č ek, H., Sabol č ec, V., Messmer, B., Foroutan, N., Kargaran, A. H., Raffel, C., Jaggi, M., Werra, L. V., &
Wolf, T. (2025). FineWeb2: One Pipeline to Scale Them All – Adapting Pre-Training Data Processing to Every Language.
https://arxiv.org/abs/2506.20920
Peng, B., Goldstein, D., Anthony, Q., Albalak, A., Alcaide, E., Biderman, S., Cheah, E., Du, X., Ferdinan, T., Hou, H.,
Kazienko, P., GV, K. K., Kocoń, J., Koptyra, B., Krishna, S., Jr., R. M., Lin, J., Muennighoff, N., Obeid, F., … Zhu, R.-J.
(2024). Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence. https://arxiv.org/abs/2404.05892
Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2023). YaRN: Efficient Context Window Extension of Large Language
Models. https://arxiv.org/abs/2309.00071
Peng, H., Pappas, N., Yogatama, D., Schwartz, R., Smith, N. A., & Kong, L. (2021). Random Feature Attention.
https://arxiv.org/abs/2103.02143
Petty, J., van Steenkiste, S., Dasgupta, I., Sha, F., Garrette, D., & Linzen, T. (2024). The Impact of Depth on Compositional
Generalization in Transformer Language Models. https://arxiv.org/abs/2310.19956
Polo, F. M., Weber, L., Choshen, L., Sun, Y., Xu, G., & Yurochkin, M. (2024). tinyBenchmarks: evaluating LLMs with fewer
examples. https://arxiv.org/abs/2402.14992
Press, O., Smith, N. A., & Lewis, M. (2022). Train Short, Test Long: Attention with Linear Biases Enables Input Length
Extrapolation. https://arxiv.org/abs/2108.12409
Pyatkin, V., Malik, S., Graf, V., Ivison, H., Huang, S., Dasigi, P., Lambert, N., & Hajishirzi, H. (2025). Generalizing Verifiable
Instruction Following. https://arxiv.org/abs/2507.02833

Qin, Z., Han, X., Sun, W., Li, D., Kong, L., Barnes, N., & Zhong, Y. (2022). The Devil in Linear Transformer.
https://arxiv.org/abs/2210.10340
Qin, Z., Yang, S., Sun, W., Shen, X., Li, D., Sun, W., & Zhong, Y. (2024). HGRN2: Gated Linear RNNs with State Expansion.
https://arxiv.org/abs/2404.07904
Qiu, Z., Huang, Z., Zheng, B., Wen, K., Wang, Z., Men, R., Titov, I., Liu, D., Zhou, J., & Lin, J. (2025). Demons in the Detail:
On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models.
https://arxiv.org/abs/2501.11873
Qwen Team. (2025). Qwen3-Next: Towards Ultimate Training & Inference Efficiency. Alibaba Cloud. https://qwen.ai/blog?
id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., & others. (2019). Language models are unsupervised
multitask learners. In OpenAI blog (Vol. 1, p. 9).
Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2024). Direct Preference Optimization: Your
Language Model is Secretly a Reward Model. https://arxiv.org/abs/2305.18290
Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., & Bowman, S. R. (2024). Gpqa: A
graduate-level google-proof q&a benchmark. First Conference on Language Modeling.
Rozière, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., Adi, Y., Liu, J., Sauvestre, R., Remez, T., Rapin, J.,
Kozhevnikov, A., Evtimov, I., Bitton, J., Bhatt, M., Ferrer, C. C., Grattafiori, A., Xiong, W., Défossez, A., … Synnaeve, G.
(2024). Code Llama: Open Foundation Models for Code. https://arxiv.org/abs/2308.12950
Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units.
https://arxiv.org/abs/1508.07909
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024).
DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.
https://arxiv.org/abs/2402.03300
Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. https://arxiv.org/abs/1911.02150
Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivats, S., Vosoughi, S., Chung, H. W., Tay, Y., Ruder, S., Zhou, D., Das, D., &
Wei, J. (2022). Language Models are Multilingual Chain-of-Thought Reasoners. https://arxiv.org/abs/2210.03057
Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M.,
Marafioti, A., Alibert, S., Cord, M., Wolf, T., & Cadene, R. (2025). SmolVLA: A Vision-Language-Action Model for Affordable
and Efficient Robotics. https://arxiv.org/abs/2506.01844
Singh, S., Romanou, A., Fourrier, C., Adelani, D. I., Ngui, J. G., Vila-Suero, D., Limkonchotiwat, P., Marchisio, K., Leong, W.
Q., Susanto, Y., Ng, R., Longpre, S., Ko, W.-Y., Ruder, S., Smith, M., Bosselut, A., Oh, A., Martins, A. F. T., Choshen, L., …
Hooker, S. (2025). Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation.
https://arxiv.org/abs/2412.03304
Sirdeshmukh, V., Deshpande, K., Mols, J., Jin, L., Cardona, E.-Y., Lee, D., Kritz, J., Primack, W., Yue, S., & Xing, C. (2025).
MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs.
https://arxiv.org/abs/2501.17399
Smith, L. N., & Topin, N. (2018). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.
https://arxiv.org/abs/1708.07120
Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2023). RoFormer: Enhanced Transformer with Rotary Position
Embedding. https://arxiv.org/abs/2104.09864
Sun, Y., Dong, L., Zhu, Y., Huang, S., Wang, W., Ma, S., Zhang, Q., Wang, J., & Wei, F. (2024). You Only Cache Once:
Decoder-Decoder Architectures for Language Models. https://arxiv.org/abs/2405.05254
Takase, S., Kiyono, S., Kobayashi, S., & Suzuki, J. (2025). Spike No More: Stabilizing the Pre-training of Large Language
Models. https://arxiv.org/abs/2312.16903
Team, 5, Zeng, A., Lv, X., Zheng, Q., Hou, Z., Chen, B., Xie, C., Wang, C., Yin, D., Zeng, H., Zhang, J., Wang, K., Zhong, L.,
Liu, M., Lu, R., Cao, S., Zhang, X., Huang, X., Wei, Y., … Tang, J. (2025). GLM-4.5: Agentic, Reasoning, and Coding (ARC)
Foundation Models. https://arxiv.org/abs/2508.06471
team, F. C., Copet, J., Carbonneaux, Q., Cohen, G., Gehring, J., Kahn, J., Kossen, J., Kreuk, F., McMilin, E., Meyer, M., Wei,
Y., Zhang, D., Zheng, K., Armengol-Estapé, J., Bashiri, P., Beck, M., Chambon, P., Charnalia, A., Cummins, C., … Synnaeve,

G. (2025). CWM: An Open-Weights LLM for Research on Code Generation with World Models.
https://arxiv.org/abs/2510.02387
Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ramé, A., Rivière, M.,
Rouillard, L., Mesnard, T., Cideron, G., bastien Jean-Grill, Ramos, S., Yvinec, E., Casbon, M., Pot, E., Penchev, I., …
Hussenot, L. (2025). Gemma 3 Technical Report. https://arxiv.org/abs/2503.19786
Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., Chen, Z., Cui, J., Ding, H.,
Dong, M., Du, A., Du, C., Du, D., Du, Y., Fan, Y., … Zu, X. (2025). Kimi K2: Open Agentic Intelligence.
https://arxiv.org/abs/2507.20534
Team, L., Zeng, B., Huang, C., Zhang, C., Tian, C., Chen, C., Jin, D., Yu, F., Zhu, F., Yuan, F., Wang, F., Wang, G., Zhai, G.,
Zhang, H., Li, H., Zhou, J., Liu, J., Fang, J., Ou, J., … He, Z. (2025). Every FLOP Counts: Scaling a 300B Mixture-of-Experts
LING LLM without Premium GPUs. https://arxiv.org/abs/2503.05139
Team, M., Xiao, C., Li, Y., Han, X., Bai, Y., Cai, J., Chen, H., Chen, W., Cong, X., Cui, G., Ding, N., Fan, S., Fang, Y., Fu, Z.,
Guan, W., Guan, Y., Guo, J., Han, Y., He, B., … Sun, M. (2025). MiniCPM4: Ultra-Efficient LLMs on End Devices.
https://arxiv.org/abs/2506.07900
Tian, C., Chen, K., Liu, J., Liu, Z., Zhang, Z., & Zhou, J. (2025). Towards Greater Leverage: Scaling Laws for Efficient
Mixture-of-Experts Language Models. https://arxiv.org/abs/2507.17702
Toshniwal, S., Moshkov, I., Narenthiran, S., Gitman, D., Jia, F., & Gitman, I. (2024). OpenMathInstruct-1: A 1.8 Million Math
Instruction Tuning Dataset. https://arxiv.org/abs/2402.10176
Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., von Werra, L., Fourrier, C., Habib, N.,
Sarrazin, N., Sanseviero, O., Rush, A. M., & Wolf, T. (2023). Zephyr: Direct Distillation of LM Alignment.
https://arxiv.org/abs/2310.16944
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). Attention Is
All You Need. https://arxiv.org/abs/1706.03762
Waleffe, R., Byeon, W., Riach, D., Norick, B., Korthikanti, V., Dao, T., Gu, A., Hatamizadeh, A., Singh, S., Narayanan, D.,
Kulshreshtha, G., Singh, V., Casper, J., Kautz, J., Shoeybi, M., & Catanzaro, B. (2024). An Empirical Study of Mamba-based
Language Models. https://arxiv.org/abs/2406.07887
Wang, B., & Komatsuzaki, A. (2021). GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model.
https://github.com/kingoflolz/mesh-transformer-jax
Wei, J., Karina, N., Chung, H. W., Jiao, Y. J., Papay, S., Glaese, A., Schulman, J., & Fedus, W. (2024). Measuring short-form
factuality in large language models. arXiv Preprint arXiv:2411.04368.
Wen, K., Hall, D., Ma, T., & Liang, P. (2025). Fantastic Pretraining Optimizers and Where to Find Them.
https://arxiv.org/abs/2509.02046
Xie, S. M., Pham, H., Dong, X., Du, N., Liu, H., Lu, Y., Liang, P., Le, Q. V., Ma, T., & Yu, A. W. (2023). DoReMi: Optimizing
Data Mixtures Speeds Up Language Model Pretraining. https://arxiv.org/abs/2305.10429
Xiong, W., Liu, J., Molybog, I., Zhang, H., Bhargava, P., Hou, R., Martin, L., Rungta, R., Sankararaman, K. A., Oguz, B.,
Khabsa, M., Fang, H., Mehdad, Y., Narang, S., Malik, K., Fan, A., Bhosale, S., Edunov, S., Lewis, M., … Ma, H. (2023a).
Effective Long-Context Scaling of Foundation Models. https://arxiv.org/abs/2309.16039
Xiong, W., Liu, J., Molybog, I., Zhang, H., Bhargava, P., Hou, R., Martin, L., Rungta, R., Sankararaman, K. A., Oguz, B.,
Khabsa, M., Fang, H., Mehdad, Y., Narang, S., Malik, K., Fan, A., Bhosale, S., Edunov, S., Lewis, M., … Ma, H. (2023b).
Effective Long-Context Scaling of Foundation Models. https://arxiv.org/abs/2309.16039
Xu, H., Peng, B., Awadalla, H., Chen, D., Chen, Y.-C., Gao, M., Kim, Y. J., Li, Y., Ren, L., Shen, Y., Wang, S., Xu, W., Gao, J.,
& Chen, W. (2025). Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math.
https://arxiv.org/abs/2504.21233
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F.,
Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., … Qiu, Z. (2025). Qwen3 Technical Report.
https://arxiv.org/abs/2505.09388
Yang, A., Yu, B., Li, C., Liu, D., Huang, F., Huang, H., Jiang, J., Tu, J., Zhang, J., Zhou, J., Lin, J., Dang, K., Yang, K., Yu, L.,
Li, M., Sun, M., Zhu, Q., Men, R., He, T., … Zhang, Z. (2025). Qwen2.5-1M Technical Report.
https://arxiv.org/abs/2501.15383

Yang, B., Venkitesh, B., Talupuru, D., Lin, H., Cairuz, D., Blunsom, P., & Locatelli, A. (2025). Rope to Nope and Back Again:
A New Hybrid Attention Strategy. https://arxiv.org/abs/2501.18795
Yang, G., & Hu, E. J. (2022). Feature Learning in Infinite-Width Neural Networks. https://arxiv.org/abs/2011.14522
Yen, H., Gao, T., Hou, M., Ding, K., Fleischer, D., Izsak, P., Wasserblat, M., & Chen, D. (2025). HELMET: How to Evaluate
Long-Context Language Models Effectively and Thoroughly. https://arxiv.org/abs/2410.02694
Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., Liu, X., Lin, H., Lin, Z., Ma, B., Sheng,
G., Tong, Y., Zhang, C., Zhang, M., Zhang, W., … Wang, M. (2025). DAPO: An Open-Source LLM Reinforcement Learning
System at Scale. https://arxiv.org/abs/2503.14476
Yuan, J., Gao, H., Dai, D., Luo, J., Zhao, L., Zhang, Z., Xie, Z., Wei, Y. X., Wang, L., Xiao, Z., Wang, Y., Ruan, C., Zhang, M.,
Liang, W., & Zeng, W. (2025). Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.
https://arxiv.org/abs/2502.11089
Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Yue, Y., Song, S., & Huang, G. (2025). Does Reinforcement Learning Really
Incentivize Reasoning Capacity in LLMs Beyond the Base Model? https://arxiv.org/abs/2504.13837
Zhao, Y., Qu, Y., Staniszewski, K., Tworkowski, S., Liu, W., Miło ś , P., Wu, Y., & Minervini, P. (2024). Analysing The Impact
of Sequence Composition on Language Model Pre-Training. Proceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), 7897–7912. https://doi.org/10.18653/v1/2024.acl-long.427
Zhou, F., Wang, Z., Ranjan, N., Cheng, Z., Tang, L., He, G., Liu, Z., & Xing, E. P. (2025). MegaMath: Pushing the Limits of
Open Math Corpora. https://arxiv.org/abs/2504.02807
Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., & Hou, L. (2023). Instruction-Following Evaluation for
Large Language Models. https://arxiv.org/abs/2311.07911
Zhu, T., Liu, Q., Wang, H., Chen, S., Gu, X., Pang, T., & Kan, M.-Y. (2025). SkyLadder: Better and Faster Pretraining via
Context Window Scheduling. https://arxiv.org/abs/2503.15450
Zuo, J., Velikanov, M., Chahed, I., Belkada, Y., Rhayem, D. E., Kunsch, G., Hacid, H., Yous, H., Farhat, B., Khadraoui, I.,
Farooq, M., Campesan, G., Cojocaru, R., Djilali, Y., Hu, S., Chaabane, I., Khanna, P., Seddik, M. E. A., Huynh, N. D., …
Frikha, S. (2025). Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance.
https://arxiv.org/abs/2507.22448
Citation For attribution in academic contexts, please cite this work
as
Loubna Ben Allal, Lewis Tunstall, Nouamane Tazi, Elie Bakouch, Ed Beeching, Carlos Miguel Patiño, Clémentine Fourrier, Thibaud Frere, Anton Lozhkov, Colin Raffel, Leandro von Werra, Thomas Wolf (2025). "The Smol Training Playbook: The Secrets to Building World-Class LLMs".
BibTeX citation


@misc{allal2025_the_smol_training_playbook_the_secrets_to_building_world_class_llms,  title={The Smol Training Playbook: The Secrets to Building World-Class LLMs},  author={Loubna Ben Allal and Lewis Tunstall and Nouamane Tazi and Elie Bakouch and Ed Beeching and Carlos Miguel Patiño and Clémentine Fourrier and Thibaud Frere and Anton Lozhkov and Colin Raffel and Leandro von Werra and Thomas Wolf},  year={2025},  }
References
Footnotes 1. Benchmaxxing refers to the practice of training a model
to perform well on a narrow set of public benchmarks,
at the expense of performing well on real-world tasks.
2. For vLLM, see: reasoning parsers, tool parsers. For
SGLang, see: reasoning parsers, tool parsers. 
3. The idea to compute these statistics comes from the
Llama 3 tech report (Grattafiori et al., 2024). 
4. The Transformers team has recently added parsers for
extracting tool calling and reasoning outputs. If these
are adopted by engines like vLLM, the compatibility
criterion may become less important in the future. 
Made with ❤  with research article template