> Do some research into the article I posted like related ideas and papers, other research directions, ways to improve this etc.

Here’s the research map I’d use around the Percepta post.

## Bottom line

My read is that the post sits at the intersection of **four existing literatures**: formal results that transformers can simulate computation, compile-to-transformer work, neural algorithmic reasoning / trace-supervised execution, and efficient-attention systems for long-horizon inference. The public artifact I found is the March 11, 2026 Percepta blog post itself, so I’d treat the strongest empirical claims as promising but not yet as settled as a paper-plus-code release would make them. ([Percepta][1])

## Where the article fits in the literature

The basic premise that “transformers can be computers” is not new. Pérez, Marinković, and Barceló showed in 2019 that Transformers are Turing complete; Universal Transformers added a recurrent inductive bias and were shown Turing-complete under assumptions; more recent work showed constant bit-size transformers are Turing complete, autoregressive LLM decoding can realize universal computation, and a 2026 theory result reduced the efficiency gap with sparse attention and fixed geometric offsets. So the article’s core philosophical claim is well aligned with current theory; what matters is whether it turns those existence results into a practical, fast executor. ([arXiv][2])

That practical gap is real. Recent graph-algorithm theory papers argue that different reasoning problems require very different depth/width/extra-token regimes, and Neural Execution Engines explicitly found that transformer-style models can fit short algorithmic tasks but lose fidelity as length grows, with binary encodings and conditional masking helping generalization. In other words: expressiveness is not the same thing as trainability or stable long-horizon execution. ([arXiv][3])

A second nearby line is **programs-into-weights**. RASP gave a programming-language view of transformer computation; Tracr compiled human-readable RASP programs into decoder-only transformer weights; Learning Transformer Programs trained interpretable-by-design transformers that can be converted back into discrete programs; and ALTA pushed this further with loops and compilation to Universal Transformers. So “compile symbolic procedures into transformer weights” is already a live research direction. Percepta’s distinct claim is not that compilation exists, but that a low-level interpreter target plus a special decoding path make it useful for long exact execution. ([arXiv][4])

There is also a much older **neural computer** lineage. Neural Turing Machines coupled networks to differentiable external memory; Neural GPU learned algorithms with a highly parallel architecture; and differentiable Forth let researchers write program sketches with learned slots. Those systems were trying to make neural nets act more like computers long before current LLMs. The modern successor is the neural algorithmic reasoning (NAR) literature: Neural Execution of Graph Algorithms, the CLRS benchmark, and the broader NAR program all focus on training models to imitate the internal steps of classical algorithms rather than only their final outputs. ([arXiv][5])

That NAR line has already branched into directions directly relevant to Percepta-style work. Some papers show you can stay competitive even **without full intermediate supervision**, as long as you regularize internal computation well; others show that **dual formulations** help on optimization problems; and recurrent models can sometimes learn on easy cases and solve harder ones by “thinking for longer” at test time. On the code side, CodeExecutor showed that explicit execution-trace pretraining and curriculum improve a model’s grasp of program execution semantics. For Hungarian-style workloads, the dual-reasoning papers are especially interesting: they suggest supervising reduced costs, potentials, or primal-dual structure, not just emitted tokens. ([arXiv][6])

Finally, the “exponentially faster inference” part belongs in the **efficient attention** family. The current literature mostly groups this area into hardware-efficient, sparse, compact-KV, and linear-attention methods. Linear Transformers reduce quadratic attention to linear-time recurrence; top-k attention computes only the largest similarities per query; recurrent-memory models reduce flat-prefix dependence by carrying learned memory across segments. From that perspective, Percepta’s hull-based 2D hard-max path looks less like a general replacement for softmax attention and more like a highly specialized sparse/data-structure trick for a restricted executor regime. ([arXiv][7])

## What seems genuinely new

From the public sources I checked, three ingredients are clearly **not** new on their own: transformer universality, compile-to-transformer weights, and training on execution or algorithm traces. The unusual part appears to be the **combination**: a low-level WebAssembly-style interpreter target, explicit execution traces, and a 2D-head geometric retrieval path intended to make exact in-model execution cheap enough to matter on CPU. That looks like a potentially novel systems synthesis, but it is also the part that most needs a public paper, code, and ablations before I’d treat it as established. ([Percepta][1])

## What I would want to validate

I would want exactness-vs-trace-length curves, branch-heavy and memory-heavy workloads, ablations on 2D heads versus larger heads, hard-max versus softmax/top-k variants, hull versus linear versus other sparse baselines, and a clean separation between **compiled/proof-backed** behavior and **learned** behavior. The efficient-attention survey is clear that most speedups involve quality tradeoffs, and the graph-reasoning literature is clear that different algorithmic problems fall into very different capability regimes. That makes careful workload-by-workload evidence essential. ([Attention Survey][8])

## Best research directions from here

First, I would keep **two lanes** separate: a compiled/proof-backed lane and a learned lane. Tracr and ALTA show that program-to-weights compilation is its own discipline; CLRS/NAR show that learned algorithm execution is another. Mixing them makes claims muddy, while keeping them separate lets you say “this bounded executor is exact because it is compiled” versus “this one generalizes because it was learned.” ([arXiv][9])

Second, I would supervise **more structure than just next token**. Neural Execution Engines, Dual Algorithmic Reasoning, and CodeExecutor all point the same way: richer internal targets help. For a Wasm-like executor, that means supervising instruction pointer, stack delta, memory diff, branch outcome, and, for optimization tasks, primal-dual state such as Hungarian potentials and slack structure. ([arXiv][10])

Third, I would not assume textbook sequential traces are the right target. “Parallel Algorithms Align With Neural Execution” is one of the most important papers here: it argues that neural reasoners are parallel processors, and that teaching them sequential algorithms wastes capacity. For Hungarian, shortest path, flow, and similar workloads, parallel or wavefront formulations may fit the model better than a literal CPU-style trace, even if you still keep a compiled sequential reference lane for truth. ([Proceedings of Machine Learning Research][11])

Fourth, I would explore **recurrent or iterative architectures** alongside flat-prefix transformers. Universal Transformers, Recurrent Memory Transformer, and “easy-to-hard” recurrent work all suggest that long-horizon computation often benefits from recurrence, dynamic halting, or explicit memory passing, rather than forcing everything through an ever-growing prefix. If the problem is “exact execution for millions of steps,” that is a strong hint to test recurrence seriously. ([arXiv][12])

Fifth, I would compare any specialized fast path against **trainable sparse/hybrid alternatives**, not just a naive linear scan. Top-k attention, compact-KV methods, recurrent-memory models, and hybrid sparse/linear methods are all serious baselines. Even if hull-based retrieval wins in the hard-max 2D regime, you want to know whether it is winning because of the geometry, because the task is unusually low-entropy, or because the model has already been heavily specialized into that regime. ([ACL Anthology][13])

Sixth, I would broaden evaluation beyond demos. CLRS is the obvious benchmark family for algorithm coverage; CodeExecutor is useful for execution-trace learning on code; Sudoku-Bench is useful if you care about unaided reasoning on hard Sudoku-style problems, but it measures a different thing from exact compiled execution. Together they separate exact execution, length generalization, and creative reasoning instead of collapsing them into one headline number. ([arXiv][14])

## Best papers to read first

* **On the Turing Completeness of Modern Neural Network Architectures** — foundational “transformers can compute” theory. ([arXiv][2])
* **Universal Transformers** — adds recurrence/dynamic halting and is much closer to “iterative computation inside transformer-like models.” ([arXiv][12])
* **Thinking Like Transformers** — RASP as a programming model for transformer computation. ([arXiv][4])
* **Tracr** — concrete compiler from RASP programs to transformer weights. ([arXiv][9])
* **ALTA** — extends compile-to-transformer work with loops and Universal Transformers. ([arXiv][15])
* **Learning Transformer Programs** — trains interpretable program-like transformers rather than only compiling them. ([arXiv][16])
* **Neural Execution Engines** — very relevant empirical lessons on why sequence models fail to generalize on long algorithmic tasks, and what helps. ([arXiv][10])
* **Neural Execution of Graph Algorithms** + **CLRS** — the modern benchmark and method culture for neural algorithmic reasoning. ([arXiv][17])
* **Parallel Algorithms Align With Neural Execution** — probably the most important “don’t just imitate textbook sequential traces” paper here. ([Proceedings of Machine Learning Research][11])
* **Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count** — useful because it shows concrete tricks like scratchpads and positional coupling that materially improve extrapolation. ([arXiv][18])
* **Transformers are RNNs** + the **efficient attention survey** — best entry point for thinking about fast decoding baselines and tradeoffs. ([arXiv][7])

The most important meta-point is that this area is no longer one question. It is really three: **can transformers represent computation, can they learn or compile it reliably, and can they execute it efficiently enough to matter?** Percepta’s article is interesting because it tries to answer all three at once. The literature says those are separable problems, and progress usually comes fastest when you keep them separated in your experiments and claims. ([arXiv][2])

A good next step would be a Psionic/Tassadar-specific memo that turns this into an architecture plan, benchmark plan, and reading order.

[1]: https://www.percepta.ai/blog/can-llms-be-computers "https://www.percepta.ai/blog/can-llms-be-computers"
[2]: https://arxiv.org/abs/1901.03429 "https://arxiv.org/abs/1901.03429"
[3]: https://arxiv.org/abs/2405.18512 "https://arxiv.org/abs/2405.18512"
[4]: https://arxiv.org/abs/2106.06981 "https://arxiv.org/abs/2106.06981"
[5]: https://arxiv.org/abs/1410.5401 "https://arxiv.org/abs/1410.5401"
[6]: https://arxiv.org/abs/2306.13411 "https://arxiv.org/abs/2306.13411"
[7]: https://arxiv.org/abs/2006.16236 "https://arxiv.org/abs/2006.16236"
[8]: https://attention-survey.github.io/files/Attention_Survey.pdf "https://attention-survey.github.io/files/Attention_Survey.pdf"
[9]: https://arxiv.org/pdf/2301.05062 "https://arxiv.org/pdf/2301.05062"
[10]: https://arxiv.org/abs/2006.08084 "https://arxiv.org/abs/2006.08084"
[11]: https://proceedings.mlr.press/v231/engelmayer24a.html "https://proceedings.mlr.press/v231/engelmayer24a.html"
[12]: https://arxiv.org/abs/1807.03819 "https://arxiv.org/abs/1807.03819"
[13]: https://aclanthology.org/2021.sustainlp-1.5.pdf "https://aclanthology.org/2021.sustainlp-1.5.pdf"
[14]: https://arxiv.org/abs/2205.15659 "https://arxiv.org/abs/2205.15659"
[15]: https://arxiv.org/abs/2410.18077 "https://arxiv.org/abs/2410.18077"
[16]: https://arxiv.org/abs/2306.01128 "https://arxiv.org/abs/2306.01128"
[17]: https://arxiv.org/abs/1910.10593 "https://arxiv.org/abs/1910.10593"
[18]: https://arxiv.org/abs/2410.15787 "https://arxiv.org/abs/2410.15787"

