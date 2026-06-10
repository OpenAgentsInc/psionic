# Chapter 5: Optimizer and Training Hyperparameters

Optimizer and Training Hyperparameters
The pieces are coming into place. We’ve run our ablations, settled on the architecture, and chosen a tokenizer. But before
we can actually launch the training, there are still some crucial decisions to make: Which optimizer should we use? What
learning rate and batch size? How should we schedule the learning rate over training?
The tempting approach here is to just borrow values from another strong model in the literature. After all, if it worked for big
labs, it should work for us, right? And in many cases that approach will work just fine, if we’re taking values from a similar
architecture and model size.
TL;DR: Your use case drives your choices.

However, we risk leaving performance on the table by not tuning these values for our specific setup. Hyperparameters
reported in the literature will have been optimized for specific data and constraints, and sometimes those constraints aren’t
even about performance. Maybe the learning rate was picked early in development and never revisited. Even when model
authors do thorough hyperparameter sweeps, those optimal values were found for their exact combination of architecture,
data, and training regime, not ours. Literature values are always a good starting point, but it’s a good idea to explore
whether we can find better values in the neighborhood.
In this section, we’ll explore the latest optimizers (and see if trusty old AdamW (Kingma, 2014) still stands the test of time),
dive into learning rate schedules that go beyond the standard cosine decay, and figure out how to tune the learning rate and
batch size given a model and data size.
Let’s start with the optimizer wars.
OPTIMIZERS: ADAMW AND BEYOND
We didn’t spare any effort to summarize the current landscape of optimizers used for LLM pretraining:
Model Optimizer
Kimi K2, GLM 4.5 Muon
Everyone else AdamW
So, you might wonder, why is everyone using AdamW?
The person writing this part of the guide thinks it’s because “people are lazy” (hi, it’s Elie), but others might more
realistically say that AdamW has been working well/better than the competition at different scales for a long time, and it’s
always a bit scary to change such a core component—especially if it’s hard (i.e., expensive) to test how well it does in very
long training runs.
So, let’s start with the classic, and the foundation of Durk Kingma’s scary Google Scholar domination: AdamW.
AdamW
Adam (Adaptive Momentum Estimation) is a first-order optimization technique, which means it looks only at the gradients. It
adapts the learning rate for each parameter using momentum derived from past gradients.
The careful reader might wonder: Hey there, aren’t you missing a W? Indeed! We specifically add the W (weight decay)
because whereas in standard stochastic gradient descent (SGD) we can simply add a  (where  are the weights) to the
loss to apply L2 regularization, if we do the same with Adam, the adaptive learning rate will also affect the L2 regularization.
This means the regularization strength becomes dependent on gradient magnitudes, weakening its effect. This is not what
we want; AdamW applies it decoupled from the main optimization loop to fix this issue.
Interestingly, over the last few years the AdamW hyperparameters have barely moved:
β₁ = 0.9, β₂ = 0.95
The optimizer is at the heart of the whole LLM training operation. It decides for every parameter what the actual update step
will be, based on the past updates, the current weights, and the gradients derived from the loss. At the same time, it is
also a memory- and compute-hungry beast and thus can impact how many GPUs you need and how fast your training is.
Moreover, comparing optimizers fairly is harder than it looks. Scale changes the dynamics in ways that can be hard to
simulate in small ablations, so hyperparameter tuning is complex. You could say, “It’s OK, I’ve tuned my AdamW for weeks, I
can just reuse the same hyperparameters to compare!” We wish so much this were true. But unfortunately, for each
optimizer, you need to do a proper hyperparameter search (1D? 2D? 3D?), which makes optimizer research hard and costly.
λθ2 θ

Grad norm clipping = 1.0
Weight decay = 0.1 (Llama 3 405B drops this to 0.01)
The same triplet is reused in Llama 1, 2, and 3 and DeepSeek-V1, V2, and V3-671B, with no changes. Was Durk Kingma
right all along, or can we do better?
Muon in One Line
Adam is a first-order method, as it only uses the gradients. Muon is a second-order optimizer that acts on the matrix view of
a parameter tensor:
1. Matrix-wise geometry versus parameter-wise updates: AdamW preconditions per parameter (diagonal second moment).
Muon treats each weight matrix as a single object and updates along  , which captures row/column
subspace structure.
2. Isotropic steps via orthogonalization: Decomposing  with singular value decomposition (SVD) separates
magnitude (  ) from directions (the left/right subspaces  ). Replacing  by  discards singular values and
makes the step isotropic in the active subspaces. It’s a bit counterintuitive at first, since throwing away  looks like
losing information, but it reduces axis-aligned bias and encourages exploration of directions that would otherwise be
suppressed by very small singular values. Whether this kind of exploration bakes different capabilities into the model
that aren’t obvious if you only look at the loss is still an open question.
3. Empirical tolerance to larger batch sizes: In practice, Muon often tolerates higher batch sizes. We’ll talk about this in
more depth in the batch size section, but this might be a key motivator for Muon adoption!
For years, the community mostly settled on AdamW, and the optimizer recipes of frontier labs are often kept secret (Qwen
don’t talk about theirs, for instance), but recently Muon has seen uptake in high-profile releases (e.g., Kimi K2 and GLM-
4.5). Hopefully we’ll see more open and robust recipes to use it in the future.
There is a wild zoo of optimizers, and the only thing researchers are more creative at than combining possible momentums
and derivates is coming up with names for them: Shampoo, SOAP, PSGD, CASPR, DION, Sophia, Lion… even AdamW has
its own variants, like NAdamW, StableAdamW, and so on. Diving into all these optimizers would be worth its own guide, but
we’ll keep that for another time. In the meantime, we recommend the amazing paper by Stanford’s Marin team (Wen et al.,
2025), who benchmarked many different optimizers to show how important hyperparameter tuning is when doing
comparisons.
A question that goes hand in hand with almost every optimizer choice is how strong the weight update should be. This is
determined by the learning rate, which typically appears as a scalar in the optimizer equations. This is a seemingly simple
topic, but as you’ll see in the next section, there are many facets to it.
LEARNING RATE
The learning rate (LR) is one of the most important hyperparameters we have to set. At each training step, it controls how
much we adjust our model weights based on the computed gradients. Choosing a learning rate that’s too low will make
training painfully slow, and we risk getting trapped in a bad local minimum. The loss curves will look flat, and we’ll burn
  
G 
t
B 
t
O 
t
θ 
t
=∇ L (θ  )θ t t−1
=μB  +G 
t−1 t
=NewtonSchulz5(B ) ≈ UV if B =UΣV  (SVD)t ⊤ t ⊤
=θ  −ηO 
t−1 t
Looking at these equations, you might wonder why is this a second-order method—I only see gradients, and no higher order
terms! The second-order optimization actually happens inside the Newton Schulz step, but we won’t go into further detail on
that here. There are plenty of high-quality resources that explain Muon in depth, so we’ll just cover the three key ideas:
G=UV⊤
G=UΣV⊤
Σ U,V G UV⊤
Σ

through our compute budget without making meaningful progress. On the other hand, setting the learning rate too high
causes the optimizer to take massive steps that overshoot optimal solutions and never converge (or the unimaginable may
happen: the loss diverges and shoots to the moon).
But the best learning rate isn’t even constant, since the learning dynamics change during training. High learning rates work
early, when we’re far from good solutions, but cause instability near convergence. This is where learning rate schedules
come in: Warm up from zero to avoid early chaos, then decay to settle into a good minimum. These patterns (warmup +
cosine decay, for example) have been validated for neural network training over many years.
💡Warmup steps
Most modern LLMs use a fixed number of warmup steps (for example, 2,000) regardless of model size and length of
training, as shown in the table at the start of the “Architecture Choices” section. Using 1–5% of training steps is common for
very short training runs. For longer runs, we’ve found that increasing the number of warmup steps doesn’t generally affect
performance.
Let’s look at some common schedules, then discuss how to pick the peak value.
Learning Rate Schedules: Beyond Cosine Decay
Many teams now use schedules where you don’t need to start decaying immediately after warmup. This is the case for the
warmup–stable–decay (WSD) (Hu et al., 2024) and multi-step (DeepSeek-AI, :, et al., 2024) variants shown in the following
plot: You maintain a constant high learning rate for most of training, then either sharply decay in the final phase (typically
the last 10–20% of tokens) for WSD or do discrete drops (steps) to decrease the learning rate. For example, DeepSeek LLM
uses a multi-step learning rate schedule with drops at 80% and 90% of training.
It has been known for years that changing the learning rate helps convergence (Smith & Topin, 2018), and for a long time
cosine decay (Loshchilov & Hutter, 2017) was the go-to schedule for training LLMs: Start at a peak learning rate after
warmup, then smoothly decrease it following a cosine curve. This approach is simple and works well. Its main disadvantage
is inflexibility; we need to know how many steps we’re going to train for up front, as the cosine cycle length must match the
total training duration. This becomes a problem in several common scenarios, such as if your model hasn’t plateaued yet
and you get access to more compute and want to train longer, or you’re running scaling laws and need to train the same
model on different token counts. Cosine decay forces you to restart from scratch.

But you probably noticed that these schedules introduce new hyperparameters compared to cosine decay. How long should
the decay phase last in WSD? And how long should each step be in the multi-step variant?
For WSD: The required cooldown duration to match cosine performance decreases with longer training runs. In general, it
is recommended to allocate 10–20% of total tokens to the decay phase (Hägele et al., 2024). We will confirm this setup
matches cosine in our ablations in the next section.
For multi-step: DeepSeek LLM’s ablations revealed that while their baseline 80/10/10 split (stable until 80%, first step
from 80–90%, second step from 90–100%) matches cosine, it’s possible to outperform it by tweaking these proportions
(e.g., using 70/15/15 and 60/20/20 splits).
We can get even more innovative with learning rate schedules. Let’s look at the schedules used in some of the other
DeepSeek models:
These schedules offer practical advantages over cosine decay. We can extend training mid-run without restarting, decay
early to get a clearer view of training progress, and run scaling law experiments across different token counts within a single
main training run. Moreover, studies show that both WSD and multi-step match the performance of cosine decay
(DeepSeek-AI, :, et al., 2024; Hägele et al., 2024) while being more practical for real-world training scenarios.


DeepSeek LLM used the baseline multi-step schedule (80/10/10). DeepSeek V2 (DeepSeek-AI, Liu, et al., 2024) adjusted
the proportions to 60/30/10, giving more time to the first decay step. DeepSeek V3 (DeepSeek-AI et al., 2025) took the
most creative approach: Instead of maintaining a constant learning rate followed by two sharp steps, it transitioned from
the constant phase with a cosine decay (from 67% to 97% of training), then applied a brief constant phase before the final
sharp step.
DeepSeek schedule changes
DeepSeek-V2 and V3’s technical reports don’t include ablations on these schedule changes. For your setup, start with
simple WSD or multi-step schedules, then consider tuning the parameters through ablations.
Let’s stop our survey of exotic learning rate schedules here and burn some GPU hours to determine what works in practice!
Ablation—WSD Matches Cosine
It’s time for an ablation. Let’s test whether WSD actually matches cosine’s performance. We won’t show multi-step
ablations here, but we recommend you check out DeepSeek LLM’s ablations, which show that multi-step matches cosine
with different phase splits.
We’ll compare cosine decay against WSD with two decay windows: 10% and 20%.




The evaluation results show similar final performance across all three configurations. Looking at the loss and evaluation
curves (specifically HellaSwag), we see an interesting pattern: Cosine achieves better loss and evaluation scores during the
stable phase (before WSD’s decay begins). However, once WSD enters its decay phase, there’s an almost linear
improvement in both loss and downstream metrics, allowing WSD to catch up to cosine by the end of training.
This confirms that WSD’s 10–20% decay window is sufficient to match cosine’s final performance while maintaining the
flexibility to extend training mid-run. We opted for WSD with 10% decay for SmolLM3.
⚠Comparing models trained with different schedulers mid-run
If you’re comparing intermediate checkpoints between cosine and WSD during the stable phase, make sure to apply a decay
to the WSD checkpoint for a fair comparison.
Now that we have a good overview of popular learning rate schedules, the next question is: What should the learning rate
actually be?
Finding the Optimal Learning Rate
To find the optimal learning rate for our specific scheduler and training setup, we could run learning rate sweeps on short
ablations like we did for architecture choices. But optimal learning rate depends on training duration: The learning rate that
converges fastest in a short ablation might not be the best one for the full run. And we can’t afford to run expensive multi-
week trainings multiple times just to test different learning rates.
Let’s start by looking at some simple sweeps we can run that help us quickly rule out learning rates that are much too high
or low. Then we’ll discuss scaling laws for hyperparameters.
Ablation—LR Sweeps
To illustrate the impact of learning rates, let’s do at a sweep on our 1B ablation model trained on 45B tokens. We’ll train
the same model, under the same setup, with four different learning rates: 1e-4, 5e-4, 5e-3, 5e-2. The results clearly show
the dangers at both extremes.



LR 5e-2 diverges almost immediately: The loss spikes early and never recovers, making the model unusable. LR 1e-4 is too
conservative; while it trains stably, it converges much more slowly than the other learning rates. The middle-ground options,
LR 5e-4 and LR 5e-3, show better convergence and comparable performance.
For SmolLM3, we trained 3B models on 100B tokens with AdamW using the WSD schedule, comparing several learning
rates. We found that 2e-4 converged much faster than 1e-4 in both loss and downstream performance, while 3e-4 was only
slightly better than 2e-4. The marginal gains from 3e-4 came with increased risk of instability during long training runs, so
we chose 2e-4 as our sweet spot.
These sweeps help us rule out learning rates that are clearly too high (divergence) or too low (slow convergence)—but
running sweeps for every model size gets expensive quickly, and more importantly, the results for these shorter training
runs may not translate exactly to what we see in a full run. This is where scaling laws become invaluable.
Before we dive into scaling laws for hyperparameters, though, let’s discuss the other critical hyperparameter that interacts
with learning rate: batch size.
BATCH SIZE
There are two ways to scale the batch size:
Increasing the batch size while staying below critical: After increasing the batch size and retuning the learning rate, you
reach the same loss with the same number of tokens as for the smaller batch size run; no data is wasted.
Increasing the batch size while staying above critical: Larger batches start to sacrifice data efficiency; reaching the same
loss now requires more total tokens (and thus more money), even if wall-clock time drops, because more chips are busy.
Let’s consider why retuning the learning rate is necessary and see how to estimate the critical batch size.
When the batch size grows, each mini-batch gradient becomes a better estimate of the true gradient. This allows you to
safely take a larger step (i.e., increase the learning rate) and reach a target loss in fewer updates. The question is how to
scale it.
Averaging over  samples:
Batch gradient: 
Mean stays the same: 
But covariance shrinks: 
The SGD parameter update is:
The variance of this update is proportional to:
So to keep the update variance roughly constant, if you scale the batch size by  , you want to scale the learning rate by
 . Let’s say you’ve computed your optimal batch size and learning rate, and you’ve found that increasing to the critical
batch size is possible and increases throughput. You’ll need to adapt the optimal learning rate as well:
The batch size is the number of samples processed before updating model weights. It directly impacts both training
efficiency and final model performance. Increasing the batch size improves throughput if your hardware and training stack
scale well across devices. But beyond a certain point, larger batches start to hurt data efficiency: The model needs more
total tokens to reach the same loss. The breakpoint where this happens is known as the critical batch size (McCandlish et
al., 2018).
B
  =g~B
   
B1 ∑i=1B g~(i)
E   =[g~B] g
Cov   =(g~B)  
BΣ
Δw=−η   g~B
Var(Δw) ∝η  
2BΣ
k
 k

A useful rule of thumb for optimizers like AdamW or Muon is square root LR scaling as batch size grows, but exactly how
this works depends on the optimizer. For instance, using AdamW, there are interactions with beta1 / beta2 that can
introduce very different behavior. A pragmatic alternative suggested by Merrill et al. (2025) is to branch training for a brief
window: Keep one run at the original batch size, start a second with the larger batch size and a rescaled LR, and only adopt
the larger size if the two loss curves align after the rescale. They warm up the learning rate and reset the optimizer state
when switching the batch size. They also set a tolerance and a time window to decide whether the losses “match,” with both
knobs chosen empirically. Their results indicate that the  estimate—which is also noisy—is underestimating the
“actual” critical batch size. This gives you a quick, low-risk way to check that the new batch/LR pair preserves training
dynamics.
The critical batch size isn’t fixed; it grows as training progresses. Early in training, the model is making big gradient steps,
so  is big. That means  is small, hence the model has a smaller critical batch size. Later, as the model
updates stabilize, larger batches become more effective. This is why some large-scale training runs don’t keep the batch
size constant and use what we call batch size warmup . For example, DeepSeek-V3 begins with a 12.6M batch for the first
~469B tokens, then increases it to 62.9M for the remainder of training. A batch size warmup schedule like this serves the
same purpose as learning rate warmup: It keeps the model on the efficient frontier as the gradient noise scale increases,
maintaining stable and efficient optimization throughout.
Another interesting approach is treating the loss as a proxy for the critical batch size. MiniMax-01 used this, and in the last
stage they trained with a 128M batch size! They didn’t increase the learning rate, so their batch size schedule acted like a
learning rate decay schedule.
Tuning batch size and learning rate
In practice, here’s how you can choose the batch size and learning rate:
First, pick the batch size and learning rate you consider optimal, either based on scaling laws (see the next section) or
from the literature.
Then tune the batch size to see if you can improve the training throughput.
The key insight is that there’s often a range between your starting batch size and the critical batch size where you can
increase it to improve hardware utilization without sacrificing data efficiency, but you must retune the learning rate
accordingly. If the throughput gain isn’t significant, or if testing a larger batch size (with rescaled learning rate) shows worse
data efficiency, stick with the previous values.
As mentioned in the note above, one way to pick your starting points for the batch size and learning rate is through scaling
laws. Let’s see how these laws work.
SCALING LAWS FOR HYPERPARAMETERS
The optimal learning rate and batch size aren’t just about model architecture and size; they also depend on compute
budget, which is determined by the number of model parameters and the number of training tokens. In practice, these
factors interact to determine how aggressive or conservative our updates should be. This is where scaling laws come in.
Scaling laws establish empirical relationships describing how model performance evolves as we increase training scale,
whether that’s through larger models or more training data (see the section at the end of this chapter for the full history).
But they can also help us predict how to adjust key hyperparameters, like the learning rate and batch size, as we scale up
training, as was demonstrated in recent work by DeepSeek LLM and Qwen2.5. This allows us to set principled defaults
rather than relying entirely on hyperparameter sweeps.
B  →critical kB  ⇒optimal η  →critical
 η  koptimal
B  
simple
∥g∥2 B  
simple

To apply scaling laws in this context, we need a way to quantify training scale. The standard metric is the compute budget,
denoted  , which can be approximated as  , where  is the number of model parameters (e.g., 1B =
1e9) and  is the number of training tokens. This is often measured in FLOPs, a hardware-agnostic way of quantifying how
much actual computation is being done. If FLOPs feel too abstract, just think of it this way: Training a 1B-parameter model
on 100B tokens consumes about 2× fewer FLOPs than training a 2B-parameter model on 100B tokens or a 1B-parameter
model on 200B tokens.
Now, how does this relate to learning rate? We can derive scaling laws that predict optimal learning rates and batch sizes
as functions of total compute budget (  ). They help answer questions like:
How should the learning rate change as I scale from 1B to 7B parameters?
If I double my training data, should I adjust the learning rate?
Let’s see how this works by walking through the approach DeepSeek LLM used. First we choose our learning rate schedule,
ideally WSD for its flexibility. Then we train models across a range of compute budgets (e.g., 1e17, 5e17, 1e18, 5e18,
1e19, 2e19 FLOPs) with different combinations of batch sizes and learning rates. In simpler terms: We train different model
sizes for different numbers of tokens, testing different hyperparameter settings. This is where the WSD schedule shines, as
we can extend the same training run to different token counts without restarting.
For each setup, we perform sweeps over learning rate and batch size and identify the configurations that result in near-
optimal performance, typically defined as being within a small margin (e.g., 0.25%) of the best validation loss (computed on
an independent validation set, with a similar distribution to the training set). Each near-optimal configuration gives us a data
point—a tuple of (compute budget  , optimal learning rate  ) or (  , optimal batch size  ). When plotted on a log–log
scale, these relationships typically follow power law behavior, appearing as approximately straight lines (as shown in the
figure below). By fitting these data points, we can extract scaling laws that describe how optimal hyperparameters evolve
with compute.
An important finding from this process is that for a fixed model size and compute budget, performance remains stable
across a wide range of hyperparameters. This means there’s a broad sweet spot rather than a narrow optimum. We don’t
need to find the perfect value, just a value that’s close enough, which makes the whole process much more practical.
Here you can see the results of the scaling laws DeepSeek LLM derived, where each dot represents a near-optimal setting:
C C≈6×N×D ND
The constant 6 comes from empirical estimates of how many floating-point operations are required to train a transformer:
roughly 6 FLOPs per parameter per token.
C
C η C B

"
The core intuition behind these results is that as training becomes larger and longer, we want more stable updates (smaller
learning rates) and more efficient gradient estimation (larger batch sizes).
These scaling laws give us starting points for the learning rate and batch size. But the objective is not “optimal samples per
gradient” but “lower loss reachable within our time and number of GPUs constraints” while still extracting the full signal from
every token.
In practice, you may be able to increase the batch size beyond the predicted optimal batch size to significantly improve
throughput without meaningfully hurting data efficiency, up to the critical batch size we discussed earlier.
SMOLLM3
So what did we end up using for SmolLM3? During ablations, we compared AdamW, AdEMAMix, and Muon on a 1B model
trained on 100B tokens. Muon was able to outperform AdamW when properly tuned but was sensitive to learning rate and
prone to divergence. AdeMAMix achieved similar loss to Muon but was less sensitive. AdamW was the most stable but
reached a higher final loss than the tuned alternatives.
However, when we scaled up to 3B, we encountered more frequent divergence with Muon and AdeMAMix. This may have
been due to a parallelism bug we discovered after finishing the ablations (see “The Training Marathon”), though we haven’t
confirmed this. We decided to use AdamW (beta1: 0.9, beta2: 0.95) with weight decay 0.1 and gradient clipping 1.
Ultimately, a very vanilla setting.
For the learning rate schedule, we chose WSD. We had used it successfully in SmolLM2, and it proved to be one of our best
decisions for ease of use and flexibility regarding total training duration plus the ability to run mid-training decay
experiments. We ran learning rate sweeps and settled on 2e-4. For the global batch size, we tested values from 2M to 4M
tokens but found minimal impact on the loss or downstream performance, so we chose 2.36M tokens, the size that gave
us the best throughput.
RULES OF ENGAGEMENT
We’ve talked a lot about the “what” (optimizer, learning rate, batch size), but just as important is the “how.” How do we
decide what’s worth experimenting with? How do we structure our time? When do we stop exploring and just train?
Allocate your time wisely between exploration and execution. Spending weeks perfecting a minor improvement from a new
method is less valuable than investing that same compute in better data curation or more thorough architecture ablations.
From our experience, though it might disappoint architecture enthusiasts, the biggest performance gains usually come from
data curation.
When in doubt, choose flexibility and stability over peak performance. If two methods perform equally well, pick the one that
offers more flexibility or that has better implementation maturity and stability. A learning rate schedule like WSD that lets
you extend training or run mid-training experiments is more valuable than a rigid schedule that might converge slightly
better.
Know when to stop optimizing and start training. There’s always one more hyperparameter to tune or one more optimizer to
try. Set a deadline for exploration and stick to it—the model you actually finish training will always beat the perfect model
you never start training.
TL;DR: Balance exploration and execution. Done is
better than perfect.



Perfect is the enemy of good, especially when we’re working with finite compute budgets and deadlines.
Scaling Laws: How Many Parameters, How Much Data?
In the early days of deep learning, before language models (and the clusters they were trained on) were “large,” training runs
were often not heavily constrained by compute. When training a model, you’d just pick the largest model and batch size that
fit on your hardware and train until the model started overfitting or you ran out of data. However, even in those early days
there was a sense that scale was helpful—for example, Hestness et al. provided a comprehensive set of results in 2017
showing that training larger models for longer produced predictable gains.
In the era of large language models, we are always compute-constrained. Why? The early notions of scalability were
formalized by Kaplan et al.’s work in “Scaling Laws for Neural Language Models,” where it was shown that language model
performance is remarkably predictable across many orders of magnitude of scale. This set off an explosion in the sizes and
training durations of language models, because it provided a way to accurately predict how much increasing scale would
improve performance. Consequently, the race to build better language models became a race to train larger models on
larger amounts of data with ever-growing compute budgets, and the development of language models quickly became
compute-constrained.
When faced with compute constraints, the most important question is whether to train a larger model or to train on more
data. Surprisingly, Kaplan et al.’s scaling laws suggested that it was advantageous to allocate much more compute to
model scale than previous best practices—motivating, for example, training the gargantuan (175B parameters) GPT-3
model on a relatively modest token budget (300B tokens). On reexamination, Hoffman et al. (2022) found a methodological
issue with Kaplan et al.’s approach, ultimately re-deriving scaling laws that suggested allocating much more compute to
training duration—which indicated, for example, that compute-optimal training of the 175B-parameter GPT-3 should have
consumed 3.7T tokens! Hoffman et al.’s revised scaling laws became known as the Chinchilla laws, named after the
Chinchilla model that motivated the update.
This development shifted the field from “make models bigger” to “train them longer and better.” However, most modern
training runs still don’t strictly follow the Chinchilla laws, because they have a shortcoming: They aim to predict the model
size and training duration that achieve the best performance given a certain compute budget, but they fail to account for the
fact that larger models are more expensive after training. Put another way, we might actually prefer to use a given compute
budget to train a smaller model for longer, even if this isn’t “compute-optimal,” because it will make inference costs cheaper
(Sardana et al., 2025; de Vries, 2023). This could be the case if we expect that a model will be see a lot of inference usage
(for example, because it’s being released openly 🤗 ). Recently, this practice of “overtraining” models beyond the training
duration suggested by scaling laws has become standard practice, and it’s the approach we took when developing
SmolLM3.
While scaling laws provide a suggestion for the model size and training duration given a particular compute budget,
choosing to overtrain means you have to decide on these factors yourself. For SmolLM3, we started by picking a target
model size of 3B parameters. Based on recent models of a similar scale, like Qwen3-4B, Gemma 3 4B, and Llama 3.2 3B,
we considered 3B to be large enough for the model to have meaningful capabilities (such as reasoning and tool calling), but
small enough to enable super-fast inference and efficient local usage. To pick a training duration, we first noted that recent
models have been extremely overtrained—for example, the aforementioned Qwen3 series claims to have been trained for
36T tokens! As a result, training duration is often dictated by the amount of compute available. We secured 384 H100s for
roughly a month, which provided a budget for training on 11T tokens (assuming a model FLOPs utilization of ~30%).
The value of scaling laws
One more ablation won't hurt! (Spoiler: It did.) Credit to sea_snell.

Despite these deviations, scaling laws remain practically valuable. They provide baselines for experimental design, people
often use Chinchilla-optimal setups to get signal on ablations, and they help predict whether a given model size can reach a
target performance. As Harm de Vries notes in “Go Smol or Go Home,” by scaling down model size you can hit a critical
model size: the minimal capacity required to reach a given loss, below which you start getting diminishing returns.
Now that we’re settled on our model architecture, training setup, model size, and training duration, we need to prepare two
critical components: the data mixture that will teach our model, and the infrastructure that will train it reliably. With
SmolLM3’s architecture set at 3B parameters, we needed to curate a data mixture that would deliver strong multilingual,
math, and code performance and set up infrastructure robust enough for 11T tokens of training data. Getting these
fundamentals right is essential—even the best architectural choices won’t save us from poor data curation or unstable
training systems.
