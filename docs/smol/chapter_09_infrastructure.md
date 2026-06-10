# Chapter 9: Infrastructure—The Unsung Hero

Infrastructure—The Unsung Hero
Now that you know everything we know about model creation and training, let’s address the critical yet underrated
component that can make or break your project (and your bank account): infrastructure. Whether you focus on frameworks,


architecture, or data curation, understanding infrastructure basics helps identify training bottlenecks, optimize parallelism
strategies, and debug throughput issues. (At a minimum, it improves communication with infrastructure teams!)
Most people training models care deeply about architecture and data, yet very few understand the infrastructure details.
Infrastructure expertise typically lives with framework developers and cluster engineers and gets treated by the rest as a
solved problem: Rent some GPUs, install PyTorch, and you’re good to go. We trained SmolLM3 on 384 H100s for nearly a
month, processing a total of 11 trillion tokens… and this was not a smooth ride! During that time, we dealt with node
failures, storage issues, and run restarts (as described in “The Training Marathon”). You need to have good contingency
plans and strategies to prepare for these kinds of issues and keep training smooth and low-maintenance.
This chapter aims to bridge that knowledge gap. Think of it as a practical guide to the hardware layer, focused on the
questions that matter for training. (Note: Each subsection starts with a TL;DR so you can choose your depth level.)
The first two sections tackle the fundamentals of how hardware works: What does a GPU actually consist of? How does
memory hierarchy work? How do CPUs and GPUs communicate? We’ll also go over what to consider when acquiring GPUs
and how to test them before committing to long training runs. Most importantly, we’ll show you at each step how to
measure and diagnose these systems yourself. The following sections are more applied: You’ll see how to make your
infrastructure resilient to failure, and how to maximally optimize your training throughput.
The name of the game of this chapter is to find and fix the bottlenecks!
Think of this as building your intuition for why certain design decisions matter. When you understand that your model’s
activations need to flow through multiple levels of cache, each with different bandwidth and latency characteristics, you’ll
naturally start thinking about how to structure your training to minimize data movement. When you see that internode
communication is orders of magnitude slower than intranode communication, you’ll understand why parallelism strategies
matter so much.
Let’s start by cracking open a GPU and seeing what’s inside.
Inside a GPU: Internal Architecture
A GPU is fundamentally a massively parallel processor optimized for throughput over latency. Unlike CPUs, which excel at
executing a few complex instruction streams quickly, GPUs achieve performance by executing thousands of simple
operations simultaneously.
The key to understanding GPU performance lies in recognizing that it’s not just about raw compute power; it’s about the
interplay between computation and data movement. A GPU can have teraflops of theoretical compute, but if data can’t
reach the compute units fast enough, that potential goes unused. This is why we need to understand both the memory
hierarchy (how data moves) and the compute pipelines (how work gets done).
Putting it simply, at the highest level, a GPU performs two essential tasks:
1. Move and store data (the memory system).
2. Do useful work with the data (the compute pipelines).
In the following subsections, we’ll explore both sides of this equation: how GPUs compute (FLOPs, Tensor Cores, precision)
and how they move data (aka the memory hierarchy, from High Bandwidth Memory [HBM] down to registers).
COMPUTE UNITS AND FLOPS
TL;DR: GPUs measure performance in FLOPs (floating-point operations per second). Modern GPUs like the H100 deliver
dramatically higher throughput at lower precision: 990 TFLOPs at BF16 vs. 67 TFLOPs at FP32. However, real-world

performance is 70–77% of theoretical peaks due to memory bottlenecks. SOTA training achieves 20–41% end-to-end
efficiency, also known as model FLOPs utilization (MFU). Use realistic numbers, not marketing specs, when planning training
runs.
GPU compute performance is measured in floating-point operations per second, or FLOPs. A FLOP is a single arithmetic
operation, like a + b , and modern GPUs can execute trillions of these per second (TFLOPs).
The fundamental building blocks of GPU compute are Streaming Multiprocessors (SMs) , independent processing units that
execute instructions in parallel. Each SM contains two types of cores : CUDA Cores for standard floating-point operations,
and specialized Tensor Cores optimized for matrix multiplication, the workhorse operation in deep learning (critical for
transformer performance).
With hundreds of SMs each executing multiple warps concurrently, a single GPU can run tens of thousands of threads
simultaneously. This massive parallelism is what enables GPUs to excel at the matrix operations that dominate deep
learning workloads.
Modern GPUs organize hundreds of these SMs across the chip. For example, the H100 SXM5 version (which is the GPU
we’re using on our cluster) contains 132 SMs. Each SM operates independently, executing groups of 32 threads called
warps in lockstep. To help there, the SMs rely on another component, the warp schedulers : By balancing instructions to
different warps, they enable the SM to “hide” latency by switching between warps when one is held up. This SIMT (single
instruction, multiple threads) execution model means all threads in a warp execute the same instruction simultaneously on
different data.
Multiple SMs within a single GPU (source: https://www.youtube.com/watch?v=ZQKMZIP3Fzg).

The following table shows theoretical peak performance (TFLOPs) across different NVIDIA GPU generations and precisions:
Precision/GPU Type A100 H100 H200 B100 B200
FP64 9.7 34 34 40 40
FP32 19.5 67 67 80 80
FP16/BF16 312 990 990 1.750 2,250
FP8 - 3,960 3,960 4,500 5,000
FP4 - - - 9,000 10,000
Source: NVIDIA, SemiAnalysis
The dramatic increase in throughput at lower precision isn’t just about raw speed; it reflects a fundamental shift in how we
think about numerical computation. FP8 and FP4 enable models to perform more operations per watt and per second ,
making them essential for both training and inference at scale. The H100’s 3,960 TFLOPs at FP8 represents a 4×
improvement over FP16/BF16, while the B200’s 10,000 TFLOPs at FP4 pushes this even further.
Note that these theoretical peak FLOPs represent the maximum computational throughput achievable under ideal conditions
, when all compute units are fully utilized and data is readily available. In practice, actual performance depends heavily on
how well your workload can keep the compute units fed with data and whether your operations can be efficiently mapped to
the available hardware.
For SmolLM3, we were going to train on NVIDIA H100 80GB HBM3 GPUs, so we first wanted to test the H100’s theoretical
TFLOPs specifications against real-world performance. For this, we used the SemiAnalysis GEMM benchmark, which tests
throughput on real-world matrix multiplication shapes from Meta’s Llama 70B training.
The following table shows achieved TFLOPs on H100 80GB GPUs depending on precision and matrix shape from the Llama
70B training workload:
Validating Theoretical Performance
Precision matters when discussing FLOPs. Tensor Cores can operate at different precisions (FP64, FP32, FP16/BF16, FP8,
FP4). The achievable throughput therefore varies dramatically, often by orders of magnitude, depending on the data type.
Lower-precision formats enable higher throughput because they require less data movement and can pack more operations
into the same silicon area. They were previously avoided because of training instabilities, but nowadays both training and
inference are increasingly being pushed toward lower precision, reaching FP8 and FP4, thanks to a range of new techniques.
Our experiments revealed the gap between theoretical peaks and achievable performance. For FP64 Tensor Core
operations, we achieved 49–56 TFLOPs, representing 74–84% of the theoretical peak (67 TFLOPs). For TF32 (TensorFloat-
32, which PyTorch uses by default for FP32 tensors on Tensor Cores), we achieved 356–396 TFLOPs, representing 72–80%
of the theoretical peak (~495 TFLOPs dense). While these results indicate excellent hardware utilization, these precisions
Shape (M, N, K)FP64 torch.matmulFP32 torch.matmulFP16 torch.matmulBF16 torch.matmulFP8 TE.Linear (autoc
(16384, 8192, 1280)51.5 TFLOPs364.5 TFLOPs686.5 TFLOPs714.5 TFLOPs837.6 TFLOPs
(16384, 1024, 8192)56.1 TFLOPs396.1 TFLOPs720.0 TFLOPs757.7 TFLOPs547.3 TFLOPs
(16384, 8192, 7168)49.5 TFLOPs356.5 TFLOPs727.1 TFLOPs752.9 TFLOPs1120.8 TFLOPs
(16384, 3584, 8192)51.0 TFLOPs373.3 TFLOPs732.2 TFLOPs733.0 TFLOPs952.9 TFLOPs
(8192, 8192, 8192)51.4 TFLOPs372.7 TFLOPs724.9 TFLOPs729.4 TFLOPs1029.1 TFLOPs

For BF16 operations, we consistently achieved 714–758 TFLOPs across different matrix shapes, representing approximately
72–77% of the H100’s theoretical 990 TFLOPs peak. This is, in practice, an excellent utilization rate for a real-world
workload!
📊Model FLOPs utilization
While kernel benchmarks measure raw TFLOPs, end-to-end training efficiency is captured by model FLOPs utilization (MFU):
the ratio of useful model computation to theoretical peak hardware performance.
Our BF16 matmul benchmarks showed we achieved 72–77% of the H100’s theoretical peak. This represents the upper
bound for what’s achievable at the kernel level for our setup. End-to-end training MFU will necessarily be lower due to more
complex non-matmul operations, communication overhead, and other auxiliary computations.
State-of-the-art MFU in training: Meta achieved 38–41% when training Llama 3 405B, while DeepSeek-V3 reached ~20–30%
on GPUs with tighter communication bottlenecks related to the MoE architecture. For SmolLM3, we achieved ~30% MFU, as
you’ll see later. Much of the gap comes from internode communication overhead in distributed training. Given our kernel-level
ceiling of ~77%, these end-to-end numbers represent roughly 50–55% efficiency relative to achievable matmul performance.
Inference workloads can reach higher MFU (>70%, closer to raw matmul performance), though published results from
production deployments are scarce.
Using PyTorch’s torch._scaled_mm kernel with e4m3 precision, we achieved 1,210–1,457 TFLOPs depending on the
matrix shape, or roughly 31–37% of the theoretical 3,960 TFLOPs peak. Why? This lower utilization percentage (in FP8)
actually doesn’t indicate poor performance; rather, it reflects that these operations become increasingly memory-bound as
compute throughput grows. The Tensor Cores can process FP8 data faster than the memory system can deliver it, making
memory bandwidth the limiting factor.
Transformer Engine’s TE.Linear achieved 547–1,121 TFLOPs depending on the shape, while torch._scaled_mm
consistently delivered higher throughput. This highlights an important lesson: Kernel implementation matters, and the
choice of API can impact performance by 2–3× even when targeting the same hardware capabilities.
For SmolLM3’s training, these practical measurements helped us set realistic throughput expectations. When planning your
own training runs, be sure to use the achievable numbers rather than theoretical peaks.
🔧Compute capability
Besides choosing the right kernel API, we also need to ensure those kernels are compiled for the right hardware generation.
Compute capability (CC) is NVIDIA’s versioning system that abstracts physical GPU details from the PTX instruction set. It
determines which instructions and features your GPU supports.
Why this matters: Kernels compiled for a specific compute capability may not run on older hardware, and you might miss
optimizations if your code isn’t compiled for your target GPU’s CC. Even worse, frameworks can silently select suboptimal
kernels—we discovered PyTorch selecting sm_75 kernels (compute capability 7.5, designed for Turing GPUs) on our H100s,
causing mysterious slowdowns. A similar issue is documented in the PyTorch community, where frameworks often default to
older, more compatible kernels rather than optimal ones. This seemingly minor detail can mean the difference between
getting 720 TFLOPs or 500 TFLOPs from the same hardware.
When using precompiled libraries or custom kernels, always verify they’re built for your hardware’s compute capability to
ensure compatibility and optimal performance. For example, sm90_xmma_gemm_..._cublas indicates a kernel compiled for
SM 9.0 (compute capability 9.0, used by the H100).
are rarely used in modern deep learning training: FP64 due to its computational cost, and TF32 because lower precisions
like BF16 and FP8 offer better performance.
The FP8 results are more nuanced. Let’s look at our results on three different matrix multiplication methods/kernels.

You can check your GPU’s compute capability with nvidia-smi --query-gpu=compute_cap or find the technical
specifications in the Compute Capability section of the NVIDIA CUDA C Programming Guide.
As we’ve seen, GPU memory seems to become a bottleneck when computations get too fast at a low precision. Let’s take a
closer look at how GPU memory works and what causes bottlenecks to occur.
GPU MEMORY HIERARCHY: FROM REGISTERS TO HBM
TL;DR: GPUs organize memory in a hierarchy from fast but small (registers, shared memory) to slow but large (HBM main
memory). Understanding this hierarchy is critical because modern AI is often memory-bound: The bottleneck is moving data,
not computing on it. Operator fusion (like Flash Attention) achieves 2–4× speedups by keeping intermediate results in fast
on-chip memory instead of writing to slow HBM. Benchmarks show the H100’s HBM3 delivers ~3 TB/s in practice, matching
theoretical specs for large transfers.
Understanding the GPU memory hierarchy is crucial for writing high-performance kernels. To make calculations, GPUs need
to read from/write to memory, so it’s important to know at what speed these transfers happen.
To visualize how memory operations flow through a GPU in practice, let’s first look at the Memory Chart from NVIDIA Nsight
Compute, a profiling graph that provides a graphical representation of how data moves between different memory units for
any kernel of your choice:
Memory Chart showing data flow through the GPU memory hierarchy during FP64 matrix multiplication on an H100.

In general, a Memory Chart shows both logical units (in green), like Global, Local, Texture, Surface, and Shared, and
physical units (in blue), like L1/TEX Cache, Shared Memory, L2 Cache, and Device Memory. Links between units represent
the number of instructions (Inst) or requests (Req) happening between units, with colors indicating the percentage of peak
utilization: from unused (0%) to operating at peak performance (100%).
You can generate this Memory Chart for any kernel using NVIDIA Nsight Compute:
It provides several key insights:
Bottleneck identification: Saturated links (shown in red/orange) indicate where data movement is constrained.
Cache efficiency: Hit rates for L1/TEX and L2 caches reveal how well your kernel utilizes the memory hierarchy.
Memory access patterns: The flow between logical and physical units shows whether your kernel has good
spatial/temporal locality.
Port utilization: Individual memory ports may be saturated even when aggregate bandwidth appears underutilized.
In our specific case, you can see how kernel instructions flow through the memory hierarchy (for FP64 matrix multiplications
on our hardware): Global load instructions generate requests to the L1/TEX cache, which may hit or miss and generate
further requests to L2, which ultimately accesses device memory (HBM) on misses. The colored rectangles inside units
show port utilization; even if individual links operate below peak, the shared data port may be saturated.
Let’s explore the underlying memory hierarchy that makes this chart possible. Modern GPUs organize memory in a hierarchy
that balances speed, capacity, and cost, a design dictated by fundamental physics and circuit constraints.
💡Optimizing memory hierarchy access
For optimal performance, aim to minimize traffic to slower memory tiers (HBM) while maximizing utilization of faster tiers
(shared memory, registers).
## Profile a specific kernel with memory workload analysis
1
ncu --set full --kernel-name "your_kernel_name" --launch-skip 0 --launch-count 1 python 
your_script.py
2
## Once profiling is complete, open the results in the Nsight Compute GUI to view the Memory Chart
3

At the bottom of this hierarchy sits HBM (High Bandwidth Memory): the GPU’s main memory, also called global memory or
device memory . The H100 features HBM3 with a theoretical bandwidth of 3.35 TB/s. HBM is the largest but slowest tier in
the memory hierarchy.
Moving up the hierarchy toward the compute units, we find progressively faster but smaller memory tiers:
L2 cache: A large SRAM-based cache shared across the GPU, typically several tens of megabytes. On the H100, this is
50 MB with a bandwidth of ~13 TB/s.
L1 cache and shared memory (SMEM): Each Streaming Multiprocessor has its own L1 cache and programmer-managed
shared memory, which share the same physical SRAM storage. On the H100, this combined space is 256 KB per SM
with a bandwidth of ~31 TB/s per SM.
Register memory (RMEM): At the top of the hierarchy, registers are the fastest storage, located directly next to compute
units. Registers are private to individual threads and provide effective bandwidth on the order of 100 TB/s per SM.
This hierarchy exists because SRAM (used for caches and registers) is fast but physically large and expensive, while DRAM
(used for HBM) is dense and cheap but slower. The result: Fast memory comes in small quantities close to compute,
backed by progressively larger pools of slower memory further away.
Why This Matters
Understanding this hierarchy is essential for kernel optimization. The key insight is that memory-bound operations are
limited by how fast you can move data, not how fast you can compute. As Horace He explains in “Making Deep Learning Go
Brrrr from First Principles,” load from memory → multiply by itself twice → write to memory takes essentially the same time
as load from memory → multiply by itself once → write to memory. The computation is “free” compared to the memory
access.
This is why operator fusion is so powerful: By combining multiple operations into a single kernel, you can keep intermediate
results in fast SRAM instead of writing them back to slow HBM between operations. Flash Attention is a perfect example of
this principle in action.
Memory hierarchy of the H100 (SXM5) GPU (source: Inside NVIDIA GPUs, by Aleksa Gordi ć ).

⚡Flash Attention: A case study in memory hierarchy optimization
Standard attention implementations are memory-bound because they materialize the full attention matrix in HBM:
1. Compute Q @ K^T → write N × N attention scores to HBM
2. Apply softmax → read from HBM, compute, write back to HBM
3. Multiply by V → read attention scores from HBM again
Flash Attention achieves its 2–4× speedup by fusing these operations and keeping intermediate results in SRAM:
Instead of computing the full attention matrix, it processes attention in tiles that fit in SRAM.
Intermediate attention scores never leave the fast on-chip memory.
Only the final output is written back to HBM.
The result: Flash Attention reduces HBM accesses from O ( N ²) to O ( N ), transforming a memory-bound operation into one
that better utilizes the GPU’s compute capabilities. This is the essence of efficient kernel design: minimize slow memory
movement, maximize fast computation .
Example: Validating Our HBM3 Bandwidth in Practice
Now that we understand the memory hierarchy, let’s put theory into practice and validate the actual bandwidth on our H100
GPUs! This is where benchmarking tools become essential.
NVBandwidth is NVIDIA’s open source benchmarking tool, designed specifically for measuring bandwidth and latency across
GPU systems. It evaluates data transfer rates for various memory copy patterns—host-to-device, device-to-host, and device-
to-device—using both copy engines and kernel-based methods. The tool is particularly valuable for assessing inter-GPU
communication (for example, via NVLink and PCIe) and validating system performance in multi-GPU environments.
You can install NVBandwidth from NVIDIA’s GitHub repository. It outputs detailed bandwidth matrices showing how efficiently
data transfers between different devices, making it ideal for diagnosing performance bottlenecks and verifying healthy GPU
interconnects.
We can use this tool to measure our H100’s local memory bandwidth using the device_local_copy test, which measures
the bandwidth of cuMemcpyAsync between device buffers local to the GPU across different message sizes:
$ ./nvbandwidth -t device_local_copy -b 2048
1
memcpy local GPU(column) bandwidth (GB/s)
2
           0         1         2         3         4         5         6         7
3
 0   1519.07   1518.93   1519.07   1519.60   1519.13   1518.86   1519.13   1519.33
4
5

The results reveal an important characteristic of memory systems: For small message sizes (< 1 MB), we’re latency-bound
rather than bandwidth-bound. The overhead of initiating memory transfers dominates performance, preventing us from
reaching peak bandwidth. However, for large message sizes (≥ 1 MB), we achieve sustained bandwidth of ~1,500 GB/s for
both read and write operations .
Since HBM bandwidth accounts for both reads and writes happening simultaneously, we sum these to get 3 TB/s total
bidirectional bandwidth (1,519 read + 1,519 write), which closely validates the H100’s theoretical 3.35 TB/s HBM3
specification.
ROOFLINE MODEL
Understanding whether your kernel is compute-bound or memory-bound determines which optimizations will help. There are
two scenarios:
If you’re memory-bound (spending most time moving data), increasing compute throughput won’t help: You need to
reduce memory traffic through techniques like operator fusion.
If you’re compute-bound (spending most time on FLOPs), optimizing memory access patterns won’t help: You need more
compute power or better algorithms.
The roofline model provides a visual framework for understanding these performance characteristics and identifying
optimization opportunities. A roofline view is available in the Nsight Compute profiling tool we mentioned earlier. Here’s what
we get when we apply it to our kernel:
Measured H100 local memory bandwidth

Let’s see how to read this chart. It has two axes:
The vertical axis ( Performance ) shows the number of FLOPs achieved, using a logarithmic scale to accommodate the
large range of values.
The horizontal axis ( Arithmetic Intensity ) represents the ratio of work to memory traffic, measured in FLOPs per byte.
Again, it uses a logarithmic scale.
The roofline itself consists of two boundaries:
The memory bandwidth boundary (the sloped line) is determined by the GPU’s memory transfer rate (HBM bandwidth).
Performance along this line is limited by how fast data can be moved.
The peak performance boundary (the flat line) is determined by the GPU’s maximum compute throughput. Performance
along this line is limited by how fast computations can be executed.
The ridge point where these boundaries meet represents the transition between memory-bound and compute-bound
regimes.
We can interpret the performance by looking at the two divided regions of the chart:
Kernels in the memory-bound region (below the sloped boundary) are limited by memory bandwidth. The GPU is waiting
for data; increasing compute power won’t help. Optimizations should focus on reducing memory traffic through
techniques like operator fusion, better memory access patterns, or increasing arithmetic intensity.
Kernels in the compute-bound region (below the flat boundary) are limited by compute throughput. The GPU has enough
data but can’t process it fast enough. Optimizations should focus on algorithmic improvements or leveraging specialized
hardware like Tensor Cores.
The achieved value (the plotted point) shows where your kernel currently sits. The distance from this point to the roofline
boundary represents your optimization headroom: The closer to the boundary, the more optimal your kernel’s performance.
Roofline chart showing kernel performance boundaries (source: NVIDIA Nsight Compute Profiling Guide).

Now that we understand what happens inside a GPU, let’s zoom out and explore how GPUs communicate with the rest of
the world.
Outside a GPU: How GPUs Talk to the World
We’ve covered how a GPU performs computation using its internal memory hierarchy, but at this point we need to address a
critical reality: A GPU doesn’t operate in isolation. Before any computation can happen, data must be loaded into the GPU’s
memory. The CPU needs to schedule kernels and coordinate work. And in distributed training, GPUs must constantly
exchange activations, gradients, and model weights with each other.
This means the external communication infrastructure is crucial. No matter how powerful your GPU’s compute units are, if
data can’t reach them fast enough, whether from the CPU, from storage, or from other GPUs, your expensive hardware will
sit idle. Understanding these communication pathways and their bandwidth characteristics is essential for maximizing
hardware utilization and minimizing bottlenecks.
In this section, we’ll look at four critical communication links that connect a GPU to the outside world:
CPU–GPU (how the CPU schedules work and transfers data to GPUs)
GPU–GPU intranode (how GPUs on the same machine communicate)
GPU–GPU internode (how GPUs on different machines communicate over the network)
GPU–storage (how data flows from storage to GPU memory)
Each of these links has different bandwidth and latency characteristics, and understanding them will help you identify where
your training pipeline might be bottlenecked. To make this easier to understand, we’ve created a simplified diagram that
highlights the most important components and communication links:
In our example, the kernel sits in the memory-bound region, indicating that there’s still room for improvement by optimizing
memory traffic.

If this looks overwhelming, don’t worry. We’ll dive into each of these connections in detail and measure their actual
bandwidths to understand the performance characteristics of each link.
CPU-TO-GPU COMMUNICATION
TL;DR: The CPU orchestrates GPU work via PCIe connections, which bottleneck at ~14.2 GB/s (PCIe Gen4 x8) for CPU-to-
GPU transfers in our P5 instance. CPU-GPU latency is ~1.4 microseconds (μs), which adds kernel launch overhead that is
problematic for workloads with many small kernels. CUDA Graphs can reduce this overhead by batching operations. NUMA
affinity is critical on multi-socket systems; running GPU processes on the wrong CPU socket adds significant latency. Modern
architectures like Grace Hopper eliminate PCIe bottlenecks with NVLink-C2C (900 GB/s vs. 128 GB/s).
The CPU is the orchestrator of GPU computation. It’s responsible for launching kernels, managing memory allocations, and
coordinating data transfers. But how fast can the CPU actually communicate with the GPU? This is determined by the PCIe
(Peripheral Component Interconnect Express) connection between them.
Understanding this link is vital, because it affects:
Kernel launch latency (how quickly the CPU can schedule work on the GPU)
Data transfer speed (how fast we can move data between CPU and GPU memory)
Synchronization overhead (the cost of CPU–GPU coordination points)
In modern GPU servers, the CPU–GPU connection has evolved significantly. While earlier systems used direct PCIe
connections, modern high-performance systems like the DGX H100 use more sophisticated topologies with PCIe switches
to manage multiple GPUs efficiently. And with the latest GB200 architecture, NVIDIA has taken this even further by placing
the CPU and GPU on the same printed circuit board, eliminating the need for external switches altogether.
Simplified diagram of the key components and communication links in our AWS P5 instance setup.

Let’s examine the physical topology of our P5 instance using lstopo and then measure the actual performance of this
critical link, to identify potential bottlenecks:
From the lstopo output, we can see two key PCIe bandwidth values in our system:
15.75GB/s: Corresponds to PCIe Gen4 x8 links (CPU to PCIe switches)
63.02GB/s: Corresponds to PCIe Gen5 x16 links (PCIe switches to GPUs)
To get a better understanding of the whole topology, we can visualize it using:
This diagram showcases the hierarchical structure of our system. The key points to notice for now are:
$ lstopo -v
1
...
2
HostBridge L#1 (buses=0000:[44-54])
3
    PCIBridge L#2 (busid=0000:44:00.0 id=1d0f:0200 class=0604(PCIBridge) link=15.75GB/s 
buses=0000:[45-54] PCISlot=64)
4
        PCIBridge L#3 (busid=0000:45:00.0 id=1d0f:0200 class=0604(PCIBridge) link=15.75GB/s 
buses=0000:[46-54] PCISlot=1-1)
5
            ...
6
            PCIBridge L#12 (busid=0000:46:01.4 id=1d0f:0200 class=0604(PCIBridge) link=63.02GB/s 
buses=0000:[53-53])
7
                PCI L#11 (busid=0000:53:00.0 id=10de:2330 class=0302(3D) link=63.02GB/s 
PCISlot=86-1)
8
                    Co-Processor(CUDA) L#8 (Backend=CUDA GPUVendor="NVIDIA Corporation" 
GPUModel="NVIDIA H100 80GB HBM3" CUDAGlobalMemorySize=83295872 CUDAL2CacheSize=51200 
CUDAMultiProcessors=132 CUDACoresPerMP=128 CUDASharedMemorySizePerMP=48) "cuda0"
9
                    GPU(NVML) L#9 (Backend=NVML GPUVendor="NVIDIA Corporation" GPUModel="NVIDIA 
H100 80GB HBM3" NVIDIASerial=1654922006536 NVIDIAUUID=GPU-ba136838-6443-7991-9143-1bf4e48b2994) 
"nvml0"
10
            ...
11
...
12
13
$ lstopo --whole-system lstopo-diagram.png
1
2


It contains two Non-Uniform Memory Access (NUMA) nodes (one memory zone per CPU socket).
Each CPU socket connects to four PCIe switches via PCIe Gen4 x8 links (15.75 GB/s).
Each PCIe switch connects to one H100 GPU via PCIe Gen5 x16 links (63.02 GB/s).
We’ll explore other components, like the NVSwitch, EFA network cards, and NVMe drives, in later sections.
The PCIe specification differs between generations, each doubling the transfer rate per lane, as shown in the following
table. Note that transfer rate is measured in GT/s (gigatransfers per second), which represents the raw signaling rate, while
throughput is measured in GB/s (gigabytes per second), which accounts for encoding overhead and represents the actual
usable bandwidth:
PCIe Version Transfer Rate (per Lane) Throughput (GB/s)
×1 ×2 ×4
1.0 2.5 GT/s 0.25
2.0 5.0 GT/s 0.5
3.0 8.0 GT/s 0.985
4.0 16.0 GT/s 1.969
5.0 32.0 GT/s 3.938
6.0 64.0 GT/s 7.563
7.0 128.0 GT/s 15.125
Theoretical PCIe bandwidths. Source: https://en.wikipedia.org/wiki/PCI_Express.

From the topology diagram and the PCIe bandwidth table, we can see that the CPU-to-GPU path goes through two PCIe
hops: first from the CPU to the PCIe switch via PCIe Gen4 x8 (15.754 GB/s), then from the PCIe switch to the GPU via PCIe
Gen5 x16 (63.015 GB/s). This means the bottleneck for CPU–GPU communication is the first hop , at 15.754 GB/s. Let’s
validate this with another utility, nvbandwidth .
The host_to_device_memcpy_ce command measures the bandwidth of cuMemcpyAsync from host (CPU) memory to
device (GPU) memory using the GPU’s copy engines:
$ ./nvbandwidth -t host_to_device_memcpy_ce -b <message_size> -i 5
1
CPU-to-GPU communication path.

The results indeed show that for small message sizes we’re latency-bound, but for large message sizes we achieve ~14.2
GB/s, which is about 90% of the theoretical 15.754 GB/s bandwidth for PCIe Gen4 x8. This confirms that in CPU–GPU
communication, the CPU-to-PCIe switch link is our bottleneck.
Beyond bandwidth, latency is equally important for CPU–GPU communication since it determines how quickly we can
schedule kernels. To measure this, we use the host_device_latency_sm test, which measures round-trip latency by
allocating a buffer on the host (CPU) and accessing it from the GPU using a pointer-chase kernel. This simulates the real-
world latency of CPU–GPU communication:
$ ./nvbandwidth -t host_device_latency_sm -i 5
1
CPU-to-GPU measured bandwidth

🚀CUDA Graphs for reducing launch overhead
CUDA Graphs can significantly reduce this overhead by capturing a sequence of operations and replaying them as a single
unit, eliminating microseconds of CPU–GPU round-trip latency for each kernel launch. This is particularly beneficial for
workloads with many small kernels or frequent CPU–GPU synchronization. For more details on understanding and optimizing
launch overhead, see “Understanding the Visualization of Overhead and Latency in NVIDIA Nsight Systems” on the NVIDIA
Technical Blog.
⚠MoE models and CPU–GPU synchronization overhead
Some implementations of mixture-of-experts models require CPU–GPU synchronization in each iteration to schedule the
appropriate kernels for the selected experts. This introduces kernel launch overhead that can significantly affect throughput,
especially when the CPU–GPU connection is slow. For example, in MakoGenerate’s optimization of DeepSeek MOE kernels,
the reference implementation dispatched 1,043 kernels with 67 CPU–GPU synchronization points per forward pass. By
restructuring the expert routing mechanism, they reduced this to 533 kernel launches and just 3 synchronization points,
achieving a 97% reduction in synchronization overhead and a 44% reduction in end-to-end latency. Note that not all MoE
The results show that the latency is approximately 1.4 μs. This explains the kernel launch overhead of a few microseconds
that we often observe in ML workloads. For workloads launching many small kernels, the added latency can become a
bottleneck; otherwise, the overhead is hidden by overlapping execution.
CPU-to-GPU measured latency

implementations require CPU–GPU synchronization (modern implementations often keep routing entirely on the GPU), but for
those that do, efficient CPU–GPU communication becomes critical for performance.
🔗Grace Hopper superchips: A different approach to CPU–GPU communication
NVIDIA’s Grace Hopper superchips take a fundamentally different approach to CPU–GPU communication compared to
traditional x86+Hopper systems. Key improvements include:
1:1 GPU-to-CPU ratio (compared to 4:1 for x86+Hopper), providing 3.5× higher CPU memory bandwidth per GPU
NVLink-C2C replacing PCIe Gen5 lanes, delivering 900 GB/s vs. 128 GB/s (7× higher GPU–CPU link bandwidth)
NVLink Switch System providing 9× higher GPU–GPU link bandwidth than InfiniBand NDR400 NICs connected via PCIe
Gen4
For more details, see the “NVIDIA Grace Hopper Superchip Architecture” whitepaper (page 11).
NUMA Affinity: Critical for Multi-Socket Performance
On multi-socket systems like our AMD EPYC 7R13 nodes (2 sockets, 48 cores each), NUMA affinity is crucial for GPU
performance . NUMA affinity refers to running processes on CPU cores that share the same socket as their target devices
(like GPUs). When your GPU process runs on CPUs from a different NUMA node than where the GPU is attached, operations
must traverse the CPU interconnect (AMD Infinity Fabric), adding significant latency and bandwidth constraints.
Examining the NUMA topology and node distances will give you a better understanding of the performance implications:
Accessing memory on the same NUMA node (distance 10) will be much faster than crossing to the other NUMA node
(distance 32). This 3.2× difference in memory access latency can significantly impact GPU performance when your process
is pinned to the wrong NUMA node.
For detailed steps for diagnosing and resolving NUMA-related performance issues, see the “Troubleshooting Interconnect”
section later in this chapter.
GPU-TO-GPU INTRANODE COMMUNICATION
TL;DR: GPUs within a node can communicate in three ways: through the CPU (slowest, ~3 GB/s, bottlenecked by PCIe), via
GPUDirect RDMA over EFA NICs (~38 GB/s), or via GPUDirect RDMA via NVLink (~786 GB/s bidirectional). NVLink is 9–112×
faster and bypasses the CPU/PCIe entirely. NCCL automatically prioritizes NVLink when available. NVLink SHARP (NVLS)
provides hardware-accelerated collectives, boosting all-reduce performance by 1.3× to 480 GB/s. However, all-to-all
operations (340 GB/s) don’t benefit from NVLS acceleration.
In distributed training, GPUs must frequently exchange gradients, weights, and activations, often gigabytes of data per
iteration. Transferring these huge amounts of data requires careful handling of communication. While the H100’s internal
HBM can read at about 3 TB/s, accidentally using the wrong flags can completely tank your GPU-to-GPU communication
bandwidth.
$ numactl --hardware
1
node distances:
2
node   0   1 
3
  0:  10  32 
4
  1:  32  10 
5

Let’s see why by examining all the ways GPUs on the same node can communicate (and all the flags you should—or should
not!—set).
Through the CPU
The naive approach uses host memory (SHM): Data travels from GPU1 through the PCIe switch to the CPU, into host
memory, back through the CPU, through the PCIe switch again, and finally to GPU2. This can be achieved (although it’s not
recommended) by setting NCCL_P2P_DISABLE=1 and FI_PROVIDER=tcp in NCCL’s environment variables. When this mode
is activated, you can verify that it’s working by setting NCCL_DEBUG=INFO , which will show messages like:
This roundabout path involves multiple memory copies and saturates both the PCIe and CPU memory buses, causing
congestion. In our topology, where four H100s share the same CPU memory buses, this congestion becomes even more
problematic when multiple GPUs attempt simultaneous communication, as they compete for the same limited CPU memory
bandwidth.
With this CPU-mediated approach, we’re fundamentally bottlenecked by the PCIe Gen4 x8 link at ~16 GB/s between the
CPU and PCIe switch. Fortunately, there’s a better way for our GPUs to communicate without involving the CPU: GPUDirect
NCCL INFO Channel 00 : 1[1] -> 0[0] via SHM/direct/direct
1
2
GPU-to-GPU communication path through CPU and main memory, showing the inefficient round trip through the PCIe switch and CPU.

RDMA.
Through libfabric EFA
GPUDirect RDMA (Remote Direct Memory Access), or GDRDMA , is a technology that enables direct communication between
NVIDIA GPUs by allowing direct access to GPU memory. This eliminates the need for data to pass through the system CPU
and avoids buffer copies via system memory, resulting in up to 10× better performance compared to traditional CPU-
mediated transfers. GPUDirect RDMA works over PCIe to enable fast GPU-to-GPU communication both within a node (as
described here) and across nodes using network interface cards (NICs) with RDMA capabilities, as we’ll see in a future
section.
Let’s take a closer look at the relevant part of the system topology and the EFA link details:
To make sure you’re enabling GPUDirect RDMA over EFA, you should set the FI_PROVIDER=efa and NCCL_P2P_DISABLE=1
environment variables. When this mode is activated, you can verify it that it’s working by setting NCCL_DEBUG=INFO , which
will show messages like:
Looking back at our topology diagram, we can see that each PCIe switch has four EFA (Elastic Fabric Adapter) NICs,
meaning each GPU has access to four EFA adapters. EFA is AWS’s custom high-performance network interface for cloud
instances, designed to provide low-latency, high-throughput inter-instance communication. On P5 instances, EFA exposes a
libfabric interface (a communication API for high-performance applications) that provides access to RDMA-like capabilities
such as GPUDirect RDMA for direct GPU-to-GPU communication across nodes.
$ lstopo -v
1
...
2
## We can see 4 such EFA devices per each PCIe switch
3
PCIBridge L#8 (busid=0000:46:01.0 id=1d0f:0200 class=0604(PCIBridge) link=15.75GB/s buses=0000:
[4f-4f] PCIVendor="Amazon.com, Inc.")
4
PCI L#6 (busid=0000:4f:00.0 id=1d0f:efa1 class=0200(Ethernet) link=15.75GB/s PCISlot=82-1 
PCIVendor="Amazon.com, Inc.")
5
    OpenFabrics L#4 (NodeGUID=cd77:f833:0000:1001 SysImageGUID=0000:0000:0000:0000 Port1State=4 
Port1LID=0x0 Port1LMC=1 Port1GID0=fe80:0000:0000:0000:14b0:33ff:fef8:77cd) "rdmap79s0"
6
...
7
8
$ fi_info --verbose
9
        fi_link_attr:
10
            address: EFA-fe80::14b0:33ff:fef8:77cd
11
            mtu: 8760            # maximum packet size is 8760 bytes
12
            speed: 100000000000  # each EFA link provides 100 Gbps of bandwidth
13
            state: FI_LINK_UP
14
            network_type: Ethernet
15
16
Each EFA link provides 100 Gbps (12.5 GB/s) of bandwidth. With 4 EFA NICs per GPU and 8 GPUs per node, this gives an
aggregate bandwidth of 100 × 4 × 8 = 3,200 Gbps (400 GB/s) per node.
NCCL INFO Channel 01/1 : 1[1] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA/Shared
1
2

While GPUDirect RDMA over EFA provides significant improvements over CPU-mediated transfers, achieving around 50 GB/s
with four EFA cards per GPU, can we do even better? This is where NVLink comes into play.
Through NVLinkNVLink** is NVIDIA’s high-speed, direct GPU-to-GPU interconnect technology that enables fast multi-GPU
communication within servers. The following table compares NVLink bandwidth across generations (showing theoretical
specifications). The H100 employs NVLink 4.0, providing 900 GB/s bidirectional bandwidth per GPU through 18 links each
operating at 50 GB/s bidirectional.
NVLink 2.0 (Volta)NVLink 3.0 (Ampere)NVLink 4.0 (Hopper)NVLink 5.0 (Blackwell)
Bandwidth 300 GB/s 600 GB/s 900 GB/s
In the DGX H100 architecture, four third-generation NVSwitches connect the eight GPUs using a layered topology where
each GPU connects with 5 + 4 + 4 + 5 links across the switches. This configuration ensures multiple direct paths between
any GPU pair with a constant hop count of just one NVSwitch, resulting in 3.6 TB/s total bidirectional NVLink network
bandwidth.
By default, NCCL prioritizes NVLink for intranode GPU communication when available, as it provides the lowest-latency and
highest-bandwidth path between GPUs on the same machine. To ensure it’s used, avoid disabling NVLink support via NCCL
environment variables—you don’t want to inadvertently prevent its use by not setting your flags properly!
NVLink enables direct GPU-to-GPU memory access without involving the CPU or system memory. When it’s unavailable,
NCCL falls back to GPUDirect P2P over PCIe, or uses the shared memory (SHM) transport when inter-socket PCIe transfers
would be suboptimal.
To verify that NVLink is being used, set NCCL_DEBUG=INFO and look for messages like:
NCCL INFO Channel 00/1 : 0[0] -> 1[1] via P2P/CUMEM
1
2
GPU-to-GPU communication path through libfabric EFA. Note that this is less efficient for intranode communications than using NVLink.

The following diagram illustrates the direct path that data takes when using NVLink.
With NVLink 4.0’s theoretical bandwidth of 900 GB/s compared to EFA’s ~50 GB/s, we expect an 18× advantage for
intranode communication. To validate this in practice, we ran NCCL’s SendRecv performance test to measure actual
bandwidth across different communication paths:
$ FI_PROVIDER=XXX NCCL_P2P_DISABLE=X sendrecv_perf -b 8 -e 8G -f 2 -g 1 -c 1 -n 100
1
GPU-to-GPU communication path through NVLink.

This shows without a doubt how much more efficient NVLink is: It achieves 364.93 GB/s compared to EFA’s 38.16 GB/s
(9× faster, or 18× bidirectional) and the CPU baseline’s 3.24 GB/s (112.6× faster). These measurements confirm why
NCCL prioritizes NVLink for intranode GPU communication, but for one more test, let’s use nvbandwidth to measure
bidirectional bandwidth between all GPU pairs using simultaneous copies in both directions:
The measured bidirectional bandwidth of 786 GB/s represents 85% of NVLink 4.0’s theoretical 900 GB/s specification.
Using NVLink for GPU-to-GPU communication has bypassed the CPU bottleneck entirely!
$ ./nvbandwidth -t device_to_device_bidirectional_memcpy_write_ce -b <message_size> -i 5
1
memcpy CE GPU(row) <-> GPU(column) Total bandwidth (GB/s)
2
           0         1         2         3         4         5         6         7
3
 0       N/A    785.81    785.92    785.90    785.92    785.78    785.92    785.90
4
 1    785.83       N/A    785.87    785.83    785.98    785.90    786.05    785.94
5
 2    785.87    785.89       N/A    785.83    785.96    785.83    785.96    786.03
6
 3    785.89    785.85    785.90       N/A    785.96    785.89    785.90    785.96
7
 4    785.87    785.96    785.92    786.01       N/A    785.98    786.14    786.08
8
 5    785.81    785.92    785.85    785.89    785.89       N/A    786.10    786.03
9
 6    785.94    785.92    785.99    785.99    786.10    786.05       N/A    786.07
10
 7    785.94    786.07    785.99    786.01    786.05    786.05    786.14       N/A
11
12
SUM device_to_device_bidirectional_memcpy_write_ce_total 44013.06
13
14
GPU-to-GPU measured bandwidth with NCCL's SendRecv test (H100 GPUs, 1 Node, 2 GPUs)

But how does this translate to collective communication patterns? Let’s measure all-reduce performance within a single
node with the all_reduce_perf benchmark from NCCL Tests:
But wait… We’re achieving 480 GB/s, which exceeds the theoretical unidirectional bandwidth of 450 GB/s for NVLink 4.0.
What is this sorcery, and how is it possible?
Diving a bit into the docs, it seems the answer lies in NVLink SHARP (NVLS), NVIDIA’s hardware-accelerated collective
operations technology. This provides an approximately 1.3× speedup for all-reduce operations on a single node with H100
GPUs!
$ ./all_reduce_perf -b 8 -e 16G -f 2 -g 1 -c 1 -n 100
1
2
NCCL's all-reduce performance test (intranode)

For technical details on how NVSwitches enable these hardware-accelerated collective operations, see the NVSwitch
architecture presentation.
Can they help in other places too? Let’s examine all-to-all performance:
$ ./all_to_all_perf -b 8 -e 16G -f 2 -g 1 -c 1 -n 100
1
2


⚡Advanced kernel optimization
Some optimized kernels separate NVLink communication from compute by assigning dedicated warps to handle transfers.
For example, ThunderKittens uses a warp-level design where specific warps issue NVLink transfers and wait for completion,
while other warps continue compute operations. This fine-grained overlap of SM compute and NVLink communication can
hide most inter-GPU communication latency. For implementation details, see the ThunderKittens blog post on multi-GPU
kernels.
While NVLink provides exceptional bandwidth within a single node, training frontier models requires scaling across multiple
nodes.
This introduces a new potential bottleneck: the internode network interconnect, which operates at significantly lower
bandwidths than NVLink.
GPU-TO-GPU INTERNODE COMMUNICATION
TL;DR: Multi-node GPU communication uses high-speed networks like InfiniBand (400 Gbps) or RoCE (100 Gbps). All-reduce
scales well (320–350 GB/s, stable across nodes), enabling massive training clusters. All-to-all degrades more sharply due
We achieve 340 GB/s for all-to-all operations, which aligns with published benchmarks showing similar performance
characteristics for H100 systems with NVLink 4.0. Unlike all-reduce, all-to-all operations don’t benefit from NVLS hardware
acceleration, which explains why we see 340 GB/s here compared to the 480 GB/s achieved with all-reduce. The all-to-all
pattern requires more complex point-to-point data exchanges between all GPU pairs, relying purely on NVLink’s base
bandwidth rather than NVSwitch’s collective acceleration features.
NCCL's all-to-all performance test (intranode)

to algorithm complexity: Latency jumps from ~13 μs intranode to 55 μs+ internode. For MoE workloads requiring frequent all-
to-all operations, NVSHMEM offers asynchronous GPU-initiated communication with significantly better performance than
CPU-orchestrated transfers.
As models scale beyond what a single node can accommodate, training requires distributing computation across multiple
nodes connected via high-speed networks. Before diving into the benchmarks, let’s look at the three key networking
technologies for connecting nodes that you’ll encounter in multi-node GPU clusters:
Ethernet has evolved from 1 Gbps to 100+ Gbps speeds and remains widely used in HPC and datacenter clusters.
RoCE (RDMA over Converged Ethernet) brings RDMA capabilities to Ethernet networks, using Explicit Congestion
Notification (ECN) for congestion control instead of traditional TCP mechanisms.
InfiniBand is NVIDIA’s industry-standard switch fabric, providing up to 400 Gbps bandwidth and sub-microsecond latency
with RDMA support that enables direct GPU-to-GPU memory access while bypassing the host CPU through GPUDirect
RDMA.
Li et al. (2022) provide further details, but as a summary:
Technology Ethernet (25–100 Gbps)Ethernet (200–400 Gbps)RoCE InfiniBand
Manufacturer Many Many ManyNVIDIA/Mellanox
Unidirectional Bandwidth (Gbps)25–100 200–400 100 400
End-to-End Latency (μs)10–30 N/A ~1 &lt;1
RDMA No No Yes Yes
On AWS P5 instances, Elastic Fabric Adapter (EFA) serves as the network interface (NIC). Each GPU connects to four 100-
Gbps EFA NICs via PCIe Gen5 x16 links, as we saw earlier.

When GPUs and network cards are connected to the same PCIe switch, as illustrated here, GPUDirect RDMA enables their
communication to occur solely through that switch. This setup allows for full utilization of the PCIe Gen5 x16 bandwidth and
avoids involving other PCIe switches or the CPU memory bus.
Theoretically, 8 PCIe switches per node × 4 EFA NICs per switch × 100 Gbps per EFA NIC gives 3,200 Gbps(400 GB/s)**
of bandwidth, which is the bandwidth we find in AWS’s P5 specs. But does this hold in practice? Let’s find out by running
the same benchmarks as before but across different nodes!
Bandwidth Analysis
Internode GPU-to-GPU communication path through libfabric EFA.

Point-to-point send/receive operations achieve around 42–43 GB/s for 2–4 nodes, but this drops to approximately 21 GB/s
for 5+ nodes. This performance degradation occurs because NCCL automatically reduces the number of point-to-point
channels per peer from 2 to 1 when scaling beyond 4 nodes, effectively halving the available bandwidth utilization, while the
theoretical maximum remains ~50 GB/s (4 EFA NICs × 12.5 GB/s each). We successfully managed to restore the full
throughput for this test on 5+ nodes by setting NCCL_NCHANNELS_PER_NET_PEER=2 , although this flag should be used with
caution as it may degrade all-to-all performance, for example (see GitHub issue #1272 for details).
The all-reduce operation demonstrates excellent performance within a single node, achieving 480 GB/s of bus bandwidth.
When scaling to 2 nodes, bandwidth remains nearly identical at 479 GB/s, after which it stabilizes at around 320–350
GB/s for 3–16 nodes. This pattern reveals an important characteristic: While there’s an initial drop when crossing node
boundaries due to the transition from NVLink to the internode network fabric, the bandwidth then scales almost constantly
as we add more nodes.
💡Scaling all-reduce across nodes
This near-constant scaling behavior beyond two nodes is actually quite encouraging for large-scale training. The relatively
stable 320–350 GB/s across 3–16 nodes suggests that parallelism strategies relying on all-reduce operations (for example,
in data parallelism) can scale to hundreds or even thousands of GPUs without significant per-GPU bandwidth degradation.
This logarithmic scaling characteristic is typical of well-designed multi-tier network topologies using 8-rail optimized fat trees,
where each of the 8 GPUs connects to a separate switch rail to maximize bisection bandwidth. Modern frontier training
clusters routinely operate at 100,000+ GPUs, and this stable scaling behavior is what makes such massive deployments
feasible.
When working with different bandwidth links (NVLink within nodes vs. internode network), consider adapting your parallelism
strategy to each bandwidth tier to fully utilize all available bandwidth. See the Ultra-Scale Playbook for detailed guidance on
optimizing parallelism configurations for heterogeneous network topologies.
The all-to-all operation shows more dramatic scaling challenges: Starting at 344 GB/s for a single node, bandwidth drops to
81 GB/s at two nodes and continues declining to approximately 45–58 GB/s for larger clusters. This steeper degradation
reflects the all-to-all pattern’s intensive network demands, where each GPU must communicate with every other GPU across
nodes, creating significantly more network congestion than all-reduce operations.
Bandwidth scaling of collective operations across different numbers of nodes on our AWS P5 instances, using recommendations from
aws-samples/awsome-distributed-training.

Latency Analysis
Latency measurements reveal the fundamental cost of crossing node boundaries. Send/receive operations maintain
relatively stable latencies of 40–53 μs across all multi-node configurations, demonstrating that point-to-point
communication latency is primarily determined by the base network round-trip time rather than cluster size, though some
variation suggests network topology and routing effects still play a role.
All-reduce operations show minimal latency of 12.9 μs within a single node, but this jumps to 55.5 μs for 2 nodes and
continues increasing nearly linearly with cluster size, reaching 235 μs at 16 nodes. This progression reflects both the
increased communication distance and the growing complexity of the reduction tree across more nodes.
All-to-all operations exhibit similar trends, starting at 7.6 μs for single-node communication but climbing to 60 μs at 2
nodes and reaching 621 μs at 16 nodes. The superlinear growth in latency for all-to-all operations indicates that network
congestion and coordination overhead compound as more nodes participate in the collective.
🚀NVSHMEM for optimized GPU communication
With the rise of MoE architectures, which require frequent all-to-all communication for expert routing, optimized GPU
communication libraries have become increasingly critical.
NVSHMEM is gaining significant traction as a high-performance communication library that combines the memory of multiple
GPUs into a partitioned global address space (PGAS). Unlike traditional MPI-based approaches that rely on CPU-orchestrated
data transfers, NVSHMEM enables asynchronous, GPU-initiated operations that eliminate CPU–GPU synchronization
overhead.
NVSHMEM offers several key advantages for GPU communication: Through technologies like GPUDirect Async, GPUs can
bypass the CPU entirely when issuing internode communication, achieving up to 9.5× higher throughput for small messages
(&lt;1 KiB). This is particularly beneficial for collective operations that require intensive network communication patterns.
The library currently supports InfiniBand/RoCE with Mellanox adapters (CX-4 or later), Slingshot-11 (libfabric CXI), and
Amazon EFA (libfabric EFA). For applications requiring strong scaling with fine-grained communication, NVSHMEM’s low-
Latency scaling of collective operations across different numbers of nodes on our AWS P5 instances, using recommendations from [aws-
samples/awsome-distributed-training](https://github.com/aws-samples/awsome-distributed-training/blob/main/micro-benchmarks/nccl-
tests/slurm/nccl-tests-container.sbatch).

overhead, one-sided communication primitives can significantly improve performance compared to traditional CPU proxy
methods.
Learn more in the NVSHMEM documentation and this detailed NVIDIA blog post on GPUDirect Async.
When bandwidth measurements fall short of expectations, several factors could be limiting performance. Understanding
these potential bottlenecks is essential for achieving optimal interconnect utilization.
TROUBLESHOOTING INTERCONNECT
If you’re experiencing lower than expected bandwidth, systematically check the following areas.
Library Versions
Outdated NCCL, EFA, or CUDA libraries may lack critical performance optimizations or bug fixes. Always verify that you’re
running recent, compatible versions of all communication libraries. AWS, for example, regularly updates the Deep Learning
AMIs with library versions optimized for their hardware. It’s also recommended to log these library versions for important
experiments.
CPU Affinity Configuration
Improper CPU affinity settings can significantly impact NCCL performance by causing unnecessary cross-NUMA traffic. Each
GPU should be bound to CPUs on the same NUMA node to minimize memory access latency. GitHub issue #1017
demonstrates how using NCCL_IGNORE_CPU_AFFINITY=1 and --cpu-bind none helped reduce container latency
significantly in practice. You can read more about it in the Enterprise Support Portal.
Network Topology and Placement
Understanding your network topology is crucial for diagnosing performance issues. Cloud placement groups, while helpful,
don’t guarantee minimal network hops between instances. In modern datacenter fat-tree topologies, instances placed under
different top-level switches will experience higher latency and potentially lower bandwidth due to additional network hops in
the routing path.
For AWS EC2 users, the Instance Topology API provides valuable visibility into network node placement. Instances sharing
the same network node at the bottom layer (directly connected to the instance) are physically closest and will achieve the
lowest-latency communication.
Minimizing network hops between communicating nodes directly translates to better interconnect performance. For small-
scale experiments and ablations, ensuring your instances are co-located on the same network switch can make a
measurable difference in both latency and bandwidth utilization.
Correct Environment Variables
Missing or incorrect environment variables for your network adapter can severely limit bandwidth utilization. Communication
libraries like NCCL rely on specific configuration flags to enable optimal performance features such as adaptive routing,
GPU-initiated transfers, and proper buffer sizing.
Level 1nn-4aed5...
Level 2nn-48b0a...
Level 3nn-d2ad4... Level 3nn-2d36a... Level 3nn-65fc9... Level 3nn-fbb73... Level 3nn-65290... Level 3nn-27373...Level 3nn-5adeb...Level 3nn-dbe4f... Level 3nn-fde84... Level 3nn-3c5c0... Level 3nn-94247... Level 3nn-8f3c1...
ID: 02e1b4f9ip-26-0-171-102p5.48xlarge ID: 05388ebfip-26-0-171-230p5.48xlarge ID: 03bfac00ip-26-0-168-30p5.48xlarge ID: d92bab46ip-26-0-168-95p5.48xlarge ID: 97a542e4ip-26-0-163-158p5.48xlarge ID: e2c87e43ip-26-0-167-9p5.48xlarge ID: afa887eaip-26-0-168-120p5.48xlarge ID: 66c12e70ip-26-0-167-177p5.48xlarge ID: 9412bdf3ip-26-0-168-52p5.48xlarge ID: 87bd4dc8ip-26-0-167-111p5.48xlarge ID: b001549bip-26-0-166-244p5.48xlarge ID: 10ed8172ip-26-0-107-245p5.48xlarge ID: 7c1d0a09ip-26-0-168-238p5.48xlarge ID: 925ce932ip-26-0-167-217p5.48xlarge ID: c9bc34dbip-26-0-171-168p5.48xlarge ID: 328d5d04ip-26-0-167-127p5.48xlarge
N e t w o r k  t o p o l o g y  v i s u a l i z a t i o n  s h o w i n g  i n s t a n c e  p l a c e m e n t .
Level 1nn-4aed5...
Level 2nn-48b0a...
Level 3nn-d2ad4... Level 3nn-2d36a... Level 3nn-65fc9... Level 3nn-fbb73... Level 3nn-65290... Level 3nn-27373...Level 3nn-5adeb...Level 3nn-dbe4f... Level 3nn-fde84... Level 3nn-3c5c0... Level 3nn-94247... Level 3nn-8f3c1...
ID: 02e1b4f9ip-26-0-171-102p5.48xlarge ID: 05388ebfip-26-0-171-230p5.48xlarge ID: 03bfac00ip-26-0-168-30p5.48xlarge ID: d92bab46ip-26-0-168-95p5.48xlarge ID: 97a542e4ip-26-0-163-158p5.48xlarge ID: e2c87e43ip-26-0-167-9p5.48xlarge ID: afa887eaip-26-0-168-120p5.48xlarge ID: 66c12e70ip-26-0-167-177p5.48xlarge ID: 9412bdf3ip-26-0-168-52p5.48xlarge ID: 87bd4dc8ip-26-0-167-111p5.48xlarge ID: b001549bip-26-0-166-244p5.48xlarge ID: 10ed8172ip-26-0-107-245p5.48xlarge ID: 7c1d0a09ip-26-0-168-238p5.48xlarge ID: 925ce932ip-26-0-167-217p5.48xlarge ID: c9bc34dbip-26-0-171-168p5.48xlarge ID: 328d5d04ip-26-0-167-127p5.48xlarge
N e t w o r k  t o p o l o g y  v i s u a l i z a t i o n  s h o w i n g  i n s t a n c e  p l a c e m e n t .

For example, when using AWS EFA, ensure you’re setting the recommended NCCL and EFA environment variables for your
instance type. The AWS EFA cheatsheet provides comprehensive guidance on optimal flag configurations for different
scenarios.
Container-Specific Considerations
When using containers (Docker/Enroot), several configuration steps are critical for optimal NCCL performance:
Shared and pinned memory: Docker containers default to limited shared and pinned memory resources. Launch
containers with -shm-size=1g --ulimit memlock=-1 to prevent initialization failures.
NUMA support: Docker disables NUMA support by default, which can prevent cuMem host allocations from working
correctly. Enable NUMA support by invoking Docker with -cap-add SYS_NICE .
PCI topology discovery: Ensure /sys is properly mounted to allow NCCL to discover the PCI topology of GPUs and
network cards. Having /sys expose a virtual PCI topology can result in suboptimal performance.
Now that you know how to debug bottlenecks in GPU–CPU and GPU–GPU communication, let’s have a look at an aspect of
GPU communication that typically gets less attention—communication with the storage layer!
GPU-TO-STORAGE COMMUNICATION
TL;DR: GPU–storage I/O impacts training through data loading and checkpointing. GPUDirect Storage (GDS) enables direct
GPU-to-storage transfers, bypassing the CPU for better performance. Even without GDS enabled in our cluster, local NVMe
RAID (8 × 3.5 TB drives in RAID 0) delivers 26.59 GiB/s and 337k IOPS (6.3× faster than network storage), making it ideal
for checkpoints.
The connection between GPUs and storage systems is often overlooked but can significantly impact training efficiency.
During training, GPUs need to continuously read data from storage and periodically write model states back to storage (aka
checkpointing). For modern large-scale training runs, these I/O operations can become bottlenecks if not properly
optimized.
Understanding Storage Topology
The physical connections between GPUs and storage devices follow a similar hierarchical structure to GPU interconnects.
Storage devices connect through PCIe bridges, and understanding this topology helps explain performance characteristics
and potential bottlenecks.
Looking at the system topology from lstopo , we can see how NVMe drives connect to the system. In our P5 instance, we
have one NVMe SSD per GPU:
A natural question would be whether GPUs can directly access NVMe drives without involving the CPU. The answer is yes,
through GPUDirect Storage (GDS) .
GPUDirect Storage, part of NVIDIA’s GPUDirect family of technologies, enables a direct data path between storage (local
NVMe or remote NVMe-oF) and GPU memory. It eliminates unnecessary memory copies through CPU bounce buffers by
PCIBridge L#13 (busid=0000:46:01.5 id=1d0f:0200 class=0604(PCIBridge) link=15.75GB/s buses=0000:
[54-54] PCIVendor="Amazon.com, Inc.")
1
PCI L#11 (busid=0000:54:00.0 id=1d0f:cd01 class=0108(NVMExp) link=15.75GB/s PCISlot=87-1 
PCIVendor="Amazon.com, Inc." PCIDevice="NVMe SSD Controller")
2
    Block(Disk) L#9 (Size=3710937500 SectorSize=512 LinuxDeviceID=259:2 Model="Amazon EC2 NVMe 
Instance Storage" Revision=0 SerialNumber=AWS110C9F44F9A530351) "nvme1n1"
3
4

allowing the direct memory access (DMA) engine near the storage controller to move data directly into or out of GPU
memory. This reduces CPU overhead, decreases latency, and significantly improves I/O performance for data-intensive
workloads like training on large multimodal datasets.
To verify whether GPUDirect Storage is properly configured on your system, you can check the GDS configuration file and
use the provided diagnostic tools:
The NVMe: Supported line informs us that GDS is currently configured to work for NVMe drives, while the Unsupported
flag indicates that it is not configured for any other storage types. If GDS is not properly configured for your storage type,
refer to the NVIDIA GPUDirect Storage Benchmarking and Configuration Guide for instructions on modifying the configuration
file at /etc/cufile.json .
Block Storage Devices
To understand the storage devices available on your system, you can use lsblk :
The output shows the block device hierarchy on the system. Key observations here are:
$ /usr/local/cuda/gds/tools/gdscheck.py -p
1
 =====================
2
 DRIVER CONFIGURATION:
3
 =====================
4
 NVMe               : Supported   
5
 NVMeOF             : Unsupported
6
 SCSI               : Unsupported
7
 ScaleFlux CSD      : Unsupported
8
 NVMesh             : Unsupported
9
 DDN EXAScaler      : Unsupported
10
 IBM Spectrum Scale : Unsupported
11
 NFS                : Unsupported
12
 BeeGFS             : Unsupported
13
 WekaFS             : Unsupported
14
 Userspace RDMA     : Unsupported
15
 --Mellanox PeerDirect : Enabled
16
 --rdma library        : Not Loaded (libcufile_rdma.so)
17
 --rdma devices        : Not configured
18
 --rdma_device_status  : Up: 0 Down: 0
19
 =====================
20
21
$ lsblk --fs -M
1
    NAME        FSTYPE            LABEL                   UUID                                 
FSAVAIL FSUSE% MOUNTPOINT
2
...
3
    nvme0n1
4
    └─nvme0n1p1 ext4              cloudimg-rootfs         24ec7991-cb5c-4fab-99e5-52c45690ba30  
189.7G    35% /
5
┌ ┈▶  nvme1n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
6
├ ┈▶  nvme2n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
7
├ ┈▶  nvme3n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
8
├ ┈▶  nvme8n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
9
├ ┈▶  nvme5n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
10
├ ┈▶  nvme4n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
11
├ ┈▶  nvme6n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
12
└┬ ▶  nvme7n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
13
 └ ┈┈ md0         xfs                                       dddb6849-e5b5-4828-9034-96da65da27f0   
27.5T     1% /scratch
14
15

nvme0n1p1 is the root Amazon EBS filesystem mounted at / , using 35% of its full 291 GB capacity.
Eight NVMe drives ( nvme1n1 through nvme8n1 ) are configured as a RAID array named MY_RAID .
The RAID array is exposed as /dev/md0 , formatted with XFS, and mounted at /scratch with 28 TB available (8 × 3.5
TB).
Network Storage
In addition to local NVMe storage, the system has access to network-attached storage systems:
This output shows:
The local NVMe RAID array ( /scratch ) provides the fastest I/O performance, while the network filesystems offer larger
capacity for shared data storage.
💾Storage technology refresher
RAID (Redundant Array of Independent Disks): Combines multiple drives to improve performance and/or reliability through
data striping, parity, or mirroring.
NVMe (Non-Volatile Memory Express): A high-performance storage protocol for SSDs that connects directly to PCIe, delivering
higher throughput and lower latency than SATA/SAS.
WekaFS: A high-performance parallel filesystem designed for AI/ML workloads, providing low-latency access and high
throughput across multiple nodes.
FSx Lustre: A parallel filesystem designed for high-performance computing that separates metadata and data services
across different servers to enable parallel access. While effective for large files, it can struggle with metadata-intensive
AI/ML workloads involving many small files.
Benchmarking Storage Bandwidth
To understand the performance characteristics of each storage system, we can benchmark their read/write speeds using
GPUDirect Storage. Here’s a comprehensive parametric benchmark script that tests various configurations:
The arrows (┈▶) indicate that multiple NVMe devices are members of the same RAID array, which then combines into the
single md0 device.
$ df -h
1
Filesystem                                         Size  Used Avail Use% Mounted on
2
/dev/root                                          291G  101G  190G  35% /
3
weka-hopper.hpc.internal.huggingface.tech/default  393T  263T  131T  67% /fsx
4
10.53.83.155@tcp:/fg7ntbev                         4.5T  2.9T  1.7T  63% /admin
5
/dev/md0                                            28T  206G   28T   1% /scratch
6
7
/dev/root (291 GB Amazon EBS) is the root filesystem, at 35% capacity.
/fsx (393 TB WekaFS) is 67% full with 131 TB available.
/admin (4.5 TB FSx Lustre) is 63% full with 1.7 TB available.
/dev/md0 (28 TB local NVMe RAID) is only 1% full with 28 TB available at /scratch . This is our RAID array of 8 × 3.5
TB NVMe instance-store SSDs.

The benchmark evaluates storage system performance in terms of throughput, latency, and IOPS, as well as:
Scalability: How performance changes with different thread counts and I/O sizes. This reveals optimal configurations for
different workload patterns:
Small I/O sizes (64k to 256k) typically maximize IOPS but may not saturate bandwidth.
Large I/O sizes (2M to 8M) typically maximize throughput but reduce IOPS.
Thread count affects both: More threads can increase total IOPS and throughput up to hardware limits.
Transfer method efficiency: Comparing GPU_DIRECT vs. CPU_GPU vs. CPUONLY shows the benefit of bypassing CPU
memory:
GPU_DIRECT uses RDMA to transfer data directly to GPU memory, bypassing the CPU entirely (lowest latency, highest
efficiency, best IOPS for small operations).
CPU_GPU is the traditional path, where data goes to CPU memory first and is then copied to the GPU (adds CPU
overhead and memory bandwidth contention, reduces effective IOPS).
CPUONLY is the baseline CPU-only I/O without GPU involvement.
📊IOPS
IOPS is the number of individual I/O operations completed per second, calculated as ops / total_time from the gdsio
output. IOPS is particularly important for:
Random access patterns with small I/O sizes
Workloads with many small files or scattered data access
Database-like operations where latency per operation matters more than raw bandwidth
Higher IOPS indicates better ability to handle concurrent, fine-grained data access.
gdsio -f /<disk_path>/gds_test.dat -d 0 -w <n_threads> -s 10G -i <io_size> -x 1 -I 1 -T 10
1
2

The benchmarks reveal dramatic performance differences across our four storage systems:
/scratch** (local NVMe RAID) dominates with 26.59 GiB/s throughput and 337k IOPS , making it 6.3× faster than FSx
for throughput and 6.6× better for IOPS. This local RAID array of 8 × 3.5 TB NVMe drives delivers the lowest latency
(190 μs at peak IOPS) and scales exceptionally well with thread count, achieving peak performance at 64 threads with
1M I/O sizes for throughput.
/fsx** (WekaFS) provides solid network storage performance at 4.21 GiB/s and 51k IOPS , making it the best choice
for shared data that needs reasonable performance. FSx achieves its best throughput (4.21 GiB/s) using CPUONLY
transfer, while its best IOPS (51k) uses GPU_DIRECT transfer.
Benchmark results comparing storage system performance across varying thread counts and I/O sizes. The heatmaps visualize
throughput (GiB/s) and IOPS patterns, revealing optimal configurations for each storage tier. Note: GPUDirect Storage (GDS) is not
currently supported in this cluster configuration.

/admin(FSx Lustre)** and /root(EBS)** show similar modest performance at around 1.1 GiB/s throughput but differ
significantly in IOPS capability. /admin achieves its peak throughput (1.13 GiB/s) with GPU_DIRECT transfer and peaks
at 17k IOPS with CPU_GPU transfer (24× better than /root ), making it more suitable for workloads with many small
operations. The poor IOPS performance of \root (730) confirms it’s best suited for large sequential operations only.
Note that GPUDirect Storage is not currently enabled in our cluster, which explains why GPU_DIRECT results for NVMe
storage ( /scratch and /root ) underperform compared to CPUONLY transfers. With GDS properly configured, we would
expect GPU_DIRECT to show significant advantages for direct GPU-to-storage transfers, particularly for the high-performance
NVMe arrays.
Across all storage types, maximum throughput occurs at 1M I/O sizes, while maximum IOPS occurs at the smallest tested
size (64k). This classic trade-off means choosing between raw bandwidth (large I/O) and operation concurrency (small I/O)
based on workload characteristics. For ML training with large checkpoint files, the 1–8M range on /scratch provides
optimal performance.
SUMMARY
At this point, you should have a comprehensive understanding of the storage hierarchy and how different components
interact in your training infrastructure. But here’s the key insight we hope you take home: Identifying bottlenecks is what
separates theoretical knowledge from practical optimization.
In this chapter, we’ve measured actual bandwidths at every level of the stack: HBM3’s 3 TB/s within a single GPU, NVLink’s
786 GB/s between GPUs in a node, PCIe Gen4 x8’s 14.2 GB/s for CPU–GPU transfers, the internode network’s 42 GB/s
for point-to-point communication, and storage systems ranging from 26.59 GiB/s (local NVMe) down to 1.1 GiB/s (shared
filesystems). These measurements reveal where your training pipeline will slow down and are essential for achieving high
MFU.
However, raw bandwidth numbers alone don’t tell the complete story. Modern training systems can overlap computation with
communication , effectively hiding communication costs behind compute operations. This parallelization helps alleviate the
bottleneck even when interconnects are slow. For detailed strategies on overlapping compute and communication to
maximize throughput, see the Ultra-Scale Playbook.
The following diagram synthesizes all our benchmarked measurements into a single view, showing how bandwidth
decreases dramatically as we move further from the GPU:

Now that we know how to identify bottlenecks in our hardware and software setup, let’s see how we can go one step further
and ensure we have a resilient system that can run stably for months.
Building Resilient Training Systems
Having fast hardware is just the baseline requirement for good and stable infrastructure for LLM training. To go from a
training amateur to a professional, you need to think beyond raw speed and focus on the less glamorous but critical
infrastructure pieces that make the entire training experience smoother and ensure minimal downtime.
In this section, we’ll shift our focus from hardware and software optimization to production readiness : building systems
robust enough to survive inevitable failures, automated enough to run without constant babysitting, and flexible enough to
adapt when things go wrong.
NODE HEALTH MONITORING AND REPLACEMENT
Having enough fast GPUs is important for training, but since LLM trainings run for weeks or months rather than a day or two,
tracking GPU health over time is critical. GPUs that pass initial benchmarks can start to show thermal throttling, memory
errors, or performance degradation during extended training runs. In this section, we will share how we approach this
challenge and the tools we use.
Before launching SmolLM3, we ran comprehensive GPU diagnostics. We used GPU Fryer, an internal tool that stress-tests
GPUs for thermal throttling, memory errors, and performance anomalies. We also ran NVIDIA’s DCGM Diagnostics, a widely
used tool for validating GPU hardware, monitoring performance, and identifying root causes of failures or power anomalies
through deep diagnostic tests covering compute, PCIe connectivity, memory integrity, and thermal stability. These up-front
tests caught two problematic GPUs that would have caused issues during training.
The following table shows what can be tested with DCGM Diagnostics:


Test Level Duration Key Tests
r1 (Short) Seconds PCIe/NVLink, GPU Memory, Memory BW
r2 (Medium) < 2 mins + Diagnostics, Targeted Stress
r3 (Long) < 30 mins + Targeted Power, NVBandwidth, Memory Stress
r4 (Extra Long) 1–2 hours + Input EDPp (all tests)
Source: NVIDIA DCGM Diagnostics documentation
Here’s an example of our results:
$ dcgmi diag -r 2 -v -d VERB
1
Successfully ran diagnostic for group.
2
+---------------------------+------------------------------------------------+
3
| Diagnostic | Result |
4
+===========================+================================================+
5
| -----  Metadata  ----------+------------------------------------------------ |
6
| DCGM Version | 3.3.1 |
7
| Driver Version Detected | 575.57.08 |
8
| GPU Device IDs Detected | 2330,2330,2330,2330,2330,2330,2330,2330 |
9
| -----  Deployment  --------+------------------------------------------------ |
10
| Denylist | Pass |
11
| NVML Library | Pass |
12
| CUDA Main Library | Pass |
13
| Permissions and OS Blocks | Pass |
14
| Persistence Mode | Pass |
15
| Environment Variables | Pass |
16
| Page Retirement/Row Remap | Pass |
17
| Graphics Processes | Pass |
18
| Inforom | Pass |
19
20
+-----  Integration  -------+------------------------------------------------+
21
| PCIe | Pass - All |
22
| Info | GPU 0 GPU to Host bandwidth:  14.26 GB/s, GPU |
23
| 0 Host to GPU bandwidth:  8.66 GB/s, GPU 0 b |
24
| idirectional bandwidth: 10.91 GB/s, GPU 0 GPU |
25
| to Host latency:  2.085 us, GPU 0 Host to GP |
26
| U latency:  2.484 us, GPU 0 bidirectional lat |
27
| ency:  3.813 us |
28
29
...
30
+-----  Hardware  ----------+------------------------------------------------+
31
| GPU Memory | Pass - All |
32
| Info | GPU 0 Allocated 83892938283 bytes (98.4%) |
33
| Info | GPU 1 Allocated 83892938283 bytes (98.4%) |
34
| Info | GPU 2 Allocated 83892938283 bytes (98.4%) |
35
| Info | GPU 3 Allocated 83892938283 bytes (98.4%) |
36
| Info | GPU 4 Allocated 83892938283 bytes (98.4%) |
37
| Info | GPU 5 Allocated 83892938283 bytes (98.4%) |
38
| Info | GPU 6 Allocated 83892938283 bytes (98.4%) |
39
| Info | GPU 7 Allocated 83892938283 bytes (98.4%) |
40
41
+-----  Stress  ------------+------------------------------------------------+
42
43

As described in “The Training Marathon,” because SmolLM3 was trained on a Slurm managed cluster, we booked a fixed
48-node reservation for the entire run. This setup allowed us to track the health and performance of the exact same nodes
over time. During training, we continuously monitored key metrics across all nodes, such as GPU temperatures, memory
usage, compute utilization, and throughput fluctuations. We used Prometheus to collect DCGM metrics from all GPUs and
visualized them in Grafana dashboards for real-time monitoring. For detailed setup instructions on deploying Prometheus
and Grafana for GPU monitoring on AWS infrastructure, see the example setup guide in the awsome-distributed-training
repo. A Slack bot alerted us when any node showed suspicious behavior, allowing us to proactively replace failing hardware
before it crashed the entire training run.
This multi-layered approach meant any hardware issues became manageable interruptions.
Thermal Reality Check: When GPUs Slow Down
Marketing specs assume perfect cooling, but reality is messier. GPUs automatically reduce clock speeds when they
overheat, cutting performance below theoretical maximums even in well-designed systems.
We monitored the DCGM_FI_DEV_CLOCK_THROTTLE_REASONS metric from DCGM to detect thermal throttling. When this
metric shows nonzero values, the GPUs are automatically reducing clock speeds due to overheating. The following
dashboard shows how these throttling events manifest in practice:
Thermal throttling doesn’t just hurt the affected GPU; it cascades across your entire distributed training setup. During our
testing, we observed how a single throttling node can dramatically impact collective communication performance:
This Grafana dashboard shows thermal throttling events across our GPU cluster. The bars in the bottom panel indicate when GPUs
automatically reduced clock speeds due to overheating.

This chart shows all-reduce bandwidth degrading as we scale from 1 to 16 nodes. Notice the sharp drop after 14 nodes,
from 350 GB/s to 100 GB/s, while we expected the bandwidth to stay above 300 GB/s. This wasn’t a network issue: A
single node with thermal throttling became the bottleneck, forcing all other nodes to wait during gradient synchronization. In
distributed training, you’re only as fast as your slowest node.
👉  Key lesson: Before committing to long training runs, stress-test your hardware using tools like those mentioned earlier
to identify thermal and power limitations. Monitor temperatures continuously using DCGM telemetry and plan for real-world
thermal limits. It’s also good practice to verify that GPU clocks are set to maximum performance. For a deeper dive into why
GPUs can’t sustain their advertised performance due to power constraints, see Horace He’s excellent analysis on power
throttling.
CHECKPOINT MANAGEMENT
Checkpoints are our safety net during long training runs. We save them regularly, for three practical reasons: recovering
from failures, monitoring training progress through evaluation, and sharing intermediate models with the community for
All-reduce bandwidth degradation across nodes during our stress testing. The sharp drop after 14 nodes (from 350 GB/s to 100 GB/s)
was caused by a single thermally throttled GPU, demonstrating how one slow node can bottleneck the entire distributed training pipeline.

research. The recovery aspect matters most. If our run fails, we want to restart from the latest saved checkpoint so we lose
at most the save interval if we resume immediately (e.g., 4 hours of training if we save every 4 hours).
💡Automate your resume process
Try to automate your resume process. On Slurm, for example, you can just use SBATCH --requeue so the job restarts
automatically from the latest checkpoint. That way, you avoid losing time waiting for someone to notice the failure and
manually restart.
There are two important details to keep in mind when implementing your resume mechanism:
Checkpoint saving should happen in the background without impacting training throughput.
Watch your storage. Over a 24-day run, saving every 4 hours means ~144 checkpoints. With large models and optimizer
states, this adds up fast. In our case, we store only one local checkpoint (the latest saved) at a time and offload the
rest to S3 to avoid filling up cluster storage.
A Painful Lesson from the Past
During our first large-scale run (StarCoder-15B), training proceeded smoothly through multiple restarts. Then, on the final
day, we discovered the entire checkpoint folder had been deleted by a leftover rm -rf $CHECKPOINT_PATH command at
the very end of the script from old throughput tests. This destructive command only triggered when the Slurm job actually
finished, which hadn’t happened in previous restarts.
Luckily, we had the checkpoint from the day before saved, so it only cost us one day of retraining. The takeaways were
clear: Never leave destructive commands in production scripts, and automate checkpoint backups immediately after saving
rather than relying on manual intervention.
In our Nanotron trainings, we save checkpoints every 2 hours locally, immediately upload each one to S3, then delete the
local copy once backup is confirmed. On resume, we pull from S3 if the latest checkpoint isn’t available locally. This
approach saves storage, ensures backups are available, and enables quick recovery.
AUTOMATED EVALUATIONS
Running evaluations manually becomes a bottleneck fast. They look simple until you’re doing them repeatedly. Running
benchmarks and tracking and plotting results for every run adds up to significant overhead. The solution? Automate
everything up front!
For SmolLM3, we used LightEval to run evaluations on Nanotron checkpoints. Every saved checkpoint triggered an
evaluation job on the cluster. The results were pushed directly to Weights & Biases or Trackio, so we just opened the
dashboard and watched the curves evolve. This saved us a huge amount of time and kept eval tracking consistent
throughout the run.
If you can automate only one thing in your training setup, automate evaluations.
Finally, let’s take a look at how we can optimize the training layout to maximize the throughput.
Optimizing Training Throughput
Our final infrastructure consideration is optimizing the training layout, or how the model is distributed across the available
GPUs. This brings us to a key question…

HOW MANY GPUS DO WE NEED?
After all this talk about specs and benchmarks, you still need to answer one practical question: How many GPUs should you
actually rent or buy?
Determining the right number of GPUs requires balancing training time, cost, and scaling efficiency. Here’s the framework
we used.
Basic sizing formula:
This formula breaks down the problem into three key components:
Total FLOPs required: The computational work needed to train your model (depends on model size, training tokens, and
architecture)
Per-GPU throughput: How many FLOPs per second each GPU can actually deliver (not the theoretical peak!)
Target training time: How long you’re willing to wait for training to complete
The key insight: You need to estimate  realistic throughput , not rely on peak specs. This means accounting for model
FLOPs utilization (the percentage of theoretical peak performance you actually achieve in practice).
For SmolLM3, our calculation looked like this:
Model size: 3B parameters
Training tokens: 11T tokens
Target training time: ~4 weeks
Expected MFU: 30% (based on similar-scale experiments)
First, we calculate the total FLOPs needed using the standard transformer approximation of  6N** FLOPs per token  (where
N = parameters):
With our expected MFU of 30%, our effective per-GPU throughput becomes:
Plugging this into our sizing formula, we get:
This calculation pointed us toward 375–400 H100s, and we secured 384, a number that aligned well with our parallelism
strategy and gave us a realistic four-week timeline with some buffer for unexpected issues like node failures and restarts.
Why More GPUs Isn’t Always Better: Amdahl’s Law in Action
Here’s a counterintuitive truth:  Adding more GPUs can actually make your training slower. This is where Amdahl’s law
comes into play.
Amdahl’s law states that the speedup from parallelization is fundamentally limited by the serial (non-parallelizable) portion
of your workload. In LLM training, this “serial” portion is primarily  communication overhead : the time spent synchronizing
GPU count=  
Per-GPU throughput×Target training timeTotal FLOPs required
Total FLOPs=6×3×10 params×9 11×10  tokens=12 1.98×10  FLOPs23
Effective throughput=720×10  FLOPs/sec×12 0.30=216×10  FLOPs/sec12
GPU count=  
216×10  FLOPs/sec×4 weeks×604,800 sec/week12
1.98×10  FLOPs23
=  ≈5.23×1020
1.98×1023
379 GPUs

gradients/weights/activations across GPUs that can’t be parallelized away.
The formula is:
If for a small model like ours communication takes 10% of each training step, then no matter how many GPUs you add,
you’ll never get more than a 10× speedup. Worse, as you add more GPUs the communication fraction
often increases, because:
More GPUs = more all-reduce participants = longer synchronization.
Network latency/bandwidth becomes the bottleneck.
Small models can’t hide communication behind compute.
For SmolLM3, we used weak scaling principles: Our global batch size scaled with our GPU count, maintaining roughly 8k
tokens per GPU globally. This kept our communication-to-computation ratio reasonable while maximizing throughput.
FINDING THE OPTIMAL PARALLELISM CONFIGURATION
Once you’ve secured your GPUs, the next challenge is configuring them to actually train efficiently. For this, the parallelism
strategy becomes critical.
Step 1: Fitting a Training Step in Memory
The first question is simple: Does our SmolLM3 3B model even fit in a single H100’s 80 GB of memory? To answer this, we
used Nanotron’s predict_memory tool, which estimates memory consumption for model parameters, optimizer states,
gradients, and activations.
Maximum speedup=  
Serial fraction+  
Number of processorsParallel fraction
1
We follow the Ultra-Scale Playbook’s approach to finding optimal training configurations. The playbook breaks the problem
into three sequential steps: First ensure the model fits in memory, then achieve your target batch size, and finally optimize
for maximum throughput. Let’s walk through how we applied this to SmolLM3.
GPU scaling in distributed LLM training: Amdahl's law in action

The results showed we were pushing close to the 80 GB limit. This meant we needed some form of parallelism that would
reduce the per-GPU memory footprint, be it tensor parallelism (splitting model layers across GPUs), pipeline parallelism
(splitting model depth across GPUs), or ZeRO optimizer sharding (distributing optimizer states). Without at least one of
these strategies, we wouldn’t be able to train efficiently, or possibly at all.
Step 2: Achieving the Target Global Batch Size
Once we knew the model would fit in memory with some form of parallelism, we needed to determine how to achieve our
target global batch size (GBS) of approximately 2 million tokens. This constraint gives us our first equation:
where:
DP (data parallelism) is the number of data-parallel replicas.
MBS (micro-batch size) is the number of tokens processed per GPU per micro-batch.
GRAD_ACC (gradient accumulation) is the number of forward/backward passes before an optimizer step.
SEQLEN (sequence length) is the number of tokens per sequence (4,096 for the first pretraining stage).
We also have a hardware constraint from our 384 H100s:
where:
TP (tensor parallelism) denotes the number of GPUs used to shard a model’s weight matrices within each layer.
PP (pipeline parallelism) denotes the number of pipeline stages used to partition the model’s layers across GPUs.
These two equations defined our search space.
Step 3: Optimizing Training Throughput
With our constraints established, we needed to find the parallelism configuration that would maximize training throughput.
The search space is defined by the hardware topology and model architecture.
GBS=DP×MBS×GRAD_ACC×SEQLEN≈2M tokens
DP×TP×PP=384=2 ×7 3
Memory timeline from Nanotron's predict_memory tool showing SmolLM3 3B peaks at 74 GB, approaching the H100's 80 GB limit.

Our hardware setup presents two distinct types of interconnects, as we saw earlier: NVLink for intranode communication
(900 GB/s) and EFA for internode communication (~50 GB/s). This topology naturally suggests using at least two forms of
parallelism to match our network characteristics. The dramatic bandwidth difference between these interconnects will
heavily influence which parallelism strategies work best.
From a model perspective, SmolLM3’s architecture constrained our options. Since we’re not using a mixture-of-experts
architecture, we don’t need expert parallelism . Similarly, training with a 4,096-token sequence length in the first stage
meant context parallelism wasn’t required. This left us with three primary parallelism dimensions to explore: data
parallelism (DP) , tensor parallelism (TP) , and pipeline parallelism (PP) .
Given our constraints from Step 2, we needed to sweep across several parameters:
DP with ZeRO variants (ZeRO-0, ZeRO-1, ZeRO-3): Values from 1 to 384, constrained to multiples of 2 and/or 3.
TP (1, 2, 3, 4, 6, 8): Keep within a single node to fully leverage NVLink’s high bandwidth.
PP (1..48): Split model depth across GPUs.
MBS (2, 3, 4, 5): Depending on memory savings from parallelism, we can increase MBS to better utilize Tensor Cores.
Activation checkpointing (none, selective, full): Trade additional compute for reduced memory and communication.
Kernel optimizations : Use CUDA Graphs and optimized kernels where available.
While this may seem like an overwhelming number of combinations, a practical approach is to benchmark each dimension
independently first and eliminate configurations that significantly hurt throughput. The key insight is that not all parallelism
strategies are created equal. Some introduce communication overhead that far outweighs their benefits, especially at our
scale.
In our case, pipeline parallelism showed poor performance characteristics. PP requires frequent pipeline bubble
synchronization across nodes, and with our relatively small 3B model, the communication overhead counteracted any
potential benefits. Additionally, we didn’t have access to highly efficient PP schedules that could eliminate the pipeline
bubble entirely, which further limited PP’s viability. Similarly, ZeRO levels above 0 introduced significant all-gather and
reduce-scatter operations that hurt throughput more than they helped with memory. These early benchmarks allowed us to
narrow our search space dramatically, focusing on configurations that combined data parallelism with modest tensor
parallelism.
To evaluate each configuration, we ran benchmarks for five iterations and recorded tokens per second per GPU , which is
ultimately the metric we care about. We used Weights & Biases and Trackio to log throughputs and configurations, making it
easy to compare different parallelism strategies.
After systematically benchmarking the available options in Nanotron, we settled on DP = 192 , which leverages internode
EFA bandwidth for data-parallel gradient synchronization. This means 192 independent model replicas, each processing
different batches of data. For tensor parallelism, we chose TP = 2 , keeping tensor-parallel communication within a single
node to fully exploit NVLink’s high bandwidth. This splits each layer’s weight matrices across two GPUs, requiring fast
communication for the forward and backward passes.
This configuration achieved our target global batch size of approximately 2 million tokens (192 × 3 × 1 × 4,096 ≈ 2.3M)
while maximizing throughput on our 384-H100 cluster. You can see the full training configuration in stage1_8T.yaml .
That brings us to the end of our infrastructure walkthrough. Let’s now step back and reflect on the bigger picture: what
we’ve covered, what we’ve learned, and where this all leads.
Our micro-batch size ( MBS = 3 ) strikes a balance between memory usage and compute efficiency. Larger batch sizes
would better utilize Tensor Cores, but we were already pushing close to memory limits. Finally, we opted for ZeRO-0,
meaning no optimizer state sharding. While ZeRO-1 or ZeRO-3 could have reduced the memory footprint, the communication
overhead from gathering and scattering optimizer states across our 384 GPUs would have significantly hurt throughput.