# Reference Articles
  - https://tinkerd.net/blog/machine-learning/distributed-training/
   - https://www.youtube.com/watch?v=toUSzwR0EV8&t=2s
   

Before exploring the different techniques of Distributed Training, it is essential to understand why it is needed.

With advancements in both technology and hardware, the size of deep learning models has grown significantly. Modern Large Language Models (LLMs) are trained on massive datasets and have billions of parameters, making them too large to fit within the memory of a single GPU.

If such models were trained on a single GPU, the process could take hundreds of years to complete. Distributed training techniques, such as Fully Sharded Data Parallel (FSDP), help overcome these limitations by distributing the workload across multiple GPUs. This not only accelerates training but also enables the development of increasingly larger and more capable models. The more data a model can learn from, the better its performance.

A model comprises of below 
- Parameters (Weights) - Calculated during Forward Propagation
- Gradients - Calculate during Backward Propagation
- Optimizer State (Ex:- Adam Optimizer has additionally has 3 more parameters Momentum, Velocity )
- Token Embeddings
- Positional Embeddings
  

NCCL Primitives - [DOCS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
---------------
- Broadcast
- Reduce
- All Reduce
- All Gather
- Reduce Scatter

Distributed Data Parallel 
----------------------------
Deep Learning models consist of two main components mentioned below . Distributed Data Parallel (DDP) helps improve training speed, particularly when the number of parameters is relatively small, but the dataset is large.
- parameters (model weights)
- data. 

When a dataset is too large to fit into GPU VRAM, there are two main options:

- Scaling the infrastructure (adding more GPUs or nodes). However, this has limitations since GPU VRAM cannot be scaled indefinitely.
- Dividing the dataset into smaller batches so that each batch fits into the available VRAM.

While batching allows training on large datasets, training sequentially (one batch at a time) can be inefficient and slow. This is where Distributed Data Parallel (DDP) comes into play.

With DDP, instead of processing batches sequentially, we distribute batches across multiple GPUs and train in parallel. For example, if we have 4 GPUs in a single node, we can divide the dataset into 4 batches and assign one batch to each GPU.

To enable this, we need to replicate the model across all GPUs, ensuring each GPU has an identical copy of the model. Each GPU processes a different batch of data independently. After processing, gradients are synchronized across all GPUs using an all-reduce operation (**NCCL Library** ), ensuring model updates remain consistent. The same can also be extended to GPU across different Nodes.

There is lot happening behind the scenes for co-ordinating the training process between GPU's (Intra Node) & Inter Node. Below are the high- level steps that are performed

1. Divide the Batches across GPU's
2. Go through the Forward Pass (Each Batch that resides on respective GPU)
3. Calculate Local Gradients (on Each GPU)
4. Perform All Reduce Operation to bring all the Local Gradients to One of the GPU s'
5. Once the Gradients are accumulated and calculated , pass the Gradients back to all the GPU's
6. Each GPU calculates peforms the Update Optimizater State for the corresponding Weights (Parameters)

<img width="336" alt="image" src="https://github.com/user-attachments/assets/dd1c3bf1-f629-49c7-8d5c-ca92707426aa" />


Pipeline Parallelism
---------------------------
As we briefly looked at the Distributed Data Parallel, it address the challenges with Model Training with Large Dataset and Model can fit on single GPU, after the birth of Transformer Architecture we evidenced unprecednted increase in the size of the model , each Model has large number
of parameters, if the model can't be fit into memory **Distributed Data Parallel** alone would not solve the problem as this approach relies on fitting entire model in Memory, Pipeline Parllelsim to the rescue which pivot the model to be ditributed across GPU's rather than distributing data, Pipeline Parallelism as a concept can be further implemented in two different ways

 - Vertical splitting the Model (**Model Parallelism**)

    In this approach the layers of the model are split across the available GPU's ex:- if we have 4 Hidden Layers and 4 GPU's split layers and train each layer on one GPU,
    the downside of this approach is while one layer is being trained on one GPU the other GPU sits idle , which is not efficient way of using the precious resources.
   
 - Horizontal Spllitting of Model(**Tensor Parallelism**)

    Here the parameters, Gradients & Optimizer States are split across multiple GPU's meaning if we need to calcualte Dot Product W.X, We split these matrices across 4 GPU's and calculate
    the dot product across different GPU's and bring the Parameters to one GPU for calculating the Gradients for Backward Propagation using NCCL Operations

 - Mix Data & Model Parallelism

    In this approach we split the Model Vertically and interleave the different Layers on different GPU's , during this process Layer 1 is trained by GPU-1 On Batch-1, the same Batch now 
    passed onto GPU-2 for processing next layer etc. Once the GPU-1 completes the processing of Batch-1, instead of keeping this idle we feed next mini Batch to GPU-1.

   
   




