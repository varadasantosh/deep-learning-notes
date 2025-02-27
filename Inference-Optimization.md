# References:- 
- https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- https://training.continuumlabs.ai/inference/why-is-inference-important/paged-attention-and-vllm
  
# Inference Process:
  The inference process for a Large Language Model (LLM) is divided into two  phases: Prefill and Decode. Each phase has distinct computational requirements based on the specific operations 
  performed during that stage. Below are the high level steps performed during Inference

  
  - Model Weights or Parameters are Loaded to GPU 
  - Prefill phase process the Input Prompt and generate the tokens for the user prompt, generated tokens are loaded to GPU for single forward pass through the model,
    outcome of the forward pass gives the logits for tnext token in the sequence. the full forward pass has key operations like Calculating Attention Score , Softmax, Masking etc...
    among the operations mentioned, Attention Score calculation is most computationally expensive which requires General Matrix Multiplication(GEMM) this can be done in parallel
    for all the tokens involved hence the forward pass requires more FLOPS, due to the aforementioned behaviour Prefill phase is considered to be Compute Bound Operation. During
    this phase the KV Cache is also generated , we will understand the importance of KV Cache in below
  - Decode phase operates in autoregressive manner generating tokens iteratively until the EOS token is prodcued (or) Maximum Output Token is reached, for new token generation step (nth   
    step) token generated at `n-1` step would be considered as `Query vector`, at each decoding step KV cache from the VRAM is loaded to GPU. K,V values of the `n-1th` token  are generated 
    and appended to existing K & V matrices , after appending the new K,V values  Query vector(n-1 token) is multiplied with updated Key Matrix which produces the Attention Scores required for 
    further processing , these scores further normalized by applying `Softmax` function . these attention weights or probabilites thus produces are multiplied with Value matrix to calculate the 
    final probablities for each token present in the vocabulary, like mentioned this is done iteratively until EOS token is produced or Max tokens is reached, important point to notice here is
    that during this whole process there is lot of transfer operations between VRAM and On-Chip memory , KV Cache is updated at each step and moved to VRAM and loaded from VRAM to On-chip memory
    for next token generation ,similarly weights required for computation for other parts of the Model are also loaded as required , but the difference between the weights & KV cache being
    that weights are not updated but KV Cache is updated every time.Hence the Decoding step is considered to be Memory Bound operation as more time is spent on data transfer.
    


# KV Cache

 Memory Required for KV Cache :- 2 * Precision * $n_{layers}$ * $d_{model}$ * seqlen * batch
 - Precision:- 4 Bytes for FP32, 2 Bytes for FP16 , 1 Byte for INT8
 - $n_{layers}$:- Number of Layers in the Model Architecture
 - $d_{model}$:- Dimesions of the Embeddings used in Model Architecture
 - seqlen:- Sequence Length is maximum sequence length of the sentence 
 - batch:- Number of Prompts Batched together

  Like Described in the earlier section on Inference Process, Decode phase is involves more data transfers and repetitive operations for calculating the attention scores with K & V matrices
  multiple times , for instance if we imageine a Completion Model, If the user initial Prompt is 4 tokens and max token length is 4096 tokens if no EOS token is produced until 4096 token
  the process of generating token continues until 4096 tokens are produced , during this process the attention scores are calculated across multiple layers of the Model 4092 times, each time 
  the operation performed in earlier steps needs to be recomputed , inorder to avoid recompute the KV Cache keeps storing the values of K, V matrices instead of generating them time and again,
  this saves time in processing unncessary computation. Below image can help us to understand the behaviour.

  There are further Optimizations to this like `charcter.ai` does this further to Cache the KV layer across different Layers

   ![image](https://github.com/user-attachments/assets/862f1d2e-7041-4735-b62f-591d238c0a2a)

   https://huggingface.co/blog/not-lain/kv-caching

 
# Paged Attention
 https://training.continuumlabs.ai/inference/why-is-inference-important/paged-attention-and-vllm
 
# Batching
  Batching is the process of combining Multiple Requests or Prompts for processing , ORCA - Distributed Serving System for Transformer-Based Generative Models proposed two techniques for
  batching **Static Batching** & **Continuous Batching** 
  - Static Batching as the name indicates at a given time few requests(ex:-**64**) are combined togehter and sent to GPU for Inferencing ,Once the processing for these requests are completed
    the next batch is loaded to GPU, this mode of batching is more simple and easy to process also throughput is higher , one disadvantage with the static batching is if the Batch has requests 
    which leads to different sizes of sequences, though one of the sequence generation is completed , meaning the prompt completion of one user is completed the output is not yet sent to the 
    user ,as the other requests are still being processed , once all the requests are done with processing , the results are offloaded from GPU to CPU, thus it can hamper user experience. This 
    also leads to inefficient usage of GPU , as few cores of GPU sits idle
  - Continous Batching address the issues that are arised in Static Batching, in this process when one of the requests are done with generation of sequence, a new sequence or request will
    replace the completed sequence , while other requests in the batch are still being processed.This helps in better utilizatio of GPU , also helps to provide the results as early as possible

    Determing the Batch size can be critical , it depends on factors like resources we have, for instance if we consider each request as 1 batch which is not efficient way to use the resources, 
    we can increase this until we reach optimum usage, some experiments indicatea that Batch size of 64 can produce good Through put with acceptable Levels of Latency but again this size should
    be determined through several experiments based on the computing resources

    Though below representation of Contiuous Batching is misleading, one important thing to be aware is for Batch of requests to be combined , all the lenghts should be same where as the prompt
    might be of different lenght and generated sequence lenghts also could be of different sizes, suppose if we have two requests like below it is not posssible to batch them togehter due to
    varying lengths, hence there are few operations that depends on the uniform batch size where few operations does not depend on the uniformity of batch size, hence these operations like
    matrix multiplications which expects uniform batch size can't be handled in batch and thus processed in sequence , the other set of operations which are independent of dimension they
    can be processed in batch, this process is known as Selective Batching, this is form of continuous baching where a completed request is replaced by new request

    - Request 1, K and V cache shape is [D, 4, D/H] (4 tokens),
    - Request 2, K and V cache shape is [D, 7, D/H] (7 tokens),

    Static Batching
    ----------------
    ![image](https://github.com/user-attachments/assets/cbf55f1a-28ee-4c69-9a4a-9c4e88f597ed)

    Continuous Batchining
    -------------------
    ![image](https://github.com/user-attachments/assets/6f36fc62-5085-411c-be03-9476d8f7719d)

    Ref:- https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/#fn:1
    
    
    
# Quantization
# Speculative Decoding
# Inference Engines
  - NVIDIA Triton
  - VLLM
  - 


