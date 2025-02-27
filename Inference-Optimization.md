# References:- 
- https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- https://training.continuumlabs.ai/inference/why-is-inference-important/paged-attention-and-vllm
  
# Inference Process:
  Large Language Model (LLM) Inference process mainly comprises of two pahses **Prefill** & **Decode**, each phase has different computaional requiements according to the way the operations 
  performed in phase , for instance in the Prefill phase we complete the forward pass using all the tokens part of the initial prompt , which includes mutiple FLOPS but major contribution is 
  coming from calculating Attention in all those layers part of Model  Architecture which is a GEMM operation (Matrix Multiplication). Hence this is computationally intensive opertation and 
  considered as compute bound operation.

  Decode phase , during this phase 
  
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

 
# Paged Attention
 https://training.continuumlabs.ai/inference/why-is-inference-important/paged-attention-and-vllm
# Continuous Batching

# Quantization
# Speculative Decoding
# Inference Engines
  - NVIDIA Triton
  - VLLM
  - 


