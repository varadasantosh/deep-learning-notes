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
  - Prefill phase process the Input Prompt and generate the tokens for the user prompt, generated tokens are loaded to GPU for single forward pass throughthe model,
    outcome of the forward pass gives the logits for tnext token in the sequence. the full forward pass has key operations like Calculating Attention Score , Softmax, Masking etc...
    among the operations mentioned, Attention Score calculation is most computationally expensive which requires General Matrix Multiplication(GEMM) this can be done in parallel
    for all the tokens involved hence the forward pass requires more FLOPS, due to the aforementioned behaviour Prefill phase is considered to be Compute Bound Operation. During
    this phase the KV Cache is also generated , we will understand the importance of KV Cache in below
  - Decode phase generates tokens iteratively until the EOS token is reached or Maximum Sequence Length/Maximum Output Tokens is reached, during this process the token generated at `n-1`
    step would be considered as new Query vector, and the current step being `nth` we load KV cache from the VRAM to GPU to calculate Attention Score, we add new column for K & V vectores to   
    existing K, V matrices and peform the matrix multiplication  
    
  


# KV Cache

 Memory Required for KV Cache :- 2 * Precision * $n_{layers}$ * $d_{model}$ * seqlen * batch
 - Precision:- 4 Bytes for FP32, 2 Bytes for FP16 , 1 Byte for INT8
 - $n_{layers}$:- Number of Layers in the Model Architecture
 - $d_{model}$:- Dimesions of the Embeddings used in Model Architecture
 - seqlen:- Sequence Length is maximum sequence length of the sentence 
 - batch:- Number of Prompts Batched together
 
# Paged Attention
 https://training.continuumlabs.ai/inference/why-is-inference-important/paged-attention-and-vllm
# Continuous Batching
# Prefill
# Quantization
# Speculative Decoding


