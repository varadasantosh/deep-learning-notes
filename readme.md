# Transformers
- [Encoder]
   - [Embeddings]
     - [Positional Embeddings](#Positional-Embeddings)
     - [Rotatory Embeddings](#ROTATORY-EMBEDDINGS)
  - Attention
    - [Self Attention](#SELF-ATTENTION)
    - [Multi Head Attention](#MULTIHEAD-ATTENTION)
    - [Cross Attention](#CROSS-ATTENTION)
    - [Flash Attention](#FLASH-ATTENTION)
  - [Layer Normalization](#LAYER-NORMALIZATION)
  - [FeedForward Layer](#FEED-FORWARD-LAYER)

# Encoder

# Positional-Embeddings

# Rotatory-Embeddings

# SELF-ATTENTION

# MULTIHEAD-ATTENTION

# CROSS-ATTENTION

# FLASH-ATTENTION

  Flash Attention is IO Aware & Exact Attention. To understand this, we need to be aware of Vanilla Attention (Self-Attention), which is pivotal for Transformer Architecture. Additionally, having some knowledge of GPU Architecture is beneficial.

**Self-Attention Recap**: In order to calculate Self-Attention, the following steps are performed:

  
   $$
   \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
   $$

   1. The input embeddings `x` with dimensions (batch_size, sequence_len, n_dim) are passed through three linear layers with weights  $W_q$ ,  $W_k$  &  $W_v$ . As a result, we obtain the matrices **Q**, **K**, and **V**, which have the same dimensions as `x`:

      - \( Q \): Query matrix
      - \( K \): Key matrix
      - \( V \): Value matrix
      - \( d_k \): Dimensionality of the key vectors

 
   2. **Q** - Query Matrix &  **K** -Key Matrix are moved to SM (Streaming Multiprocessor) On-chip Memory for Matrix Multiplication Operation (`GEMM`), Result of this operation is moved to HBM(VRAM) in GPU
   3.  We need to apply Softmax on the result of Multiplication of **Q** & **$${K^T}$$** to calculate the probablities , hence this result again needs to be moved from HBM to SM On-Chip Memory 
   4.  After the Softmax is calculated , result of the same is moved to HBM(VRAM), The size of the Softmax matrix would be of **(batch_size,seq_len,seq_len)**
   5.  Next step is to perform Matrix multiplication between the probabilities calculated in earlier step using Softmax & the **V** Values matrix whose size is **(batch_size,seq_len,n_dim)**, hence these both matrices need to be moved from HBM to On-Chip memory
   6.  Matrix multiplication is performed between Softmax Values & **V** values matrix to get the final attention score

   From the above steps we can infer that majorly the there are two types of operations one being Matrix Multiplications which is FLOPS(Floating Point Operations), other is data movement between HBM to SM (On-Chip Memory), due to the availability of more number of GPU & each GPU having multiple cores
   the Floating point operations are calculated faster , once this is done threads present inside the SM sit idle until they get new set of instructions and Data on which these instructions need to be performed , this transfer of data is major bottleneck due to the bandwidth between the HBM and SM
   hence the processing power is limited by transfer or number of I/O operations , this is addressed by Flash Attention. Flash Attention address this problem by dividing the matrices into multiple blocks , also using the tiling to calculate the softmax for intermediate attention score, also it optimizes
   the backpropagation , instead of storing these attention scores the algorithm calculates them on the fly , hence the I/O operatiosn are optimized.

# LAYER-NORMALIZATION

# FEEDFORWARD LAYER
