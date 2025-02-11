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
   5.  Next step is to perform Matrix multiplication between the probabilities(Normalizing the dot product between Q,K) calculated in earlier step using Softmax & the **V** Values matrix whose size is **(batch_size,seq_len,n_dim)**, hence these both matrices need to be moved from HBM to On-Chip memory
   6.  Matrix multiplication is performed between Softmax Values & **V** values matrix to get the final attention score

   From the above steps we can infer that majorly the there are two types of operations one being Matrix Multiplications which is FLOPS(Floating Point Operations), other is data movement between DRAM(HBM) to SRAM (On-Chip Memory), due to the availability of more number of GPU & each GPU having multiple cores
   the Floating point operations are calculated faster , once this is done threads present inside the **SM** are idle until they get new set of instructions and Data on which these instructions need to be performed , this makes these operations Memory bound as the time taken to move the data between SRAM (On Chip Memory) & DRAM 
   is more than the time taken to perform FLOPS (Matrix Multiplicaton in this case).
   
   
   Flash Attention address this problem by dividing the matrices into multiple blocks , and peforms fusing of kernal operation by calculating the softmax on the fly by applying **tiling** technique , this fusing of kernel operation reduces the storing of intermediate results and memory transfers, in addition
   the same calculations are recomputed during backpropagation instead of storing them on HBM and reading them to On-Chip memory this reduces the I/O operations which is bottleneck in Self Attention.

   Flash attention divides the matrix into small tiles and the operations like dot product between Q,${K^T}$ are performed and result of this is passed to another kernel function which calculates softmax on this tile, furhter this result is passed to another kernel which calculates the dot product between softmax values and V matrix,
   as these data is passed through multiple kernel functions we don't store the intermediate results on HBM.

   Though this looks simple , major limitation is here with softmax calculation which needs all the scores at the same time as the formula of softmax mentioned below needs all the values to calculate this , fortunately this was already solved earlier using tiling in which each blocks softmax is calculated when t

   $$
   \text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n}\exp(x_j)}
   $$
   

   Reference Links:-

   1. https://horace.io/brrr_intro.html
   2. https://training.continuumlabs.ai/inference/why-is-inference-important/flash-attention-2
   3. https://www.youtube.com/watch?v=IoMSGuiwV3g
   4. https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad#:~:text=So%20basically%2C%20in%20order%20to,statistics%20for%20each%20of%20the
   5. https://www.nvidia.com/en-us/on-demand/session/gtc24-s62546/


   Standard Attention vs Flash Attention from Hugging Face:-
   ------------------
   ![image](https://github.com/user-attachments/assets/8ce6ec2f-2df2-4d5e-b643-598ba3b27097)


# LAYER-NORMALIZATION

# FEEDFORWARD LAYER
