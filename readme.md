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

 Let us briefly recap how we provide input to Transfomer Model, we know that models(machines that execute the models) don't understand text 
 hence we need to convert text to numbers such that models can interpret the numbers, we tokenize the sentence , tokenizing sentence is a 
 breaking sentence to small words that tokenizer algorithm trained on , tokenizer is like dictionay of words, the sentence is broken into chunks
 acconrding the tokenizers dictionary of words which is also known as `vocab_siz` ,  these tokens are still in text form, after breaking the 
 sentnce to chunks , we convert the tokens to IDs. if we pass these token ID's as is they don't cpature contextual information, it is just like 
 Bag of Words where each word represents one vector and does not take carry any meaning with it , inorder to embed meaning to these numbers we
 pass these tokens to Embedding Models like `BERT` , these models are capable of generating embeddings which are dense representation of words 
 we pass to the model . The Embedding Models helps us to add some meaning, but the meaning of this word can change according to its position in
 the sentence , which is generally known as context , though position encodings directly can't help with finding context , they will help us in 
 the process of going towards finding contextual meaning of the wrod by adding the position information to the Embedding  , for instance the 
 below sentences have same set of words or tokens but their meaning is different due to the position of each word
 ex:- **Economoy is moving upward** , **Is Economy moving upward**  
 If we are using **RNN** models to process input in sequence they can inherently capture positional information also they only know the words 
 that occured in the past, but they are not aware of the future tokens, where as in **Transformer** models the processing is parallelized this 
 brings challenge in preserving the order of the tokens , hence the need of postional encodings.
 
 We need to inject the infomration of the position of the token along with Embeddings, In Original papaer on "Attention is All You Need" 
 researchers proposed about encoding the position information along with Embeddings , to encode the position along with Embedding there are two 
 approaches **Learned Embeddings** & **Fixed Embeddings** the downside of learned parameters the model is , model can't process sequence length 
 that is more than what it has seen during training process. 
 
If we can comeup with formula that allows us to encode the position of the token irrespective of the length of the sequence that helps us to process the sequence whose length that is not seen during the training this is **Fixed Embeddings**.The original process proposed the Sin & Cosine functions with different frequencies using the **position** of the token and **embedding dimension** . Paper mentions that they tried both approaches and don't see significant differences in the accuracies of both approach.  Fixed embeddings using Sin & Cosine function gives the advantage of processing sequences longer than the sequences encountered during the training they prefered this approach.
   
 - $PE_{(pos,2i)} =  sin(pos/10000^(2i/d_{model}))$ (Even Dimensions of Embeddings)
 - $PE_{(pos,2i+1)} =  cos(pos/10000^(2i/d_{model}))$ (Odd Dimensions of Embeddings)

 Below is the pytorch code snippet to calculate Positional Embeddings
 


# Rotatory-Embeddings
   
   https://towardsdatascience.com/understanding-positional-embeddings-in-transformers-from-absolute-to-rotary-31c082e16b26/
   https://aiexpjourney.substack.com/p/an-in-depth-exploration-of-rotary-position-embedding-rope-ac351a45c794
   https://medium.com/@DataDry/decoding-rotary-positional-embeddings-rope-the-secret-sauce-for-smarter-transformers-193cbc01e4ed

   - When we were discussing about Positional Embeddings mentioned in foundational paper "Attention is All You Need" , we got to know the 
   importance of positional encodings , also we got to know the two different approaches `Learned` & `Fixed` though the paper preferred
   fixed positional embeddings over learned embeddings, the later results and metrics showed that fixed positional embeddings could not
   accurately process the relation ship between different words or tokens , especially when the sequence whose length is more than sequence
   encoutered during training.

   - One more problem with Fixed Positional Embeddings is as we are adding the positional information which is of the size of the embedding
   dimension like **$x_{m}$ + PE**, the way of encoding positional information helps to capture the positional information of the token
   when we are adding these encodings whose dimension is equal to the dimension of the Embedding , the information generated by Embeddings
   might be changing the original Embeddings which we would like to avoid , in Machine Learning we generally don't want to tamper with input data
   we will try to pass the input as is possible except the feature engineering to derive more features we don't want to modify the input data.
   the positinal information that we are adding in fixed embeddings they don't have any pattern or no parameters involved in training , if the 
   same token appears at different positions in the same sentence or in different sentences across batch models are not able to generalize the
   information

   Inorder to address the above mentioned problems we need an approach that captures the relative information of the tokens which would be
   helpful for Self Attention process to capture the relation ship between the tokens using the positional information of them, without 
   modifying the Embeddings, hence the birth of Relative Position Embeddings, though techincally we can capture the relative position of the
   embeddings using *N*N* matrix , this is computationally expensive. There are also few other issues with pure relative positional embeddings
   reasearchers came up with Rotary Positional Embeddings
   

# SELF-ATTENTION
   - https://lilianweng.github.io/posts/2018-06-24-attention/
   - https://github.com/jessevig/bertviz
     
  Evolution of Attention Mechanisms - Attention was firts introduced as part of seq-to-seq (Encoder-Decoder) models in the domain of Neural Machine Translation to translate text from one 
  language to other language. Initial Architectures of encoder-decoder models were composed of encoder and decoder both are RNN's, it is also possible to combine both simple RNN as part of 
  encoder and GRU or LSTM for decoder , encoder takes sentence in source language as input and generates context vector of fixed lentgh , which would be passed as input to Decoder , Decoder 
  takes the context vector and tries to map it to corresponding word or text in target language, this has few limitations as the single context vector generated by the encoder RNN could not 
  capture the entire meaning of the sentence in source language which resulted less accurate results, especially as the length of the sequence grows the accuracy drops.
  
  Inorder to overcome the issues of initial seq-to-seq models, researchers came up with an approach to capture all hidden states of encoder pass them to decoder to capture the meaning or 
  context , but now the challenge is to know which hidden state could be contributing more to find the next word in target language, this is not simple as the source and target languages 
  have different semantics , researchers came up with an approach to build alignment model (Single Layer Feed Forward Neural Network) that takes hidden state of previous decoder timestep $S_{i- 
  1}$ and encoder hidden state $h_{j}$ vector to build context vector $C_{t}$ using alignment model , the alignment model computes the compatability scores between the previous decoder hidden 
  state and each hidden state of encoder , thus computed compatability scores are passed through softmax function to normalize the scores, these scores are multiplied with each hidden state
  of the encoder to calculate the weighted scores of the encoder hidden states, all these weighted hidden states are added which results in context vector this is passed as one of the inputs
  the Decoder timestep $S_{i}$ along with hidden state of previous decoder timestep, **this lays the foundation for the Attention Mechanism, the attention that we discussed is Bahdanau Attention
  this is also called Additive attention as we are adding all the context vectors to calculate the alignment scores**, this triggered further improvements and `Loung Attention` proposed 
  different ways to calculate alginment scores to calculate the relevance between each hiddent vector of encoder and current decoder state, as part of Loung attention they also managed to avoid
  the alginment model, which reduces the number of parameters to be trained. Below is the reference picture of how Bahdanu Attention works

  ![image](https://github.com/user-attachments/assets/ddd1c4a0-165c-4fc7-a984-d24ea680cb90)

 - **Attention in Transformers**

    The above mentioned attentions **Bahdanau & Luong**  paved way for attention in Transformers, there are few disadvantages with the prior Attention mechanism major one being both of them are 
    sequential in nature, as we process one token after the other this makes training process tedious and time taking, as we see the birth the of Large Language models that are trained on 
    Billions of tokens, this would not have been possible without Self Attention which calculates these Attention scores in parallel which were referred as Alignment scores in Bahdanau & Loung 
    Attentions, to make this parallel processing possible Self Attention follows below steps.

    1. Tokenize Sentence - Breaks the sentence into tokens
    2. Generate Embeddings for the tokens
    3. Pass the Embeddings tokens through  4 different Linear Layers to generate Q,K,V & O matrices, each linear layer has its corresponding weight matrices, $W_{Q}$ , $W_{K}$ $W_{V}$ & $W_{O}$
       these weights are learned through the training process.
       
       - X * $W_{Q}$ = Q - Query Vector 
       - X * $W_{K}$ = K - Key Vector  
       - X * $W_{V}$ = V - Value Vector
       - X * $W_{O}$ = O - Output Vector

       Dimensions:-
       -----------
       - X -> T  * $d_{model}$
       - $W_{Q}$ -> $d_{model}$ * $d_{k}$
       - $W_{K}$ -> $d_{model}$ * $d_{k}$
       - $W_{V}$ -> $d_{model}$ * $d_{k}$
       - $W_{O}$ -> $d_{model}$ * $d_{k}$
         
       - T - Sequence Length
       - $d_{model}$  - Length of Embeddings
       - $d_{k}$ - Output dimensions of $W_{Q}$,$W_{K}$ & $W_{V}$, this can be same as $d_{model}$ as well
   
   
    5. Calculate the Scaled Dot Product Between Q (Query) & K (Key) vectors to find how each token relates to other token , this is simialr to calculation of alignment scores in earlier Seq-to- 
       Seq RNN models

       Scaled Dot Product Attention: -   $\left( \frac{QK^T}{\sqrt{d_k}} \right)$

        ![image](https://github.com/user-attachments/assets/25cd0ce2-7734-4ecf-b0f3-e33cce29cd78)

       
    6. Result of Scaled Dot Product Attention is passed through Softmax to normalize the attention scores
             
       Normalize Attention Scores:-  $\text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)$

    7. Multiply these Attention scores with $W_{V}$ to calculate the weighted attentions
    
    8. Result of the Weighted attentions is thus multiplied by $W_{O}$ output projections.

     Below is the code snippet that explains above steps briefly, though this is not exactly what is being used in Transformer Architecture, as we use Multi Head Attention which we will discuss
     but this is the core of the  Attention calculation

     ```
         import torch
         import torch.nn as nn
         from torch import Tensor
         
            
         class Attention(nn.Module):
           """Single attention head"""
         
           def __init__(self, embedding_dim: int, attention_dim: int):
             super().__init__()
             torch.manual_seed(0)
         
             # Initialising weights
             self.wk = nn.Linear(embedding_dim, attention_dim, bias=False)
             self.wq = nn.Linear(embedding_dim, attention_dim, bias=False) 
             self.wv = nn.Linear(embedding_dim, attention_dim, bias=False) 
         
           def forward(self, embedded: Tensor) -> Tensor:
             # calculating Query, Key and Value
             q = self.wq(embedded)
             k = self.wk(embedded) 
             v = self.wv(embedded) 
             
             # calculating attention scores
             attn_score = q @ torch.transpose(k, -2, -1) / (k.shape[-1] ** 0.5) # [batch_size, num_words, num_words]
         
             # below 2 lines is for masking in decoder block
             upper_triangular  = torch.triu(attn_score, diagonal=1).bool()
             attn_score[upper_triangular] = float("-inf")
         
             # applying softmax
             attn_score_softmax = nn.functional.softmax(attn_score, dim = -1) # [batch_size, num_words, num_words]
         
             # getting weighted values by multiplying softmax of attention score with values
             weighted_values = attn_score_softmax @ v # 
         
             return weighted_values
     ```

     # Visualizing Self Attention using Llama Model:- https://github.com/varadasantosh/deep-learning-notes/blob/tensorflow/Visualize_Self_%26_Multi_Head_Attention.ipynb
      
      - Download the Llama Model from Hugging Face

        ```
           from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
           import torch
         
           model_name= "meta-llama/Llama-3.2-3B-Instruct"
           tokenizer = AutoTokenizer.from_pretrained(model_name)
           model = AutoModel.from_pretrained(model_name, output_attentions=True)
        ```
      - Llama Model Architecture , Llama model being Decoder only we can see there are only 28 Decoder Layers, no encoder layers
        ```
        LlamaModel(
           (embed_tokens): Embedding(128256, 3072)
           (layers): ModuleList(
             (0-27): 28 x LlamaDecoderLayer(
               (self_attn): LlamaAttention(
                 (q_proj): Linear(in_features=3072, out_features=3072, bias=False)
                 (k_proj): Linear(in_features=3072, out_features=1024, bias=False)
                 (v_proj): Linear(in_features=3072, out_features=1024, bias=False)
                 (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
               )
               (mlp): LlamaMLP(
                 (gate_proj): Linear(in_features=3072, out_features=8192, bias=False)
                 (up_proj): Linear(in_features=3072, out_features=8192, bias=False)
                 (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
                 (act_fn): SiLU()
               )
               (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
               (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
             )
           )
           (norm): LlamaRMSNorm((3072,), eps=1e-05)
           (rotary_emb): LlamaRotaryEmbedding()
         )
        ```
      - Tokenize the Input Sentence & Pass it through the Llama Model
        
        ```
          
         import torch
         
         text = "the financial bank is located on river bank"
         inputs = tokenizer(text, return_tensors="pt").to("cuda")
         token_ids = inputs.input_ids[0]
         tokens = tokenizer.convert_ids_to_tokens(token_ids)
         model = model.to("cuda")
         with torch.no_grad():
             inputs = inputs.to("cuda")
             outputs = model(**inputs)

        ```  
     - Get The Attention Matrix from the Outputs, there are 28 Layers , we can see the below dimensions of the `attention_matrix` of length 28 & each layer's attention matrix is of shape               (1,24,9,9) - This is because Llama Model has 24 Heads (This refers to Multi Head attention) and sequence length of tokens that we passed is of length 9 hence the dimension of each head          is 9*9 
       
       ```
          attention_matrix = outputs.attentions
       ```
       <img width="233" alt="image" src="https://github.com/user-attachments/assets/d7219113-ed31-4c54-b36d-366426cec86b" />

       
     - Get Attentions from final layer, calculate the avg attention scores across all heads and plot the heatmap to find relation ship, though from the below heatmap we can't find stronger 
       contextual relation ship between tokens like financial & bank , river & bank we can see them when we go through individual heads of multihead attention, but one thing we can observe in 
       the attention score heatmap is all the elements above diagonal are zero. This is because the Decoder part of model has casual attention which prevents each token from attending to future        tokens of the sequence, this is important as transformers do the self attention in parallel, where as in RNN the attention always sequentially , hence we don't step on to future
       tokens, in transformers this is not the case as we are processing all the tokens in parallel.

       ```
            import seaborn as sns
            import matplotlib.pyplot as plt
            avg_attn =attention_matrix[27][0].mean(dim=0)
            sns.heatmap(avg_attn.cpu(), cmap="viridis",annot=True,fmt=".2f",xticklabels=tokens,yticklabels=tokens )
            plt.title(f"Attention Matrix (Layer 28)",fontdict={'fontsize':25})
            plt.show()
       ```
       <img width="435" alt="image" src="https://github.com/user-attachments/assets/2e553cb2-22cc-42b6-b42d-fab81d98d13c" />


 
# MULTIHEAD-ATTENTION
The Multi Head Attention is an extension to Self Attention, while the Self Attentin defined in Transfomers helps us to overcome limitations faced by RNN's, if we look at the above pitcutre we are calculating the attention over all heads of Llama Model , Multi Head Attention helps us to attend diferent aspects of elements in a sequence, in such case single weighted average is not
good option, to understand different aspects we divide the Query, Key & Value matrices to different Heads and calculate the attention scores of each head, to calculate attention for each head
we apply the same approach mentioned above,after the attention scores of each head are calculated we concatenate the attention scores of all the heads, this approach yeilds better results than
finding the attention as a whole, during this process the weight matrices that are split are learned for each head.

Llama Model(`Llama-3.2-3B-Instruct`) referred above has Embedding Dimensions of size - **3072** &  Number of Heads - **24**, thus our Query , Key & Values are split into 24 heads each head would
be of size 3072/24 = 128

Multihead(Q,K,V) =  Concat($head_{1}$, $head_{2}$,.....$head_{24}$)
$head_{i}$ = Attention(Q$W_{i}$^Q, K$W_{i}$^K, V$W_{i}$^V)

Below image captures the process of calculating MultiHead Attention

![image](https://github.com/user-attachments/assets/67f3bb0b-1ac6-454e-ab5c-6975c368a9e3)

MultiHead Implemenation in Pytroch:-
----------------------------------
```
   class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q= nn.Linear(d_model,d_model)
        self.W_k= nn.Linear(d_model,d_model)
        self.W_v= nn.Linear(d_model,d_model)
        self.W_o= nn.Linear(d_model,d_model)

    def scaled_dot_product_attention(self,Q,K,V,mask=None):

        attention_scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0,-1e9)

        attention_probs = torch.softmax(attention_scores,dim=-1)
        output = torch.matmul(attention_probs,V)
        return output

    def split_heads(self,x):
        batch_size,seq_length,d_model = x.size()
        return x.view(batch_size,seq_length,self.num_heads,self.d_k).transpose(1,2)

    def combine_heads(self,x):
        batch_size,_,seq_length,d_k = x.size()
        x.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model)

    def forward(self,Q,K,V,mask=None):

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attention_output = self.scaled_dot_product_attention(Q,K,V,mask)
        output = self.W_o(self.combine_heads(attention_output))
        return output

```

Visulization of MultiHead Attention using Llama Model:- https://github.com/varadasantosh/deep-learning-notes/blob/tensorflow/Visualize_Self_%26_Multi_Head_Attention.ipynb
-----------------------------
We use the same Model & Input text we considered for Self Attention and look at one of the Heads of the Last layer in to see if they are able to attend contextual relation ships between different tokens

 - Download the Llama Model from Hugging Face

        ```
           from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
           import torch
         
           model_name= "meta-llama/Llama-3.2-3B-Instruct"
           tokenizer = AutoTokenizer.from_pretrained(model_name)
           model = AutoModel.from_pretrained(model_name, output_attentions=True)
        ```
 - Llama Model Architecture , Llama model being Decoder only we can see there are only Decoder Layers (28 Layers), no encoder layers

   ```
        LlamaModel(
           (embed_tokens): Embedding(128256, 3072)
           (layers): ModuleList(
             (0-27): 28 x LlamaDecoderLayer(
               (self_attn): LlamaAttention(
                 (q_proj): Linear(in_features=3072, out_features=3072, bias=False)
                 (k_proj): Linear(in_features=3072, out_features=1024, bias=False)
                 (v_proj): Linear(in_features=3072, out_features=1024, bias=False)
                 (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
               )
               (mlp): LlamaMLP(
                 (gate_proj): Linear(in_features=3072, out_features=8192, bias=False)
                 (up_proj): Linear(in_features=3072, out_features=8192, bias=False)
                 (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
                 (act_fn): SiLU()
               )
               (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
               (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
             )
           )
           (norm): LlamaRMSNorm((3072,), eps=1e-05)
           (rotary_emb): LlamaRotaryEmbedding()
         )
   ```
 - Tokenize the Input Sentence & Pass it through the Llama Model
        
        ```
          
         import torch
         
         text = "the financial bank is located on river bank"
         inputs = tokenizer(text, return_tensors="pt").to("cuda")
         token_ids = inputs.input_ids[0]
         tokens = tokenizer.convert_ids_to_tokens(token_ids)
         model = model.to("cuda")
         with torch.no_grad():
             inputs = inputs.to("cuda")
             outputs = model(**inputs)

        ```  
 - Get The Attention Matrix from the Outputs, we can see the below dimensions of the `attention_matrix` of length 28 and infer that there are 28 attention layers which we can also see from 
   model architecture & each layer's attention matrix is of shape (1,24,9,9) - This is because Llama Model has 24 Heads (This refers to Multi Head attention) and sequence length 
   of tokens that we passed is of length 9 hence the dimension of each head is 9*9
       
       ```
          attention_matrix = outputs.attentions
       ```
   ![image](https://github.com/user-attachments/assets/5d332484-abe1-4c7b-8f99-9d481b6971a9)

      
- Get the last Layer from the Attention matrix and draw a heatmap for each head, as the Llama model we are referring has 24 heads below code snippet , will produce 24 heatmaps for each heads
  but due to lack of space and to keep it simple we will look at one of the heads , which captures the context of the same token `bank` according to its position and context

  ```
      import seaborn as sns
      import matplotlib.pyplot as plt
      
      fig, axes = plt.subplots(24, 1, figsize=(10, 200))
      for i, ax in enumerate(axes.flat):
          sns.heatmap(attention_matrix[27][0][i].cpu(), ax=ax, cmap="viridis",annot=True,fmt=".2f",xticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),yticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) )
          ax.set_title(f"Head {i+1}",fontdict={'fontsize':25})  
      plt.tight_layout()
      plt.show()
  ```

<img width="701" alt="image" src="https://github.com/user-attachments/assets/a08027b7-2a41-4578-b437-2f4f0d84a65a" />

<img width="511" alt="image" src="https://github.com/user-attachments/assets/d3eb0e3d-7e84-420f-a070-92eb634b45ff" />




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

 
   2. **Q** - Query Matrix &  **K** -Key Matrix are moved to SM (Streaming Multiprocessor) On-chip Memory for Matrix Multiplication Operation (`GEMM`), Result of this operation is moved to HBM(High Bandwidth Memory) in GPU
   
   3.  We need to apply Masking on the result of Multiplication of **Q** & **$${K^T}$$** to ensure padding tokens get zero probabilities after applying softmax ,
       this result again needs to be moved from HBM to SM On-Chip Memory.
   4.  After applying Masking operation, the same matrix is moved from On-chip Memory to HBM
   5.  Next step would be to apply Softmax operation on the matrix whose size is (batch_size,seq_len,seq_len), to apply softmax the matrix is moved from HBM to On-chip memory.    
   6.  After the Softmax is calculated , result of the same is moved to HBM(High Bandwidth Memory), The size of the Softmax matrix would be of **(batch_size,seq_len,seq_len)**
   7.  Next step is to perform Matrix multiplication between the probabilities(Normalizing the dot product between Q,K) calculated in earlier step using Softmax & the **V** Values matrix whose size is **(batch_size,seq_len,n_dim)**, hence these both matrices need to be moved from HBM to On-Chip memory
   8.  Matrix multiplication is performed between Softmax Values & **V** values matrix to get the final attention score

   From the above steps we can infer that majorly the there are two types of operations one being Matrix Multiplications which is FLOPS(Floating Point Operations), other is data movement       
   between DRAM(HBM) to SRAM (On-Chip Memory), due to massive parallel processing capabilities of GPU Floating point operations are calculated faster , once this is done threads present inside 
   the  **SM** are idle until they get new set of instructions and Data on which these instructions need to be performed , **this makes these operations Memory bound as the time taken to move 
   the data between SRAM (On Chip Memory) & DRAM  is more than the time taken to perform FLOPS (Matrix Multiplicaton in this case)**
   
   
   Flash Attention address this problem by dividing the matrices into multiple blocks , and peforms fusing of kernal operations ( Kernels are functions) , fusing Kernel operations can be 
   considered as chaining different functions on each set of blocks, this fusing of kernel operation reduces the need for storing of intermediate results and memory transfers, also the same 
   calculations are recomputed during backward propagation , instead of storing them and moving them between memory layers, though these two operations increase the number of FLOPS the time 
   taken to calculate the attention matrix is less duration this reduces the I/O operations which is bottleneck in Self 
   Attention.

   
   Flash attention divides the matrix into small tiles and the operations like dot product between Q,${K^T}$ are performed and result of this is passed to another kernel function which 
   calculates mask & passes the output to another function that calculates softmax , furhter this result is passed to another kernel which calculates the dot product between softmax values and 
   V matrix, as these data is passed through multiple kernel functions within SRAM we don't store the intermediate results on HBM.

   
   But here lies the major challenge, inorder to calculate the Softmax we need all the values at once to perfrom sum operation  which is required to calculate(denominator), this is required 
   as we need to divide each element of the dot matrix by sum of all the elments(which is Softmax formula) , as we are dividing the matrix into multiple blocks to perfrom kernel fusion 
   (chaining kernel functions like Dot product, masking and Softmax ) calculating the total sum is not possible ,  hence we need a way to calculate the softmax for these batches accurately,
   fortunately this can be addressed calculatin online softmax, which uses tiling technique which is metioned in NVIDIA researcher [paper](https://arxiv.org/abs/1805.02867), this approach 
   allow us to calculate the softmax for individual blocks and when we are merging them we incrementally calculate the final softmax using the formaula mentioned below until we reach final 
   merging on all the blocks 
   
   <img width="492" alt="image" src="https://github.com/user-attachments/assets/c5f58700-db05-4027-ab1b-a0814ec769cc" />


   $$
   \text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n}\exp(x_j)}
   $$
   
  # Few intresting points to note here is the number of FLOPS (Floating point operations) are more in number than the Self Attention , but the time taken is less compared to Self Attention as we are working on small cunks which makes it faster move the data between HBM and On-Chip memory and On-chip Memory to HBM memory , as we are dividing into multiple chunks this also allows us to increase Sequence Lenght which is Context Length of model, hence we can have more context length for the training model. 

  # [Online Softmax Calculation](https://github.com/varadasantosh/deep-learning-notes/blob/tensorflow/Flash_Attention_Calculations(Online_Softmax).ipynb) 
   
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
