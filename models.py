import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the internals of the Transformer
# except where indicated to implement the multi-head
# attention. 


# Remove this later
# You may subclass nn.module,
# use built-in Linear modules, and built-in implementations of nonlinearities (tanh, sigmoid, and softmax),
# initializations, loss functions, and optimization algorithms.

def clones(module, N):
    "A helper function for producing N identical layers (each with their own parameters)."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1

class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_keep_prob):
        super(RNNCell, self).__init__()

        self.tanh = nn.Tanh()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(1-dropout_keep_prob)
        
        self.Wx = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.Wh = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.k = np.sqrt(1 / hidden_dim)

    def init_weights_uniform(self):
        nn.init.uniform_(self.Wx.weight.data, a=-self.k, b=self.k)  # W_x
        nn.init.uniform_(self.Wh.weight.data, a=-self.k, b=self.k)  # W_h
        nn.init.uniform_(self.Wh.bias.data, a=-self.k, b=self.k)  # b_h

    def forward(self, inputs, hidden):
        inputs = self.dropout(inputs)
        output = self.tanh(self.Wx(inputs) + self.Wh(hidden))
        return output

class RNN(nn.Module):  # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()
        # Initialization of the parameters of the recurrent and fc layers.
        # Your implementation should support any number of stacked hidden layers
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the
        # provided clones function (as opposed to a regular python list), in order
        # for Pytorch to recognize these parameters as belonging to this nn.Module
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.dropout = nn.Dropout(1-self.dp_keep_prob)
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.hidden_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.hidden_layers.append(
                RNNCell(
                    input_dim=self.emb_size if i==0 else self.hidden_size,
                    hidden_dim=self.hidden_size,
                    dropout_keep_prob=self.dp_keep_prob,
                )
            )
        self.out_layer=nn.Linear(self.hidden_size,self.vocab_size)
        self.init_weights_uniform()

    def init_weights_uniform(self):
        # Initialize all the weights uniformly in the range [-0.1, 0.1]
        # and all the biases to 0 (in place)
        nn.init.uniform_(self.embeddings.weight.data, a=-0.1, b=0.1)
        for h in self.hidden_layers:
            h.init_weights_uniform()
        nn.init.uniform_(self.out_layer.weight.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.out_layer.bias.data, a=-0.1, b=0.1)

    def init_hidden(self):
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        return torch.zeros([self.num_layers, self.batch_size, self.hidden_size])

    def forward(self, inputs, hidden):
        # Compute the forward pass, using a nested python for loops.
        # The outer for loop should iterate over timesteps, and the
        # inner for loop should iterate over hidden layers of the stack.
        #
        # Within these for loops, use the parameter tensors and/or nn.modules you
        # created in __init__ to compute the recurrent updates according to the
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details,
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """
        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size], device=inputs.device)
        embd_input = self.embeddings(inputs)

        for t in range(self.seq_len):
            x = embd_input[t]  # x shape: [batch_size, emb_size]
            h = []
            for layer,h_layer in enumerate(self.hidden_layers):
                temp = h_layer(x, hidden[layer])
                x = temp 
                h.append(temp) 
            hidden = torch.stack(h)
            #dropout the last hidden layer
            h_dropout_out=self.dropout(x)
            logits[t] = self.out_layer(h_dropout_out)  

        # logits shape: [seq_len,batch_size, vocab_size]
        # hidden shape: [num_layers,batch_size, hidden_size]
        return logits, hidden 

    def generate(self, input, hidden, generated_seq_len):
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
        samples = torch.zeros([generated_seq_len, self.batch_size], device=input.device)
        embd_input = self.embeddings(input)

        for i in range(generated_seq_len):
            x = embd_input[0]  # x shape: [batch_size, embed_size]
            h = []
            for layer,h_layer in enumerate(self.hidden_layers):
                temp = h_layer(x, hidden[layer])
                x = temp 
                h.append(temp) 
            hidden = torch.stack(h)
            h_dropout_out=self.dropout(x)
            logits = self.out_layer(h_dropout_out)
            out=F.softmax(logits,dim=1)
            out=Categorical(probs=out).sample()
            samples[i] = out
            embd_input=self.embeddings(out) #convert to embeddings as input to next step
        return samples


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim,  dropout_keep_prob):
        super(GRUCell, self).__init__()
        self.Wr=nn.Linear(input_dim,hidden_dim, bias=False) 
        self.Ur=nn.Linear(hidden_dim,hidden_dim) 

        self.Wz=nn.Linear(input_dim,hidden_dim, bias=False) 
        self.Uz=nn.Linear(hidden_dim,hidden_dim) 
        
        self.Wh=nn.Linear(input_dim,hidden_dim, bias=False) 
        self.Uh=nn.Linear(hidden_dim,hidden_dim) 

        self.dropout=nn.Dropout(1-dropout_keep_prob)
        self.tanh=nn.Tanh()
        self.sigm=nn.Sigmoid()
        self.k = np.sqrt(1 / hidden_dim)

    def init_weights_uniform(self):
        nn.init.uniform_(self.Wr.weight.data, a=-self.k, b=self.k)  
        nn.init.uniform_(self.Wz.weight.data, a=-self.k, b=self.k)  
        nn.init.uniform_(self.Wh.weight.data, a=-self.k, b=self.k)
        
        nn.init.uniform_(self.Ur.weight.data, a=-self.k, b=self.k)
        nn.init.uniform_(self.Uz.weight.data, a=-self.k, b=self.k)  
        nn.init.uniform_(self.Uh.weight.data, a=-self.k, b=self.k)

        nn.init.uniform_(self.Ur.bias.data, a=-self.k, b=self.k)  
        nn.init.uniform_(self.Uz.bias.data, a=-self.k, b=self.k)  
        nn.init.uniform_(self.Uh.bias.data, a=-self.k, b=self.k)  

    def forward(self, inputs, hidden):
        inputs=self.dropout(inputs)
        r=self.sigm(self.Wr(inputs) + self.Ur(hidden))    
        z=self.sigm(self.Wz(inputs) + self.Uz(hidden))    
        th=self.tanh(self.Wh(inputs) + self.Uh(r*hidden))  
        out=((1-z)*hidden)+(z*th)                     
        return out

# Problem 2
class GRU(nn.Module):  
    """
    Follow the same instructions as for RNN (above), but use the equations for
    GRU, not Vanilla RNN.
    """

    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.dropout = nn.Dropout(1-self.dp_keep_prob)
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.hidden_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.hidden_layers.append(
                GRUCell(
                    input_dim=self.emb_size if i==0 else self.hidden_size,
                    hidden_dim=self.hidden_size,
                    dropout_keep_prob=self.dp_keep_prob,
                )
            )
        self.out_layer=nn.Linear(self.hidden_size,self.vocab_size)
        self.init_weights_uniform()

    def init_weights_uniform(self):
        # Initialize all the weights uniformly in the range [-0.1, 0.1]
        # and all the biases to 0 (in place)
        nn.init.uniform_(self.embeddings.weight.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.out_layer.weight.data, a=-0.1, b=0.1)
        nn.init.zeros_(self.out_layer.bias.data)
        for h in self.hidden_layers:
            h.init_weights_uniform()

    def init_hidden(self):
        return torch.zeros([self.num_layers, self.batch_size, self.hidden_size])

    def forward(self, inputs, hidden):
       logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size], device=inputs.device)
       embd_input = self.embeddings(inputs)
       for t in range(self.seq_len):
            x = embd_input[t]  # x shape: [batch_size, embed_size]
            h = []
            for layer,h_layer in enumerate(self.hidden_layers):
                temp = h_layer(x, hidden[layer])
                x = temp 
                h.append(temp) 
            hidden = torch.stack(h)
            h_dropout_out=self.dropout(x)
            logits[t] = self.out_layer(h_dropout_out) 
       return logits, hidden

    def generate(self, input, hidden, generated_seq_len):
        samples = torch.zeros([generated_seq_len, self.batch_size], device=input.device)
        embd_input = self.embeddings(input)

        for i in range(generated_seq_len):
            x = embd_input[0] 
            h = []
            for layer,h_layer in enumerate(self.hidden_layers):
                temp = h_layer(x, hidden[layer])
                x = temp 
                h.append(temp) 
            hidden = torch.stack(h)
            x_dropout_out=self.dropout(x)
            logits = self.out_layer(x_dropout_out)
            out=F.softmax(logits,dim=1)
            out=Categorical(probs=out).sample()
            samples[i] = out
            #convert to embeddings as input to next step
            embd_input=self.embeddings(out)
        return samples


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""


# This code has been modified from an open-source project, by David Krueger.
# The original license is included below:
# MIT License
#
# Copyright (c) 2018 Alexander Rush
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------
class SingleAttention(nn.Module):
    def __init__(self, n_units, d_k, dropout):
        super(SingleAttention, self).__init__()

        self.n_units = n_units
        self.d_k = d_k

        self.Wq = nn.Linear(self.n_units, self.d_k)
        self.Wk = nn.Linear(self.n_units, self.d_k)
        self.Wv = nn.Linear(self.n_units, self.d_k)
        self.k = np.sqrt(1/self.n_units)
        self.dropout = nn.Dropout(dropout)

    def init_weights_uniform(self):
        nn.init.uniform_(self.Wq.weight.data, a=-self.k, b=self.k)
        nn.init.uniform_(self.Wk.weight.data, a=-self.k, b=self.k)
        nn.init.uniform_(self.Wv.weight.data, a=-self.k, b=self.k)
        if self.Wq.bias is not None:
            nn.init.uniform_(self.Wq.bias.data, a=-self.k, b=self.k)
        if self.Wk.bias is not None:
            nn.init.uniform_(self.Wk.bias.data, a=-self.k, b=self.k)
        if self.Wv.bias is not None:
            nn.init.uniform_(self.Wv.bias.data, a=-self.k, b=self.k)

    def forward(self, Q, K, V, mask=None):
        Q_ = self.Wq(Q)
        K_ = self.Wk(K)
        V_=self.Wv(V)
        softmax=torch.einsum('bik,bkj->bij', Q_, K_.transpose(1, 2))/np.sqrt(self.d_k)
        # Softmax masked
        if mask is not None:
            mask_float=mask.to(dtype=torch.float32)
            softmax_mask = (softmax*mask_float) - 1e9*(1 - mask_float)
        else:
            softmax_mask=softmax
        softmax_output = F.softmax(softmax_mask,-1)
        softmax_output_dropout=self.dropout(softmax_output)
        output = torch.einsum('bik,bkj->bij',softmax_output_dropout, V_)

        return output
# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        self.n_heads=n_heads
        self.n_units=n_units
        self.k = np.sqrt(1/self.n_units)
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0

        self.attention_layers = nn.ModuleList()
        for _ in range(self.n_heads):
            self.attention_layers.append(SingleAttention(self.n_units,self.d_k,dropout))
        self.Wo=nn.Linear(n_units, n_units)
        self.init_weights_uniform()
        
    def init_weights_uniform(self):
           nn.init.uniform_(self.Wo.weight.data, a=-self.k, b=self.k)
           if self.Wo.bias is not None:
              nn.init.uniform_(self.Wo.bias.data, a=-self.k, b=self.k)
           for layer in self.attention_layers:
            layer.init_weights_uniform()


        # TODO: create/initialize any necessary parameters or layers
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout

    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size: (batch_size, seq_len, self.n_units, self.d_k)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.
        H_c= torch.empty((query.shape[0], query.shape[1], 0), device=query.device)
        mask_float=mask.to(dtype=torch.float32)
        for layer in self.attention_layers:
            H_c=torch.cat((H_c,layer(query,key,value,mask_float)),-1)
        out=self.Wo(H_c)
        return  out # size: (batch_size, seq_len, self.n_units)


# ----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        # print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# ----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # apply the self-attention
        return self.sublayer[1](x, self.feed_forward)  # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """

    def __init__(self, layer, n_blocks):  # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6,
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# ----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """

    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
