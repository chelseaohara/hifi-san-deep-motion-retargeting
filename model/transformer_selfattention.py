""" See  https://peterbloem.nl/blog/transformers """

#import torch
from torch import bmm, nn
import torch.nn.functional as F
from model.ropend import RoPENd

class SelfAttention(nn.Module):
    """
    Self-attention transforms a sequence of vectors v_i to a sequence of vectors.  Each vector is embedded
    as  query Q(v_i), K(vi), and value V(v_i) vectors, where Q,K,V are learned linear (neural-ish?) transformations
    The i'th vector in the new sequence is a sum of the value vectors, where weights are how similar (dot product)
    the query vector for v_i is and the key vector for v_j.  Output is squence of same dimensions in this
    implementation.

    We’ll represent the input as a batch of sequences of vectors a sequence of t vectors of dimension k as a t by k
    matrix. Including a minibatch dimension b, gives us an input tensor of size
    (b,t,k).
    """

    def __init__(self, seqlen, k, oldk=None, heads=4, mask=False):
        # k is embedding dimension
        super().__init__()
        assert k % heads == 0
        self.k, self.heads, self.oldk = k, heads, oldk
        if self.oldk == None: self.oldk = k

        # the embedding functions (not necc. linear?)
        self.tokeys = nn.Linear(oldk, k, bias=False)
        self.toqueries = nn.Linear(oldk, k, bias=False)
        self.tovalues = nn.Linear(oldk, k, bias=False)

        self.posenc = RoPENd((seqlen, k))
        # This will be applied after the multi-head self-attention operation.
        self.unifyheads = nn.Linear(k, k)

    def forward(self, x):
        b, t, _oldk = x.size()  # t vectors of length k, each viewed as having h parts
        k = self.k
        h = self.heads

        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)

        keys = self.posenc(keys)
        queries = self.posenc(queries)
        """
        This gives us three vector sequences of the full embedding dimension k. As we
        saw above we can now cut these into h chunks. we can do this with a simple view
        operation:
        """

        s = k // h
        keys = keys.view(b, t, h, s)  # 2*s because added positional info
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)
        """
        Next reshape the tensors to add a dimension that iterations over the
        heads. For a single vector in our sequence you can think of it as reshaping a
        vector of dimension k into a matrix of h by k//h:
        """

        """
        Next, we need to compute the dot products. This is the same operation for every
        head, so we fold the heads into the batch dimension. This ensures that we can
        use torch.bmm() as before, and the whole collection of keys, queries and values
        will just be seen as a slightly larger batch.

        Since the head and batch dimension are not next to each other, we need to
        transpose before we reshape. (This is costly, but it seems to be unavoidable.)
        """

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        """ the dot products of keys and queries to get weights for values"""
        # Get dot product of queries and keys, and scale
        dot = bmm(queries, keys.transpose(1, 2))
        # -- dot has size (b*h, t, t) containing raw weights

        # scale the dot product
        dot = dot / (s ** (1 / 2))

        # normalize
        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights

        # apply the self attention to the values
        out = bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        # Note output is b X t X sh, where s*h is k, so same as input
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, seqlen, k, oldk=None, heads=None):
        super().__init__()
        if oldk == None: oldk = k
        if heads == None: heads = k
        self.attention = SelfAttention(seqlen, k, oldk=oldk, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended)  # used to + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)
