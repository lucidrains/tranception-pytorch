import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from einops_exts import rearrange_many
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# relative positional bias

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_sim):
        h, i, j, device = *qk_sim.shape[-3:], qk_sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_sim + self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent = False)

        return bias

# helper classes

class ReluSquared(nn.Module):
    """ found with neural architecture search in Primer paper """
    def forward(self, x):
        return F.relu(x) ** 2

def FeedForward(dim, mult = 4):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        ReluSquared(),
        nn.Linear(hidden_dim, dim)
    )

class CausalAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)

        self.alibi = AlibiPositionalBias(heads = heads)

        self.to_q = nn.Sequential(
            nn.Conv1d(dim, inner_dim, 1, bias = False),
            nn.Conv1d(inner_dim, inner_dim, 3, bias = False, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Conv1d(dim, inner_dim, 1, bias = False),
            nn.Conv1d(inner_dim, inner_dim, 3, bias = False, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Conv1d(dim, inner_dim, 1, bias = False),
            nn.Conv1d(inner_dim, inner_dim, 3, bias = False, groups = inner_dim)
        )

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = F.pad(x, (2, 0))

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = rearrange_many((q, k, v), 'b (h d) n -> b h n d', h = self.heads)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        alibi_bias = self.alibi(sim)
        sim = sim + alibi_bias

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# classes

class Tranception(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens = 21,
        heads = 8,
        dim_head = 64,
        ff_mult = 4
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CausalAttention(dim = dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(
        self,
        x,
        mask = None
    ):
        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.to_logits(x)
