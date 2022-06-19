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

class LearnedAlibiPosBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.slopes = nn.Parameter(slopes)
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

class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class CausalAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        ds_conv_kernel_sizes = (0, 3, 5, 7) # heads were grouped into 4 groups and given a depthwise conv after the queries / keys / values projection
    ):
        super().__init__()
        self.groups = len(ds_conv_kernel_sizes)
        assert heads >= self.groups and (heads % self.groups) == 0, f'heads must be greater than {self.groups} and divisible by {self.groups}'

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.heads_per_group = heads // self.groups

        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)

        # ds convs with different kernel sizes for 4 groups of heads

        self.qkv_ds_convs = nn.ModuleList([])

        for _ in range(3): # for queries, keys, values
            ds_convs = nn.ModuleList([])

            for kernel_size in ds_conv_kernel_sizes:
                if kernel_size == 0:
                    ds_convs.append(nn.Identity())
                    continue

                ds_convs.append(CausalDepthwiseConv1d(dim_head * self.heads_per_group, kernel_size))

            self.qkv_ds_convs.append(ds_convs)

        # learned alibi positional bias for 4 groups of heads

        self.learned_alibi_pos_biases = nn.ModuleList([LearnedAlibiPosBias(heads = self.heads_per_group) for _ in range(self.groups)])

        # outward projection

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        device, heads_per_group = x.device, self.heads_per_group

        x = self.norm(x)
        x = rearrange(x, 'b n d -> b d n')

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        q, k, v = rearrange_many((q, k, v), 'b (h d) n -> b h d n', h = self.heads)

        # apply causal depthwise conv to queries, keys, values (a la Primer) with different kernel sizes across 4 groups of heads

        def apply_causal_ds_conv_to_grouped_heads(args):
            projs, ds_convs = args
            batch = projs.shape[0]

            projs = rearrange_many(projs.split(heads_per_group, dim = 1), 'b h d n -> b (h d) n')
            conv_out = [fn(t) for fn, t in zip(ds_convs, projs)]
            conv_out = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h = heads_per_group), conv_out)
            conv_out = torch.cat(tuple(conv_out), dim = 1)
            return rearrange(conv_out, 'b h d n -> b h n d')

        q, k, v = map(apply_causal_ds_conv_to_grouped_heads, zip((q, k, v), self.qkv_ds_convs))

        # scale and similarity

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # learned alibi pos bias across 4 groups of heads
        # so heads specialize to looking at different distances of kmers

        grouped_sims = sim.split(self.heads // self.groups, dim = 1)
        grouped_sims = [(alibi(sim_group) + sim_group) for alibi, sim_group in zip(self.learned_alibi_pos_biases, grouped_sims)]

        sim = torch.cat(grouped_sims, dim = 1)

        # causal mask

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention, but of course

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

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
        ff_mult = 4,
        ds_conv_kernel_sizes = (0, 3, 5, 7)
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CausalAttention(dim = dim, heads = heads, dim_head = dim_head, ds_conv_kernel_sizes = ds_conv_kernel_sizes),
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
