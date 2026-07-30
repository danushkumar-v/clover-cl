# Real pretrained backbones: what changed, and why

Status: **in progress (P11).** The "before" columns are final and were
measured; the "after" columns are filled in as each method family lands.

## The problem this document records

CLOVER v2 was built and tested against `TinyViT` — a clean-room minimal
Vision Transformer (`depth=2`, `embed_dim=16`, 4 patches, randomly
initialised) sized for the 8×8 synthetic dataset that the CI safety gate
runs on. That was a deliberate choice: it makes the whole test suite run on
CPU in seconds, and it let every method be validated for *structural*
correctness — finite losses, revisit-safe label handling, resume fidelity —
without a GPU or a dataset download.

It also meant that until now, **only methods whose backbone contract is a
plain `forward(x) -> features` could use a real pretrained checkpoint.**
That is exactly one of the nine: SimpleCIL. The other eight are
prompt-based (L2P, DualPrompt, CODA-Prompt) or adapter-based
(APER-Adapter, RanPAC, EASE, MOS, TUNA), and they all work by splicing
*into* a transformer's internals — prepending prompt tokens, injecting
prefix key/value pairs into specific attention blocks, or hanging parallel
bottleneck branches off each block's MLP. A stock `timm`
`VisionTransformer` offers no way in: its `Block.forward` accepts only `x`,
and its `Attention.forward` has no key/value prefix.

The measured consequence, from two completed cluster batches on CIFAR-100:

| batch | backbone | methods | AIA |
|---|---|---|---|
| `matrix_simplecil_vitb16` | ViT-B/16, ImageNet-pretrained | SimpleCIL only | **0.82** |
| `matrix_full` | TinyViT, random init | all 9 | **0.08 – 0.11** |

SimpleCIL is the same code in both, so the 10× gap is the backbone alone.
Chance accuracy on 100 classes is 0.01, and a stream-averaged AIA around
0.08–0.11 is what an untrained feature extractor produces. `matrix_full`
therefore validates that the *framework* runs end to end at scale — 189/189
runs completed on Snellius, resumable, with finite metrics throughout — but
its accuracies are not a method comparison and must never be presented as
one.

## Three layers of change (not one)

Wiring a real backbone in is not a single patch. Three independent things
were sized for TinyViT:

**1. The hook surface.** `TinyViT` exposes `patch_tokens`,
`query_features`, `forward_tokens(tokens, prefix_kv=, adapter=)` and
`forward(x, prompt_tokens=, prefix_kv=, adapter=)`. timm exposes none of
them. Fix: a wrapper that re-implements timm's block and attention forward
*using timm's own submodules* — so pretrained weights load and behave
normally — with the two hook points inserted at exactly the positions
`TinyViT` uses. Semantic fidelity is the bar: the adapter branch reads the
post-attention residual stream (not `norm2(x)`), and `prefix_kv` is
concatenated onto K/V after the q/k norms with queries untouched. A method
must be structurally identical on both backbones; only scale differs.

**2. Scale-dependent structural defaults.** Several methods carry constants
that are only meaningful at TinyViT's dimensions. DualPrompt's default
general/expert prompt layers are `(0,)` and `(1,)` — which exist because
TinyViT has exactly two blocks. RanPAC's random-projection width was
reduced from the published 10000 to 256 to suit a 16-dimensional feature.
Adapter bottlenecks are 8 wide against a 16-dimensional stream. Run these
unchanged on ViT-B/16 and nothing crashes — the methods simply stop being
the published methods. Fix: derive every such constant from the base
model's own `depth`/`feature_dim`, defaulting to the published value at
ViT-B/16 scale.

**3. Training hyperparameters.** The safety gate uses `epochs=40`,
`lr=3e-2` — values found empirically to make a *randomly initialised*
TinyViT converge on synthetic 8×8 images, and documented as such. They are
actively harmful for a pretrained ViT. Fix: per-method defaults carried as
data from the verified LAMDA-PILOT mirrors in the private bench repo
(`bench/configs/methods/*.yaml`), with provenance recorded per method.

## Per-method change log

Filled in as each family lands. "Before" = state at commit `7ffe941`.

| method | family | before (TinyViT-scale) | after (ViT-B/16) | why |
|---|---|---|---|---|
| SimpleCIL | none (plain forward) | already worked with a real timm ViT | unchanged | its base *is* its backbone; no splice needed |
| L2P | prompt pool | _tbd_ | _tbd_ | _tbd_ |
| DualPrompt | prefix K/V | `g_layers=(0,)`, `e_layers=(1,)` — fits depth 2 only | _tbd_ | layer indices must derive from depth |
| CODA-Prompt | prefix K/V, soft | pool sliced per experience at tiny width | _tbd_ | _tbd_ |
| APER-Adapter | adapter | bottleneck 8 | _tbd_ | _tbd_ |
| RanPAC | adapter + RP head | `M=256` (published: 10000) | _tbd_ | reduced only to suit a 16-dim feature |
| EASE | growing adapters | one set per experience, 2 blocks | _tbd_ | head is `[classes, blocks, block_dim]` |
| MOS | EMA-merged adapter | bottleneck 8 | _tbd_ | _tbd_ |
| TUNA | EMR-merged adapters | bottleneck 8 | _tbd_ | _tbd_ |

## Configuration surface

Two different keys, and the distinction matters:

```yaml
# SimpleCIL has no mechanism wrapper -- its base IS its backbone:
method: {name: simplecil, backbone: vit_base_patch16_224, pretrained: true}

# The other eight keep their mechanism and swap the base *underneath* it:
method: {name: l2p, base_model: vit_base_patch16_224, pretrained: true}
```

Setting `backbone:` on a wrapper method replaces the whole mechanism with a
bare ViT — the prompt pool or adapter stack silently disappears, and the
method degenerates to a linear probe. The shipped configs use the correct
key per method; if you write your own, check this first.

## Parameter budget at ViT-B/16 scale

_Filled in by P11-C: trainable parameters per experience and total stored
state after a 10-experience stream, per method. Recorded because several
methods store per-experience adapter sets, and unbounded growth is a
property worth reporting rather than discovering on a cluster._
