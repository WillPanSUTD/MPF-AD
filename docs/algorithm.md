# Algorithm Details

This document gives a self-contained mathematical description of the two stages
implemented in this repository. Stage 1 is the depth-normal guided illumination
renderer in `Depth-Normal_Rendering/`; Stage 2 is the YOLO-WaveletTiny detector
under `YOLO-WT/ultralytics/`.

[ [中文版](algorithm.zh-CN.md) ]

---

## Stage 1 — Depth-Normal Guided Illumination Rendering

The renderer takes a single-channel 32-bit float depth map
$D(x, y) \in \mathbb{R}$ (sentinel `< -1e30` marks invalid pixels) and produces
a final BGR image $C'(x, y) \in [0, 255]^3$ that encodes both global and local
geometry as photometric contrast.

### 1.1 Statistical-prior depth → pseudo-color (LUT)

Implemented in [`LUT.cpp`](../Depth-Normal_Rendering/LUT.cpp).

**Goal.** Embed the global depth hierarchy of the weld surface into a stable
2D color image, without losing the small relative-depth differences caused by
defects.

**Adaptive bin width.** Use the Freedman–Diaconis rule with the inter-quartile
range $\mathrm{IQR} = Q_3 - Q_1$ of the valid depth values:

$$
h \;=\; \frac{2\,\mathrm{IQR}}{N^{1/3}},
\qquad
N_k \;=\; \left\lceil \frac{d_{\max} - d_{\min}}{h} \right\rceil.
$$

When $\mathrm{IQR} < 10^{-6}$ the depth map is essentially flat and we fall
back to $N_k = 1$. $Q_1$ and $Q_3$ are computed by linear interpolation on
the sorted vector of valid depth values.

**Histogram and CDF.** A standard 1-D histogram is accumulated:

$$
H(k) \;=\; \sum_{x,y}
\mathbb{I}\!\bigl(\,D(x, y) \in [d_{\min} + (k-1)h,\; d_{\min} + kh]\bigr),
\quad
\mathrm{CDF}(k) \;=\; \frac{1}{N}\sum_{a=1}^{k} H(a).
$$

**Lookup table.** A 256-entry BGR LUT is constructed once (deterministically)
from a piecewise blue → green → red gradient:

$$
\mathrm{LUT}[i] \;=\;
\begin{cases}
\bigl(\,255(1-2t),\; 255 \cdot 2t,\; 0\bigr) & t < 0.5,\\[2pt]
\bigl(\,0,\; 255(1-2(t - 0.5)),\; 255\cdot 2(t-0.5)\bigr) & t \ge 0.5,
\end{cases}
\qquad t = i/255.
$$

**Per-pixel mapping.** Implementation note: the production code applies the
LUT after a min–max normalization of the valid depth into $[0,255]$, which is
equivalent to applying the LUT on the index
$i(x,y) = \lfloor \mathrm{CDF}(k(x,y)) \cdot 255 \rfloor$ for a uniformly
binned histogram. Either way the output is

$$
C(x, y) \;=\; \mathrm{LUT}\bigl[i(x, y)\bigr]\;\in\; [0,255]^3,
$$

with invalid pixels written to black.

**Why this matters.** Defects with sub-millimeter height differences would
otherwise collapse onto adjacent integer values; the data-adaptive bin width
preserves their relative ordering. A deterministic LUT (no learned mapping)
also keeps the encoding interpretable across samples.

### 1.2 Surface normal estimation

Implemented in [`Phong.cpp::computeNormals`](../Depth-Normal_Rendering/Phong.cpp).

For each valid pixel $P(x, y) = (x, y, d(x, y))$ we collect 3D points from the
8-neighborhood (or 4-neighborhood, switchable). To avoid scale mismatch between
pixel offsets (1 px each) and depth values (millimeters), we apply an automatic
$z$ rescale,

$$
s \;=\; \frac{512}{d_{\max} - d_{\min}},
\qquad
\bigl(\Delta x,\; \Delta y,\; s \cdot \Delta d\bigr),
$$

so the three coordinate components live in comparable ranges. The normal is
then estimated robustly:

| Valid neighbours | Method |
|------------------|--------|
| ≥ 3              | PCA on the local 3-D neighborhood; the normal is the eigenvector of the covariance matrix corresponding to the smallest eigenvalue. |
| = 2              | Cross product $\vec{v}_1 \times \vec{v}_2$ of the two relative vectors. |
| ≤ 1              | Inherit the normal of the nearest valid pixel with the largest depth difference (search radius 1). |

The result is normalized and forced to face the camera ($n_z \ge 0$). Producing
a stable normal at *every* valid pixel — including object boundaries — is
essential because Phong shading involves the dot product
$\vec{N} \cdot \vec{L}$ and is therefore sensitive to spurious normals.

### 1.3 Phong shading

Implemented in [`Phong.cpp::renderPhong`](../Depth-Normal_Rendering/Phong.cpp).

Given the per-pixel unit normal $\vec{N}(x, y)$, normalized light direction
$\vec{L}$ and view direction $\vec{V}$, we compute the three Phong components

$$
\boxed{
\begin{aligned}
I_a &= w_a I_l, \\
I_d &= w_d I_l \,\max(\vec{N}\cdot\vec{L},\,0), \\
I_s &= w_s I_l \,\max(\vec{R}\cdot\vec{V},\,0)^{\,n},
\end{aligned}}
\qquad
\vec{R} \;=\; 2(\vec{N}\!\cdot\!\vec{L})\,\vec{N} - \vec{L}.
$$

The total per-pixel intensity is clamped to $[0, 1]$,

$$
I(x, y) \;=\; \min\bigl(1,\; I_a + I_d + I_s\bigr),
$$

and modulates the pseudo-color image:

$$
\boxed{\;C'(x, y) \;=\; I(x, y) \cdot C(x, y)\;}.
$$

Invalid pixels (no normal, or supplied via the explicit mask) are forced to
black. Because $I$ depends on $\vec{N}\cdot\vec{L}$, a small normal
perturbation at a micro-defect produces a large photometric change — this is
the "geometric-to-photometric amplification" the paper relies on.

**Default parameters used in the paper:**

| Symbol | Code variable | Value |
|--------|---------------|-------|
| $\vec{L}$ | `light`        | `(500, 150, 1500)` |
| $\vec{V}$ | `view`         | `(500, 150, 1500)` |
| $I_l$     | `lightIntensity` | `1.3` |
| $w_a$     | `ambient`      | `0.1` |
| $w_d$     | `kd`           | `0.5` |
| $w_s$     | `ks`           | `0.3` |
| $n$       | `shine`        | `32`  |

---

## Stage 2 — YOLO-WaveletTiny

YOLO-WT is a YOLOv11n derivative with two wavelet-aware modules. The full
backbone/neck wiring is in
[`cfg/models/YOLO-WT/YOLO-WT.yaml`](../YOLO-WT/ultralytics/cfg/models/YOLO-WT/YOLO-WT.yaml).
At scale `n` it ships **2.47 M parameters / 5.7 GFLOPs**.

### 2.1 WDSConv — Wavelet Depthwise-Separable Convolution

Implemented in [`AddModules/WDSConv.py`](../YOLO-WT/ultralytics/nn/AddModules/WDSConv.py).
Used at the P3/8 and P4/16 downsampling positions.

WDSConv keeps the depthwise-separable form (depthwise → pointwise), but the
depthwise core is replaced by a 3-level Haar wavelet block (`WTConv`).

**One level of decomposition.** With the four orthonormal 2×2 Haar filters,

$$
f_{LL} = \tfrac{1}{2}\!\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix},
\quad
f_{LH} = \tfrac{1}{2}\!\begin{pmatrix} 1 & -1 \\ 1 & -1 \end{pmatrix},
\quad
f_{HL} = \tfrac{1}{2}\!\begin{pmatrix} 1 & 1 \\ -1 & -1 \end{pmatrix},
\quad
f_{HH} = \tfrac{1}{2}\!\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix},
$$

the wavelet transform is implemented as a strided grouped convolution
(`groups = C`, stride 2):

$$
\bigl[X_{LL}, X_{LH}, X_{HL}, X_{HH}\bigr] \;=\; \mathrm{WT}(X).
$$

Each subband is then refined by a $K\times K$ depthwise convolution and a
learnable per-channel scale (`_ScaleModule`) initialized at $0.15$ for the
high-frequency bands. Reconstruction uses the orthonormality of the Haar basis
and is implemented as grouped *transposed* convolution (the inverse wavelet
transform, IWT). The procedure is recursed to depth 3, so the receptive field
expands geometrically while the parameter count stays low.

The WDSConv output fuses the wavelet branch back with a parallel base depthwise
conv and projects the result through a $1{\times}1$ pointwise conv:

$$
Y \;=\; \mathrm{PWConv}\bigl(\,\sigma_{\text{base}}(W_{\text{base}} * X) + \mathrm{IWT}(\cdot)\,\bigr).
$$

Practically, WDSConv lets the same convolution simultaneously expand the
receptive field (via wavelet decomposition) and disentangle low- vs.
high-frequency features — high-frequency subbands carry the cracks, pinholes,
and pit edges that are the targets of inspection.

### 2.2 IWUpsample — Inverse-Wavelet Upsampling

Implemented in [`AddModules/IWUpSample.py`](../YOLO-WT/ultralytics/nn/AddModules/IWUpSample.py).
Replaces both `nn.Upsample` calls in the YOLOv11n neck.

Conventional bilinear / nearest upsampling smears the high-frequency cues that
WDSConv just isolated. IWUpsample reverses the wavelet decomposition instead.

Given an input $X \in \mathbb{R}^{B \times C \times H \times W}$ where the
channel dimension already encodes $J$ levels of wavelet subbands
($C = 4^J \cdot C_0$), `_split_coeffs` slices it into

$$
y_L \in \mathbb{R}^{B\times C_0 \times H\times W},\qquad
\{y_H^{(j)}\}_{j=1}^{J},\quad y_H^{(j)}\in\mathbb{R}^{B\times C_0\times 3\times H\times W},
$$

corresponding to one low-frequency component and three high-frequency
sub-bands per level. It then iterates

$$
\mathit{LL}_{j} \;=\; \mathrm{IWT}\!\bigl(\mathit{LL}_{j+1},\,\mathit{LH}_{j+1},\,\mathit{HL}_{j+1},\,\mathit{HH}_{j+1}\bigr),
$$

doubling the spatial resolution at every step until the original feature map
size is restored. A final $1{\times}1$ Conv–BN–ReLU adapts the channel count to
the value requested by the YAML.

Because the Haar IWT is exactly the adjoint of the WT used inside WDSConv, the
upsampling is information-preserving in the wavelet sense — small defect
textures reach the detection head without low-pass smoothing.

### 2.3 Putting it together

The YOLO-WT YAML inserts WDSConv at the two stride-2 downsamples in the
backbone (positions P3/8 and P4/16), and replaces the two upsamples in the
neck with IWUpsample (channels 256 → 128 → output). Everything else
(`C3k2`, `SPPF`, `C2PSA`, the three `Detect` heads) is inherited from
YOLOv11n. Training uses standard ultralytics defaults — SGD, 250 epochs,
batch 16, image size 640, no pretrained weights — kept identical to the
baseline so that the modality and module ablations are clean.

---

## Reference

Finder, S. E., Amoyal, R., Treister, E., & Freifeld, O.
*Wavelet Convolutions for Large Receptive Fields.* ECCV 2024.
([arXiv:2407.05848](https://arxiv.org/abs/2407.05848)) — origin of the
wavelet-conv idea adapted in WDSConv.
