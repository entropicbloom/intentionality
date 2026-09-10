# What the correlations between neurons say about each neuron

*Draft v0.1. Figures refer to `outputs/paper/` (Figs. 1–6 main text, A1–A5 appendix).*

## Abstract

Record many neurons while they watch the same stimulus, then throw the stimulus away. What is left is how the neurons relate to each other: a correlation matrix. We ask whether that matrix alone says what each neuron represents. A transformer that sees only the correlation matrix of 512 sampled neurons, with no labels and no reference population in its input, assigns preferred orientation to neurons of mouse visual cortex at 0.38 accuracy (8 classes, majority class 0.26) and receptive-field position at R² 0.23, from in-vivo responses in the MICrONS volume; on the digital-twin substrate it reaches 0.42 and 0.48. The readout is possible because the relations between orientation classes are close to circulant, which fixes the class structure up to rotation and reflection, and a small anisotropy pins the absolute frame. Across 33 mice of the Allen Brain Observatory, orientation transfers to unseen animals only when a sampled population mixes animals, so that its matrix contains between-animal correlations; populations drawn from one animal sit at the baseline whether or not the animal was seen in training. For receptive fields, what crosses animals is where each animal's imaged patch looks on the screen, and not the layout of neurons within it. Capacity helps where labels are plentiful and stops helping where they are scarce.

## 1. Introduction

A neuron's tuning is measured by relating its activity to a known stimulus. Suppose the stimulus is unknown. Many neurons were recorded together, they saw the same thing, and only their joint activity remains. The stimulus-aligned route to tuning is closed. What remains is relational: how strongly each pair of neurons covaries. This paper asks how much of each neuron's content can be read from those relations alone.

The question has two parts. The first is about neurons: can each neuron's orientation and receptive-field position be read from the correlation matrix of a population it belongs to, with no label and no reference neuron in the input. The second is about generalisation: does a decoder trained on the relations of some neurons read the relations of other neurons, and of other brains. Between the two sits an explanation. Aggregating the correlations by orientation class exposes a symmetry: the structure between classes is fixed only up to a rotation or reflection of the orientation circle, and a small departure from that symmetry is what makes an absolute readout possible at all.

We answer these on two public datasets. The MICrONS functional-connectomics release gives 12,894 neurons from one cubic millimetre of mouse visual cortex, each with in-vivo responses to a shared natural-movie stimulus, a fitted digital-twin model, a preferred orientation and a receptive-field centre (Fig. 1). The Allen Brain Observatory gives 33 mice recorded under identical stimuli, which makes correlations between neurons of different animals defined and lets us ask what transfers between brains.

Our decoder is a transformer whose only input is the standardised correlation matrix of a sampled population, one row per neuron. Labels enter as training targets and nowhere else. Every score below is on neurons, or animals, that the training and the model selection never touched.

## 2. Methods

### 2.1 Data

**MICrONS.** Ding, Fahey, Papadopoulos et al. (2025) release 12,894 neurons co-registered to the electron-microscopy volume, from 13 two-photon scans across V1, RL, AL and LM. Each neuron has a trial-averaged in-vivo response to the oracle natural-movie clips (120 bins), shown in every scan; a digital-twin response to a shared movie (4,999 bins, which we compress to 512 principal components; the Gram is preserved to r > 0.999); an in-vivo preferred orientation with a global orientation selectivity index (gOSI); and a receptive-field centre from the twin (spike-triggered average fit, screen coordinates in [-1, 1]). We use orientation for the 5,287 neurons with gOSI ≥ 0.25, binned into 8 classes, and receptive-field position for the 11,326 neurons whose twin test correlation is ≥ 0.2. Same-scan pairs correlate about 50 % more strongly than cross-scan pairs at matched receptive-field distance; the content of the correlations is the same across scans, and all populations mix scans.

**Allen.** From Visual Coding 2P we take the 36 VISp excitatory containers with at least 150 orientation-labelled cells, 33 mice and 9,281 cells. Relations come from session A: responses to natural movies shown to every mouse (4,500 bins), or the 40 blank-subtracted condition means of drifting gratings (8 directions × 5 temporal frequencies). Orientation labels come from static gratings in session B (6 classes, cell table, gOSI ≥ 0.25, p < 0.05), so labels and relations use different stimuli. Receptive-field centres come from locally sparse noise in session C (1,504 cells with a significant fit, 23 to 112 per mouse).

### 2.2 Relations

Each neuron's response vector is centred and normalised; the relation between two neurons is the cosine of their vectors, a signal correlation. For a population of n neurons the Gram is the n × n matrix of these values, standardised by the global off-diagonal mean and standard deviation, with the diagonal zeroed.

### 2.3 Decoder

The decoder is a transformer. Each row of a population's Gram, together with three row statistics, is projected to a token; L pre-norm blocks with H-head attention update the tokens; a linear head reads out an orientation class or an (x, y) pair per token. Loss is cross-entropy or mean squared error on the labelled tokens only; unlabelled neurons stay in the population and receive no loss. The standard configuration is width 512, 8 layers, 17M parameters, populations of 512 neurons (MICrONS) or 256 (Allen). Smaller (2.2M) and larger (57M) models are reported in the appendix. Training draws thousands of random populations per epoch. Populations for Allen orientation are augmented by recomputing each training population's Gram from a random half of the 40 conditions; test Grams use all conditions.

### 2.4 Protocol

**Split.** MICrONS neurons are halved at random. Training populations are sampled inside the training half, test populations inside the test half, so no test neuron ever appears in a training population (Fig. 1d). A slice of the training half is held out to choose the epoch; the test half never influences selection.

**Regimes (Allen).** Two independent choices define a run (Fig. 4). The split decides where test neurons come from: the training animals, each halved into training and test neurons, or held-out animals. The population decides what a sample is: neurons drawn from one animal, so that the Gram is a within-circuit matrix, or from several animals, so that most entries are between-animal correlations. We call the four cells `within`, `pooledwithin`, `cross` and `pooledcross`. Held-out regimes select the epoch on further held-out mice. Every held-out cell is run on three mouse splits; every training-animal cell on three neuron splits.

**Baselines.** The majority class (0.255 in MICrONS, 0.19 to 0.20 on Allen test mice) and R² = 0.

### 2.5 Class-level tests

For the symmetry analysis we average the Gram into a K × K class-Gram, the mean relation between neurons of two orientation classes, and match a held-out half's class-Gram to a reference over all K! relabellings by Frobenius distance. Accuracy is the fraction of classes that land on their true label; accuracy modulo the dihedral group D_K counts a relabelling as correct if it differs from the truth by a rotation or a reflection of the orientation circle.

## 3. Results

### 3.1 Per-neuron content from the correlation matrix alone

On MICrONS the decoder reads orientation and receptive-field position from the Gram of 512 neurons of the test half (Fig. 2). At 17M parameters, three seeds:

| content, substrate | accuracy or R² | baseline |
|---|---|---|
| orientation, in vivo | 0.380 ± 0.006 | 0.255 |
| orientation, digital twin | 0.420 ± 0.003 | 0.255 |
| receptive field, in vivo | 0.234 ± 0.007 | 0 |
| receptive field, digital twin | 0.480 ± 0.010 | 0 |

The twin substrate is cleaner than the in-vivo one (the twin removes trial noise), and receptive fields carry more than orientation. The predictions are in the true frame: orientation accuracy modulo D_8 equals plain accuracy in every run, so the decoder is not recovering the class structure up to a rotation, it is placing each neuron at its absolute orientation.

### 3.2 Why the absolute frame is recoverable

The class-Gram of orientation is close to circulant (Fig. 3): neighbouring orientations correlate, orthogonal ones anti-correlate, and the pattern repeats around the circle. The circulant part holds 71 % of the variance in vivo and 81 % in the twin. Matching a held-out half to a reference over all 8! relabellings gives accuracy 1.0 with the raw class-Gram. After projecting onto the circulant part, absolute accuracy drops to chance (0.11) while accuracy modulo D_8 stays at 1.0. The circulant structure determines the classes up to the symmetries of the circle. The residual anisotropy, a cardinal bias whose strength differs between areas, pins which class is 0°. A label-free decoder that outputs absolute orientation must use it, and the results of 3.1 show that it does.

### 3.3 Orientation transfers between animals through mixed populations

With natural-movie relations, orientation is nearly absent from Allen correlations: same-orientation pairs correlate 0.005 more than orthogonal pairs, against 0.09 in MICrONS, and every test is at chance. With grating relations the signal is present (0.05 within a mouse, 0.044 across mice), and the decoder's four cells separate (Fig. 5):

| orientation, 6 classes, 3 splits each | single-animal populations | mixed populations |
|---|---|---|
| test neurons from the training animals | 0.210 / 0.216 / 0.217 | 0.281 / 0.283 / 0.283 |
| test neurons from held-out animals | 0.199 / 0.197 / 0.187 | 0.293 / 0.236 / 0.264 |
| baseline | 0.19–0.20 | 0.19–0.20 |

Single-animal populations sit at the baseline whether or not the animal was seen in training. Mixed populations are above it in both rows, at the same level. Transfer to a new brain costs nothing; what matters is whether the Gram contains between-animal correlations. Those exist only because all mice saw the same 40 conditions, and they measure how similar two tuning profiles are. The decoder still never sees which condition is which, so the absolute frame must again come from the population's relational structure. In this dataset a neuron's orientation is readable from where it sits in a correlation geometry that spans animals; it is not readable from its relations inside its own circuit.

### 3.4 Receptive fields: what crosses animals is the animal's screen position

Correlation falls with receptive-field distance within a mouse (0.127 at < 5° to 0.068 beyond 40°) and across mice (0.049 to 0.011). Decoding absolute screen coordinates on held-out mice gives R² that swings with the split, from 0.29 to −0.45 for single-animal populations and from 0.21 to −0.25 for mixed ones. Decomposing the saved predictions explains the swing (Fig. 6). The correlation between predicted and true mean position of each held-out mouse, pooled over 27 mouse-level predictions, is 0.36 to 0.71 (permutation p ≤ 0.03). The correlation between predicted and true position of a neuron relative to its mouse-mates is 0.0 to 0.1 over 1,259 cells. The decoder recovers where an animal's imaged patch looks, a property of the population, and not which neuron within it looks where. R² on absolute coordinates is then dominated by whether the between-mouse spread of a particular test set is reproduced at the right scale.

The within-mouse layout is present in the correlations, since same-mouse correlation falls with distance. It is not learned because each mouse contributes 23 to 112 labelled cells. MICrONS, with 11,326 labels in one animal, is where the same decoder reaches neuron-level receptive fields at R² 0.48.

### 3.5 What moves the numbers

Capacity pays where labels are plentiful and stops where they are scarce (Fig. A1). Twin receptive fields go from 0.34 to 0.48 to 0.51 R² between 2.2M, 17M and 57M parameters; twin orientation is flat at 0.42 from 7M upward; in-vivo orientation gains one point. Every model above 2M memorises its training neurons within 4 to 17 epochs (training loss 0.001 to 0.06) and is early-stopped on the held-out selection slice (Fig. A4). Population size saturates by 512 neurons in MICrONS and 256 on Allen (Fig. A2). Recomputing training Grams from a random subset of stimulus conditions helps only with capacity, and only where the dimensions are exchangeable conditions: it lifts Allen orientation at 17M and in-vivo receptive fields at 75 % of the bins, hurts the 2M model below 75 %, and destroys a Gram built on principal components (Fig. A3). On Allen, the 17M augmented model and the 2.2M standard model are equal within seed noise (0.293 ± 0.012 against 0.286 to 0.294), and split-to-split variance exceeds both (Fig. A5).

## 4. Discussion

The correlation matrix of a few hundred cortical neurons, with the stimulus unknown and no neuron labelled, determines each neuron's preferred orientation to within a few points of what a fully labelled linear readout achieves in vivo, and its receptive-field position to R² 0.23 in vivo and 0.48 on the digital twin. The mechanism is the one the symmetry analysis exposes: the relations fix the class structure up to the symmetries of the content, and a small, consistent departure from that symmetry fixes the frame.

Across animals the picture splits by content. Orientation lives in the relations between tuning profiles, which exist between any two neurons that saw the same conditions, in the same brain or not; it does not live in the within-circuit correlations of a single animal at the population sizes available here. Receptive-field position is carried at the level of an animal's patch, and the neuron-level layout needs more labels per animal than these recordings provide. Both statements are about what correlations under a shared stimulus contain, and both are measured under a protocol in which the test neurons, and in the held-out regimes the test animals, never touch training or model selection.

Three limits. MICrONS is one animal, and its training and test halves share recording sessions; the Allen held-out-animal cells are the cleaner generalisation test, and their variance across mouse splits is large. Orientation labels are noisy in both datasets (in vivo and twin labels agree on 70 % of MICrONS neurons at 8 classes; static and drifting gratings on 45 to 67 % of Allen cells), which caps every labelled and label-free number alike. The twin substrate is a model fitted to the same recordings, so the in-vivo results carry the claim about cortex and the twin results show what cleaner correlations would give.

The symmetry result is an instance of a general expectation: a representation that is equivariant to a group has a group-fixed part, identical across individuals up to relabelling, and an individual remainder (Oizumi, Lim and Kanai 2026). The matching-over-relabellings test is the discrete form of unsupervised alignment of similarity structures across individuals by optimal transport. What this paper adds is the measurement in cortex, at the level of single neurons, and the finding that the remainder, the anisotropy, is what makes the absolute readout possible.

## Appendix

- A1: test metric versus decoder parameters, both datasets.
- A2: test metric versus population size.
- A3: relation augmentation by condition subsampling.
- A4: training dynamics and epoch selection.
- A5: Allen split variance and the class-level test by relation type.
- Labelled reference models (ridge readouts from correlations to labelled training neurons) and their caveats.
- Held-out-animal RF decomposition per split.
