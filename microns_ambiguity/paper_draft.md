# What the correlations between neurons say about each neuron

*Draft v0.2. Figures refer to `outputs/paper/` (Figs. 1–6 main text, A1–A5 appendix).*

## Abstract

Record many neurons while they watch the same stimulus, then discard the stimulus. What remains is how the neurons relate to each other: a correlation matrix. We ask whether that matrix alone says what each neuron represents. A transformer that sees only the correlation matrix of 512 sampled neurons, with no labels and no reference population in its input, decodes the preferred orientation of neurons of mouse visual cortex to a mean error of 25° (an uninformed decoder: 45°) and their receptive-field position at R² = 0.23, from in-vivo responses in the MICrONS volume; on the digital-twin substrate it reaches 20° and 0.48. The readout is possible because the relations between orientation classes are close to circulant, which fixes the class structure up to rotation and reflection, and a small anisotropy pins the absolute frame. Across 33 mice of the Allen Brain Observatory, orientation transfers to unseen animals only when a sampled population mixes animals, so that its matrix contains between-animal correlations; populations drawn from one animal stay within 3° of chance whether or not the animal was seen in training. For receptive fields, what crosses animals is where each animal's imaged patch looks on the screen, and not the layout of neurons within it. Capacity helps where labels are plentiful and stops helping where they are scarce.

## 1. Introduction

What a neuron in visual cortex represents is established by relating its activity to the stimulus: a neuron prefers vertical gratings, or responds to the upper left of the screen. This paper asks whether the same facts are contained in a description that mentions no stimulus at all, the correlations between the neuron and the other neurons recorded with it. Take a population that watched the same movie, compute the correlation of every pair of neurons, and discard the movie. Does the resulting matrix determine which neuron prefers which orientation, and where each neuron's receptive field lies?

The idea that content can be read from relations is old. Shepard and Chipman proposed that internal representations need not resemble what they stand for as long as the similarities among them mirror the similarities among the things represented, a second-order isomorphism (Shepard and Chipman 1970); Edelman made the case that representation is representation of similarities (Edelman 1998). Representational similarity analysis turned the idea into a method that compares brains, models and species through their similarity structures rather than their coordinates (Kriegeskorte et al. 2008; Kriegeskorte and Kievit 2013), hyperalignment builds a shared representational space across subjects from those structures (Haxby et al. 2011), and the same principle underlies current work on aligning representations across networks and individuals (Sucholutsky et al. 2023; Huh et al. 2024). In neuroscience, the correlation structure of spontaneous activity was shown to carry the orientation map of visual cortex without any stimulus (Tsodyks et al. 1999; Kenet et al. 2003) and, more generally, to match the statistics of evoked activity (Berkes et al. 2011); functionally similar neurons connect preferentially (Ko et al. 2011; Ding et al. 2025); and population dynamics are preserved across animals performing the same behaviour (Safaie et al. 2023). Structuralist accounts of experience make the general claim that what a state represents, or how it feels, is fixed by its position in a space of relations (Lyre 2022), with Kleiner and Ludwig (2024) giving a formal definition of such structures; Lässig (2025) formulates the corresponding question for neural networks and gives the decoding framework we use here.

The spontaneous-activity results are the closest precedent, and they set the premise: a neuron's correlations with the population contain its tuning even in the dark. Tsodyks et al. (1999) showed it for single neurons by averaging the imaged population activity at the times a neuron fired and finding the evoked map of its preferred orientation. Reading the tuning out of that pattern, however, required the evoked map, a stimulus-labelled reference: correlations sort neurons by orientation and place the groups in order around the orientation circle, but on their own they say nothing about which group is vertical. The same holds for representational similarity analysis and hyperalignment, whose structures are over conditions or subjects and whose interpretation goes through a known stimulus set. Our question is one level down and one step further: whether the relations among *neurons*, with the stimulus that produced them discarded and no labelled neuron or map to compare with, fix the content of each neuron, and whether that content is readable at the scale of thousands of individually identified neurons and across animals. The closest neighbours in machine learning for neural data read properties of a neuron from population activity: NeuPRINT learns a time-invariant embedding of each neuron from the dynamics of its population and decodes cell type from it (Mi et al. 2023), NuCLR does the same with a contrastive objective and a permutation-equivariant transformer, and transfers cell type and brain region to unseen animals (Arora et al. 2025), and POYO trains one decoder across sessions and animals by tokenising spikes with learned unit embeddings (Azabou et al. 2023). These models see the activity itself; the present decoder sees only the correlation matrix, so the question it answers is what the relations alone determine, and the content it targets is stimulus tuning rather than cell class. In practice the relational situation is often the one at hand: recordings under natural behaviour have no controlled stimulus, and neurons from different sessions or animals share no stimulus-aligned coordinate system, but they do share the structure of their correlations. In principle, the answer says how much of a neuron's content is fixed by its position in the network of relations among neurons.

We answer it with a decoder. The input is the correlation matrix of a sampled population of a few hundred neurons, one row per neuron, and nothing else: no labels and no reference neurons. The output is a preferred orientation and a receptive-field position for every neuron in the population. The decoder is trained on populations whose labels are known and scored on neurons it never saw. We do this at two scales. Within one brain, using the MICrONS release (Ding et al. 2025; MICrONS Consortium 2025), 12,894 neurons from a cubic millimetre of mouse visual cortex with in-vivo responses to a shared movie, a fitted digital-twin model (Wang et al. 2025), and labels for orientation and receptive field (Fig. 1): the decoder trained on half the neurons reads the other half. Across brains, using the Allen Brain Observatory (de Vries et al. 2020), 33 mice recorded under identical stimuli: the decoder trained on some animals reads neurons of animals it never saw.

The results come with an explanation. Averaging the correlations by orientation class shows that the relations between classes are almost circulant, which fixes the classes only up to a rotation or reflection of the orientation circle; a small, consistent departure from that symmetry is what pins the absolute frame and makes the readout possible.

## 2. Methods

### 2.1 Data

**MICrONS.** The release of Ding et al. (2025) contains 12,894 neurons co-registered to the electron-microscopy volume, from 13 two-photon scans across V1, RL, AL and LM. Each neuron has a trial-averaged in-vivo response to the oracle natural-movie clips (120 bins), shown in every scan; a digital-twin response to a shared movie (4,999 bins, which we compress to 512 principal components; the Gram is preserved to r > 0.999); an in-vivo preferred orientation with a global orientation selectivity index (gOSI); and a receptive-field centre from the twin (spike-triggered-average fit, screen coordinates in [-1,1]). We use orientation, as a continuous angle, for the 5,287 neurons with gOSI ≥ 0.25, and receptive-field position for the 11,326 neurons whose twin test correlation is ≥ 0.2. The orientation labels are in-vivo measurements on both substrates; the receptive-field labels come from the twin. Same-scan pairs correlate about 50% more strongly than cross-scan pairs at matched receptive-field distance; the content of the correlations is the same across scans, and all populations mix scans. Only 2% of the receptive-field label variance lies between scans and 9% between areas, so a decoder that read scan or area identity from the Gram could reach at most that much R².

**Allen.** From Visual Coding 2P (de Vries et al. 2020) we take the 36 VISp excitatory containers with at least 150 orientation-labelled cells: 33 mice, 9,281 cells. Relations come from session A, either responses to natural movies shown to every mouse (4,500 bins) or the 40 blank-subtracted condition means of drifting gratings (8 directions × 5 temporal frequencies). Orientation labels come from static gratings in session B (six orientations 30° apart, gOSI ≥ 0.25, p < 0.05), so labels and relations use different stimuli. Receptive-field centres come from locally sparse noise in session C (1,504 cells with a significant fit, 23 to 112 per mouse).

### 2.2 Relations

Each neuron's response vector is centred and normalised; the relation between two neurons is the cosine of their vectors, a signal correlation. For a population of n neurons the Gram is the n × n matrix of these values, standardised by the global off-diagonal mean and standard deviation, with the diagonal zeroed.

### 2.3 Decoder

Each row of a population's Gram, together with three row statistics, is projected to a token; L pre-norm transformer blocks with H-head attention update the tokens; a linear head reads out, per token, the unit vector (cos 2θ, sin 2θ) of the preferred orientation θ or an (x, y) receptive-field position. The loss is the mean squared error on the labelled tokens only; unlabelled neurons stay in the population and receive no loss. The standard configuration is width 512, 8 layers and 17M parameters with populations of 512 neurons on MICrONS, and width 256, 4 layers and 2.2M parameters with populations of 256 neurons on Allen; other sizes are reported in the appendix. Training draws thousands of random populations per epoch.

### 2.4 Protocol

**Split.** MICrONS neurons are halved at random. Training populations are sampled inside the training half and test populations inside the test half, so no test neuron ever appears in a training population (Fig. 1d). A slice of the training half is held out to choose the epoch; the test half never influences selection.

**Regimes (Allen).** Two independent choices define a run (Fig. 4). The split decides where test neurons come from: the training animals, each halved into training and test neurons, or held-out animals. The population decides what a sample is: neurons drawn from one animal, so that the Gram is a within-circuit matrix, or from several animals, so that most entries are between-animal correlations. We call the four cells `within`, `pooledwithin`, `cross` and `pooledcross`. Held-out regimes select the epoch on further held-out mice. Every held-out cell is run on three mouse splits and every training-animal cell on three neuron splits.

**Metrics.** Orientation is scored by the mean absolute angular error between the decoded angle, ½ atan2 of the output, and the label, taken modulo 180°, so it lies between 0° and 90°. A decoder that carries no information produces errors spread evenly over that range, whatever it outputs, so chance is 45°. Allen labels sit on a 30° grid: a decoder that recovered every true preference exactly would still disagree with those labels by 7.5° on average, which is the floor to read Allen errors against. A frame check reports the error after the best single rotation or reflection of all decoded angles; if it is lower than the raw error, the decoder recovered the structure but not the frame. Receptive fields are scored by R² against a baseline of 0.

### 2.5 Symmetry analysis

To expose the symmetry of the relations we average the Gram into a K × K class-Gram, the mean relation between neurons of two orientation classes, project it onto its circulant part, and ask which relabellings of the classes a held-out half's class-Gram is compatible with (all K! relabellings, Frobenius distance). A relabelling that differs from the truth by a rotation or a reflection of the orientation circle is an element of the dihedral group D_K; the analysis reports whether the truth is singled out absolutely or only modulo D_K.

## 3. Results

### 3.1 Per-neuron content from the correlation matrix alone

On MICrONS the decoder reads orientation and receptive-field position from the Gram of 512 neurons of the test half (Fig. 2, Table 1). The twin substrate is cleaner than the in-vivo one, since the twin removes trial noise, and receptive fields carry more than orientation. The predictions are in the true frame: the frame check lowers the error by less than 0.1° in every run, so the decoder places each neuron at its absolute orientation rather than recovering the structure up to a rotation. About half of the neurons are decoded to within 15° (0.46 in vivo, 0.54 on the twin). Both readouts are properties of individual neurons rather than of the scan they were recorded in: centring predictions and labels per scan leaves the receptive-field R² unchanged and the orientation error within 0.2° (Appendix B), and the readout survives test Grams built from stimulus bins the decoder never trained on.

*Table 1. MICrONS, label-free per-neuron decoding on the test half; 17M decoder, 512-neuron populations, mean ± s.d. over 3 seeds.*

| content | substrate | score | chance |
|---|---|---|---|
| orientation (mean angular error) | in vivo | 25.4° ± 0.5° | 45° |
| orientation (mean angular error) | digital twin | 20.2° ± 0.4° | 45° |
| receptive field (R²) | in vivo | 0.234 ± 0.007 | 0 |
| receptive field (R²) | digital twin | 0.480 ± 0.010 | 0 |

### 3.2 Why the absolute frame is recoverable

The class-Gram of orientation is close to circulant (Fig. 3): neighbouring orientations correlate, orthogonal ones anti-correlate, and the pattern repeats around the circle. The circulant part holds 71% of the variance in vivo and 81% in the twin. A circulant class-Gram is invariant under every rotation and reflection of the orientation circle, so it cannot single out which class is 0°: after projecting onto the circulant part, a held-out half's class-Gram is compatible with all 16 elements of D_8 and with nothing else (absolute matching at chance, matching modulo D_8 at 1.0). The raw class-Gram singles out the true labelling. The difference is the residual anisotropy, a cardinal bias whose strength differs between areas, and that is what pins the frame. A label-free decoder that outputs absolute orientation must use some such departure from the symmetry, in the relations or in the uneven distribution of preferred orientations over the circle; the frame check of the previous section shows that it does recover the frame.

### 3.3 Orientation transfers between animals through mixed populations

With natural-movie relations, orientation is nearly absent from Allen correlations: same-orientation pairs correlate 0.005 more than orthogonal pairs, against 0.09 in MICrONS, and every test is at chance. With grating relations the signal is present (0.05 within a mouse, 0.044 across mice), and the decoder's four cells separate (Fig. 5, Table 2). Single-animal populations stay within 3° of chance whether or not the animal was seen in training. Mixed populations are 7° to 10° below chance in both rows, at the same level, so transfer to a new brain has no detectable cost; what matters is whether the Gram contains between-animal correlations. Those exist only because all mice saw the same 40 conditions, and they measure how similar two tuning profiles are. The decoder still never sees which condition is which, so the absolute frame must again come from the population's relational structure. In this dataset a neuron's orientation is readable from where it sits in a correlation geometry that spans animals. From its relations inside its own circuit it is barely readable at the population sizes available here, where a single animal contributes about 140 training cells and the decoder sees nearly the same population every time; whether that is a property of the circuit or of the training diversity is open.

*Table 2. Allen, mean angular error of the decoded orientation in the four regimes (2.2M decoder, 256-neuron populations), three splits per cell (of neurons within each animal for the top row, of animals for the bottom row). Chance 45°; label grid 7.5°.*

| test neurons from | single-animal populations | mixed populations |
|---|---|---|
| the training animals | 42.5° / 42.9° / 42.1° | 35.3° / 37.2° / 36.1° |
| held-out animals | 43.1° / 43.1° / 42.7° | 35.2° / 37.0° / 38.4° |

### 3.4 Receptive fields: what crosses animals is the animal's screen position

Correlation falls with receptive-field distance within a mouse (0.127 at <5° to 0.068 beyond 40°) and across mice (0.049 to 0.011). Decoding absolute screen coordinates on held-out mice gives R² that swings with the split, from 0.29 to −0.45 for single-animal populations and from 0.21 to −0.25 for mixed ones. Decomposing the saved predictions explains the swing (Fig. 6). The correlation between predicted and true mean position of each held-out mouse, pooled over 27 mouse-level predictions, is 0.36 to 0.71 (permutation p ≤ 0.03). The correlation between predicted and true position of a neuron relative to its mouse-mates is 0.0 to 0.1 over 1,259 cells. The decoder recovers where an animal's imaged patch looks, a property of the population, and not which neuron within it looks where. R² on absolute coordinates is then dominated by whether the between-mouse spread of a particular test set is reproduced at the right scale.

The within-mouse layout is present in the correlations, since same-mouse correlation falls with distance. It is not learned because each mouse contributes 23 to 112 labelled cells. MICrONS, with 11,326 labels in one animal, is where the same decoder reaches neuron-level receptive fields at R² = 0.48.

### 3.5 What moves the numbers

Capacity pays where labels are plentiful and stops where they are scarce (Fig. A1). Twin receptive fields go from 0.34 to 0.48 to 0.51 R² between 2.2M, 17M and 57M parameters. Orientation error is flat from 2.2M upward on both MICrONS substrates (20.9°, 20.2°, 19.9° on the twin at 2.2M, 17M and 57M) and on Allen (35.2°, 36.5°, 36.7°); only the 0.3M model is worse (27.5° and 27.8° on MICrONS). Every model above 2M memorises its training neurons within 10 to 20 epochs and is early-stopped on the held-out selection slice (Fig. A3). Population size saturates by 512 neurons in MICrONS and 256 on Allen (Fig. A2). Recomputing each training population's Gram from a random subset of the stimulus conditions, a label-free augmentation, changes the Allen error by less than the seed spread in either direction (35.2°, 33.5°, 37.7° and 38.0° with 100%, 75%, 50% and 30% of the conditions at 2.2M; 36.5°, 34.3° and 35.6° with 100%, 85% and 50% at 17M) and is not used. Split-to-split variance on held-out mice exceeds seed variance (Fig. A4).

## 4. Discussion

The correlation matrix of a few hundred cortical neurons, with the stimulus unknown and no neuron labelled, determines each neuron's preferred orientation to a mean error of 25° in vivo and 20° on the digital twin, against 45° for an uninformed decoder, and its receptive-field position to R² = 0.23 in vivo and 0.48 on the digital twin. The mechanism is the one the symmetry analysis exposes: the relations fix the class structure up to the symmetries of the content, and a small, consistent departure from that symmetry fixes the frame.

Across animals the picture splits by content. Orientation lives in the relations between tuning profiles, which exist between any two neurons that saw the same conditions, in the same brain or not; from the within-circuit correlations of a single animal it is barely readable at the population sizes and per-animal training diversity available here. Receptive-field position is carried at the level of an animal's patch, and the neuron-level layout needs more labels per animal than these recordings provide. Both statements are about what correlations under a shared stimulus contain, and both are measured under a protocol in which the test neurons, and in the held-out regimes the test animals, never touch training or model selection.

Three limits. MICrONS is one animal, and its training and test halves share recording sessions; the Allen held-out-animal cells are the cleaner generalisation test, and their variance across mouse splits is large. Orientation labels are noisy in both datasets (in vivo and twin labels agree on 70% of MICrONS neurons at 8 classes; static and drifting gratings on 45 to 67% of Allen cells), and the Allen labels are quantised to 30°, which caps every labelled and label-free number alike. The twin substrate is a model fitted to the same recordings, so the in-vivo results carry the claim about cortex and the twin results show what cleaner correlations would give; the receptive-field labels themselves come from the twin, so the twin receptive-field result compares a model's correlations with the same model's receptive fields, while the orientation labels are in-vivo measurements throughout.

The symmetry result is an instance of a general expectation: a representation that is equivariant to a group has a group-fixed part, identical across individuals up to relabelling, and an individual remainder (Oizumi, Lim and Kanai 2026). The matching-over-relabellings test is the discrete form of unsupervised alignment of similarity structures across individuals by optimal transport (Kawakita et al. 2024). What this paper adds is the measurement in cortex, at the level of single neurons, and the finding that the remainder, the anisotropy, is what makes the absolute readout possible.

**Code and data.** TODO: repository URL. All data are public releases.

## Appendix A. Scaling and controls

Figures A1 to A4 report the sweeps behind Section 3.5: decoder size, population size, training dynamics and split variance. Two training-side variants were tried and dropped: recomputing each training population's Gram from a random subset of the stimulus conditions, and zeroing random Gram entries. Neither changes the angular error by more than the seed spread (Section 3.5).

## Appendix B. Controls

Four controls on the MICrONS standard configuration (17M, 512-neuron populations, digital twin unless stated; Table 3).

**What the decoder reads.** With the three row statistics as the only input per neuron and no relational term, the error is 37.7°; with the statistics plus the Gram entering only as an attention bias, which makes the decoder equivariant to the order of the neurons, it is 19.6°, the same as the standard decoder (20.2°), whose input layer also projects the raw Gram row in sample order. The relations carry the readout, and the order-dependent input layer contributes nothing.

**Cross-stimulus test.** Training and selection Grams built from one random half of the stimulus bins and test Grams from the other half give 21.2° on the twin (2,500 bins per half, raw responses rather than principal components) against 20.2° with shared bins, and 34.2° in vivo (60 bins per half) against 25.4°. A control with the same 60-bin half on both sides gives 25.3°, so the in-vivo cost is not the smaller Gram but the stimulus sample: a correlation matrix over 60 movie bins partly reflects which frames it was built from, and one over 2,500 bins does not. The relational structure that the decoder reads is stable across stimuli once the stimulus sample is large.

**Scan and area.** The MICrONS neurons come from 13 scans in four areas, same-scan pairs correlate more strongly than cross-scan pairs, and the scans look at different parts of the screen, so a decoder that read scan identity from the Gram could predict each scan's mean receptive-field position and score a positive R² without knowing any neuron's own position. That is what the Allen decoder does (Section 3.4), so we checked MICrONS the same way, centring predictions and labels per scan (and per area) before scoring. Receptive fields survive: R² is 0.50 absolute and 0.50 within scan on the twin, 0.25 and 0.25 in vivo, and 0.47 and 0.22 within area. Only 2% of the label variance lies between scans and 9% between areas, so a scan-identity readout could reach at most 0.02. For orientation, a predictor that assigns every neuron its scan's or area's mean orientation scores 39°, and removing a separate rotation per scan or area changes the decoder's error by less than 0.2°. Both MICrONS results are neuron-level readouts within a scan.

**Class balance and priors.** Preferred orientations are unevenly distributed in MICrONS (19% of labelled neurons within 11° of 0°, 25% of 90°, 6 to 10% in each oblique bin), so a decoder could gain from the distribution alone. It gains little: the best constant prediction has an error of 41.3°, a predictor that assigns every neuron the circular mean of its scan 39.0° and of its area 38.7°, against 20.3° for the decoder on the twin and 24.6° in vivo (neuron split 2). Averaging the error over eight equally weighted true-orientation bins gives 24.5° on the twin and 27.9° in vivo. Removing a separate rotation per scan or per area changes the error by less than 0.2°, and the per-area errors are within 3° of each other. On Allen the labels are nearly flat (14 to 20% per orientation), the best constant is 43.9°, and the balanced error is 36.4° against 35.7° raw. What the per-bin errors do show, on both datasets, is that the readout is cardinal: neurons preferring 0° or 90° are decoded to 11° to 12° on the twin and 25° on Allen, oblique ones to 37° to 45°, and 63 to 72% of all predictions fall on the two cardinal axes (Fig. A5). The relations say how close a neuron is to a cardinal axis, which is the anisotropy of Section 3.2 seen from the decoder's side, and much less about which oblique it prefers.

*Table 3. Controls, mean angular error on held-out neurons (chance 45°).*

| control | substrate | error |
|---|---|---|
| standard decoder | twin | 20.2° |
| row statistics only, no relational term | twin | 37.7° |
| row statistics + Gram as attention bias (equivariant) | twin | 19.6° |
| disjoint stimulus bins for training and test Grams | twin | 21.2° |
| standard decoder | in vivo | 25.4° |
| disjoint stimulus bins (60 per side) | in vivo | 34.2° |
| same 60 bins on both sides | in vivo | 25.3° |
| neuron split 1 / 2 | in vivo | 25.4° / 24.6° |

## Appendix C. Classification results

Earlier runs decoded orientation as a class (8 classes on MICrONS, 6 on Allen) with the same decoder and protocol; the per-neuron accuracies are in Table 4. They lead to the same conclusions, but a class score counts a near miss as a miss, so the angular error in the main text is the better summary of a continuous label.

*Table 4. Orientation as classification, 17M decoder. MICrONS: 8 classes, majority class 0.255, mean ± s.d. over 3 seeds. Allen: 6 classes, majority class 0.19 to 0.20, three splits per cell (17M decoder with the condition-subset augmentation, 256-neuron populations; the `pooledcross` split-0 value is the mean of 3 seeds).*

| dataset | cell | accuracy |
|---|---|---|
| MICrONS | in vivo | 0.380 ± 0.006 |
| MICrONS | digital twin | 0.420 ± 0.003 |
| Allen | training animals, single-animal populations | 0.210 / 0.216 / 0.217 |
| Allen | training animals, mixed populations | 0.281 / 0.283 / 0.283 |
| Allen | held-out animals, single-animal populations | 0.199 / 0.197 / 0.187 |
| Allen | held-out animals, mixed populations | 0.293 / 0.236 / 0.264 |

## Appendix D. Labelled reference models

TODO: ridge readouts from each neuron's correlations to labelled training neurons (MICrONS in vivo 0.41 / 0.32, twin 0.49 / 0.67; Allen leave-one-mouse-out 0.35 for orientation, 0.20 for receptive fields), with the caveat that ridge is one linear labelled model and the decoder exceeded it on Allen receptive fields.

## Appendix E. Receptive-field decomposition per split

TODO: per-split table of absolute R², mouse-level correlation and within-mouse correlation for both regimes.

## Appendix figures

- **A1. Scaling with decoder size.** Test metric versus number of decoder parameters at fixed population size (512 neurons MICrONS, 256 Allen); on MICrONS receptive fields the gain continues to 57M, while the orientation error is flat from 2.2M upward on both substrates and on Allen.
- **A2. Population size.** Test metric versus the number of neurons in each sampled population; beyond a few hundred neurons the additional companions add redundant context.
- **A3. Training dynamics and epoch selection.** Test metric, selection-set metric and training loss per epoch, showing that the selection set, never the test set, picks the epoch, and that the training loss collapses within 10 to 20 epochs on every large run.
- **A4. Allen split variance and the class-level test.** Left, `pooledcross` orientation error on the same three held-out-mouse splits for the previous standard configuration (17M, augmented) and the final one (2.2M, plain); right, the class-level test with relations from natural movies versus drifting gratings, where only the grating relations move every test off chance.
- **A5. The readout is cardinal.** Mean angular error of the neurons in each true-orientation bin, with the fraction of labels and of predictions per bin, for the twin and in-vivo MICrONS decoders and the Allen `pooledcross` decoder; cardinal orientations are decoded well, obliques near chance, and most predictions fall on 0° or 90°.
