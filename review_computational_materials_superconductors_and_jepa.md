# Review: Computational Materials for Superconductors, and JEPA

*Prepared 2026-07-01*

## Part 1 — Computational materials discovery for superconductors

### Where the field stands

Superconductor discovery has historically been bottlenecked by first-principles calculation cost: predicting a critical temperature (Tc) from first principles requires DFT-based electron-phonon coupling calculations (Eliashberg theory, the McMillan-Allen-Dynes formula), which are expensive enough that only a few thousand superconductors have been characterized computationally. Since 2023, this bottleneck has been attacked from three directions simultaneously — faster surrogate models for Tc, high-throughput DFT screening pipelines, and generative models that propose new candidate structures directly.

### Surrogate models for Tc and electron-phonon coupling

Several groups have trained models to reproduce or approximate DFT-level Eliashberg/Allen-Dynes results at a fraction of the cost:

- **BEE-NET** (Bootstrapped Ensemble of Equivariant Graph Neural Networks) predicts the Eliashberg spectral function and Tc with 0.87 K mean absolute error versus DFT, and a 99.4% true-negative rate for filtering out non-superconductors — used as a prescreening stage ahead of expensive DFT confirmation ([npj Computational Materials](https://www.nature.com/articles/s41524-026-01964-8)).
- **ALIGNN**-based models predict Debye temperature, density of states, Tc, and electron-phonon coupling directly from crystal structure ([JARVIS infrastructure](https://arxiv.org/pdf/2503.04133)).
- Deep-learning approaches trained on band structures / electronic bands alone, bypassing explicit phonon calculations ([arXiv:2409.07721](https://arxiv.org/pdf/2409.07721)).
- **S2SNet**, a pretrained network specifically for superconductivity discovery ([arXiv:2306.16270](https://arxiv.org/pdf/2306.16270)).
- A tempered deep-learning approach for the electron-phonon spectral function directly ([arXiv:2401.16611](https://arxiv.org/pdf/2401.16611)).
- Density-of-states rescaling methods for high-throughput Tc prediction ([arXiv:2508.18371](https://arxiv.org/pdf/2508.18371)).

A new benchmark, **HTSC-2025**, compiles ambient-pressure high-Tc superconductors reported in 2024–2025 (X₂YH₆ systems, MXH₃ perovskites, M₃XH₈ fluorites, LaH₁₀-derived cage structures, MgB₂-derived 2D honeycombs) specifically to standardize AI-driven Tc prediction evaluation ([arXiv:2506.03837](https://arxiv.org/pdf/2506.03837)).

### High-throughput screening pipelines

Full pipelines now chain ML pre-screening with DFT confirmation at scale:

- An AI-accelerated workflow reduced 1.3 million candidate structures down to 741 dynamically and thermodynamically stable compounds with DFT-confirmed Tc > 5 K ([npj Computational Materials](https://www.nature.com/articles/s41524-026-01964-8)).
- Screening of ~1,000 2D materials via the McMillan-Allen-Dynes formula identified 34 dynamically stable structures with Tc > 5 K, including a previously unreported Mg₂B₄N₂ at 21.8 K.
- Boride-specific and 2D-specific high-throughput screens ([arXiv:2401.13211](https://arxiv.org/pdf/2401.13211), [Nano Letters](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.2c04420)).
- The **SuperC consortium** used ML prescreening to identify two new superconductors (YRu₃B₂, LuRu₃B₂) and has stated a goal of a practical room-temperature superconductor by 2033 ([phys.org](https://phys.org/news/2026-06-superconductors-yield-thousands.html)).

### Hydride superconductors (high-pressure)

Hydrides remain the highest-Tc class (LaH₁₀ and relatives), and are a major focus of ML-driven search because their chemical space is vast and DFT-only search is intractable:

- A "Large Atomic Model" — a deep-learning interatomic potential trained on first-principles data from 200,000+ hydride structures — was used to explore ~36 million ternary hydride structures across 29 elements, identifying 144 candidates with predicted Tc > 200 K at 200 GPa ([National Science Review](https://academic.oup.com/nsr/article/13/6/nwag030/8427328), [arXiv:2502.16558](https://arxiv.org/pdf/2502.16558)).
- Machine-learned interatomic potentials (MLIPs) combined with the stochastic self-consistent harmonic approximation (SSCHA) are being used to capture anharmonic and quantum nuclear effects in light-atom hydrides, where standard harmonic DFT breaks down ([Quantum Zeitgeist](https://quantumzeitgeist.com/150-prediction-advances-crystal-structure-unlock-superconducting/), [npj Computational Materials](https://www.nature.com/articles/s41524-025-01553-1)).
- A broader review of MLIPs situates this within the "centennial" of quantum mechanics, framing MLIPs as the bridge between quantum accuracy and classical-scale simulation ([Nature Computational Science](https://www.nature.com/articles/s43588-025-00930-6)).
- AI-accelerated screening for metallized σ-bonding compounds (a structural motif associated with conventional high-Tc superconductivity) — [arXiv:2606.21251](https://arxiv.org/pdf/2606.21251).

### Generative / inverse design approaches

Rather than screening a fixed candidate list, generative models now propose novel crystal structures conditioned on target properties:

- Diffusion models for crystal structure generation, including point-cloud representations, and valence/symmetry-constrained variants such as **CrysVCD** (85% thermodynamic, 68% phonon stability in generated structures) and **DiffCSP++** (space-group-conditioned) ([review, arXiv:2509.02723](https://arxiv.org/pdf/2509.02723); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11763582/)).
- LLM-guided evolutionary and hybrid LLM+diffusion approaches (**MatLLMSearch**, **LLEMA**, LLM+diffusion hybrids) treat structure generation as a search or sequence-generation problem ([arXiv:2502.20933](https://arxiv.org/pdf/2502.20933), [arXiv:2510.23040](https://arxiv.org/html/2510.23040v1)).
- **AtomGPT** applies NLP-style generation specifically to superconductor design.
- A broader survey of generative AI for crystal structures is available at [arXiv:2509.02723](https://arxiv.org/pdf/2509.02723), and a curated resource list at the [AI-for-Crystal-Materials GitHub](https://github.com/WanyuGroup/AI-for-crystal-materials/).

### Takeaways relevant to a heuristic scoring pipeline

For a project like yours that runs an offline heuristic scoring pipeline (chemistry → tc_model → graph → surprise → motif → ranking, per project memory), the field has largely moved toward: (1) learned surrogates that approximate Eliashberg/Tc at near-zero cost as a pre-filter, (2) DFT only as a final confirmation stage on a shortlist, and (3) generative/inverse-design models as a complementary route for proposing candidates rather than only screening a fixed list. None of the current published pipelines appear to fold in a "surprise" or novelty-scoring stage the way yours does — that's a plausible point of differentiation worth positioning explicitly if this is heading toward a paper.

---

## Part 2 — JEPA (Joint Embedding Predictive Architecture)

### Core idea

JEPA is Yann LeCun's proposed alternative to generative (pixel/token-reconstruction) self-supervised learning. Instead of predicting raw future observations, a JEPA predicts the *latent representation* of a target (masked patch, future frame, etc.) from the latent representation of context, using a predictor network between two encoders. The claimed advantage is that the model isn't forced to model irrelevant, unpredictable pixel-level detail — it only has to be right about what's predictable in representation space ([Turing Post overview](https://www.turingpost.com/p/jepa), [in-depth guide](https://nextwaves.com/blog/joint-embedding-predictive-architecture-jepa-a-complete-in-depth-guide)).

### Timeline

- **2022** — JEPA proposed conceptually as part of LeCun's "path towards autonomous machine intelligence" position paper.
- **2023** — **I-JEPA**, the image version, demonstrated the approach for static images.
- **2024** — **V-JEPA**, extended to video, learning temporal dynamics in latent space.
- **2025** — **V-JEPA 2**, a substantially larger video world model combining roughly a million hours of internet video with a smaller amount of robot trajectory data; the first video-trained world model to report state-of-the-art results on understanding, prediction, *and* zero-shot planning for robot control in new environments ([Meta AI](https://ai.meta.com/research/vjepa/)).
- **2026** — **V-JEPA 2.1** (March 2026) refines the recipe for more temporally consistent dense features. The approach also anchored LeCun's new venture, **AMI Labs**, which raised what's reported as Europe's largest seed round to pursue world models built on this architecture ([AI2Work](https://ai2.work/blog/yann-lecun-s-ami-labs-lands-europe-s-largest-seed-for-world-models)).

### Variants and extensions (2025–2026)

The architecture has been adapted well beyond vision:

- **VL-JEPA** — vision-language joint embedding prediction ([arXiv:2512.10942](https://arxiv.org/pdf/2512.10942)).
- **Var-JEPA** — a variational formulation bridging predictive (JEPA-style) and generative self-supervised learning ([arXiv:2603.20111](https://arxiv.org/pdf/2603.20111)).
- **ACT-JEPA** — for action/behavior representation learning ([arXiv:2501.14622](https://arxiv.org/abs/2501.14622)).
- **Demo-JEPA** — one-shot cross-embodiment imitation learning for robotics ([arXiv:2605.20811](https://arxiv.org/html/2605.20811v1)).
- **Phys-JEPA** — physics-informed latent world models for multivariate time-series forecasting, relevant if you're thinking about JEPA-style approaches outside vision ([arXiv:2606.16076](https://arxiv.org/pdf/2606.16076)).
- **GeoJEPA** — multimodal geospatial learning, addressing augmentation/sampling bias ([arXiv:2503.05774](https://arxiv.org/pdf/2503.05774)).
- Value-guided action planning built on JEPA world models ([arXiv:2601.00844](https://arxiv.org/pdf/2601.00844)).

### Theory and formal grounding

A 2026 paper offers **"A Generalization Theory for JEPA-Based World Models"**, formally characterizing the conditions under which a JEPA can learn a faithful model of the environment, and — importantly — how far current implementations fall short of that standard ([arXiv:2606.27014](https://arxiv.org/pdf/2606.27014)).

### Known limitations and critiques

- **Centralization assumption**: hierarchical JEPA (H-JEPA) implicitly assumes a single system can acquire everything needed to simulate its environment; this centralization is flagged as a structural limitation for multi-agent settings, since the resulting model only approximates the true environment state rather than accessing it directly.
- **Brittleness under distribution shift**: a 2026 benchmark paper found current JEPA-based world models collapse under minor visual perturbations ([Tech Times](https://www.techtimes.com/articles/317452/20260531/yann-lecuns-world-model-earns-formal-proof-benchmark-finds-current-models-brittle.htm)).
- **Representation collapse risk**: a known failure mode of joint-embedding predictive training generally (predictor finds a trivial/degenerate solution), still an active concern in planning applications.
- **External critique**: Eric Xing (MBZUAI) has argued that "abstract latent prediction without a generative validator is essentially meditation in a closed room" — i.e., without any mechanism to check latent predictions against ground truth, JEPA-style models risk losing contact with reality, a critique specifically aimed at the lack of a generative/verification signal.
- Despite this, LeCun continues to bet on the architecture, viewing it as the more scalable path relative to autoregressive generative pretraining.

### Relevance to a project combining JEPA framing with materials scoring

If SUPRA-JEPA is using "JEPA" more as an architectural inspiration (latent-space prediction of "what's next" in a chemistry/graph representation) rather than a literal video/vision JEPA, the generalization-theory paper ([arXiv:2606.27014](https://arxiv.org/pdf/2606.27014)) is worth reading closely — it's the most direct source for what formally distinguishes a "real" JEPA world model from a heuristic latent scorer, which is a distinction that matters if you want to claim the JEPA label defensibly.

---

## Sources

**Superconductors / computational materials:**
- [Developing a complete AI-accelerated workflow for superconductor discovery — npj Computational Materials](https://www.nature.com/articles/s41524-026-01964-8)
- [A deep learning approach to search for superconductors from electronic bands — arXiv:2409.07721](https://arxiv.org/pdf/2409.07721)
- [S2SNet: A Pretrained Neural Network for Superconductivity Discovery — arXiv:2306.16270](https://arxiv.org/pdf/2306.16270)
- [Towards the discovery of high critical magnetic field superconductors — arXiv:2601.21044](https://arxiv.org/pdf/2601.21044)
- [High-Tc superconductor candidates proposed by machine learning — arXiv:2406.14524](https://arxiv.org/pdf/2406.14524)
- [Data-Driven Superconductivity: a Review of ML Methods — Journal of Superconductivity and Novel Magnetism](https://link.springer.com/article/10.1007/s10948-026-07175-y)
- [New superconductors identified via ML prescreening — phys.org](https://phys.org/news/2026-06-superconductors-yield-thousands.html)
- [Accelerating superconductor discovery via tempered deep learning — arXiv:2401.16611](https://arxiv.org/pdf/2401.16611)
- [HTSC-2025 benchmark dataset — arXiv:2506.03837](https://arxiv.org/pdf/2506.03837)
- [High-Throughput DFT-Based Discovery of 2D Superconductors — Nano Letters](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.2c04420)
- [JARVIS Infrastructure for Materials Design — arXiv:2503.04133](https://arxiv.org/pdf/2503.04133)
- [Designing High-Tc Superconductors with BCS-inspired Screening — npj Computational Materials](https://www.nature.com/articles/s41524-022-00933-1)
- [High-throughput screening for boride superconductors — arXiv:2401.13211](https://arxiv.org/pdf/2401.13211)
- [Machine learning interatomic potentials at the centennial crossroads — Nature Computational Science](https://www.nature.com/articles/s43588-025-00930-6)
- [Computational discovery of high-Tc superconducting ternary hydrides via deep learning — National Science Review](https://academic.oup.com/nsr/article/13/6/nwag030/8427328) / [arXiv:2502.16558](https://arxiv.org/pdf/2502.16558)
- [Advances in Crystal Structure Prediction — Quantum Zeitgeist](https://quantumzeitgeist.com/150-prediction-advances-crystal-structure-unlock-superconducting/)
- [Efficient modelling of anharmonicity in PdCuH2 — npj Computational Materials](https://www.nature.com/articles/s41524-025-01553-1)
- [High-throughput Tc predictions via density of states rescaling — arXiv:2508.18371](https://arxiv.org/pdf/2508.18371)
- [AI-accelerated metallized σ-bonding screening — arXiv:2606.21251](https://arxiv.org/pdf/2606.21251)
- [Generative AI for Crystal Structures: A Review — arXiv:2509.02723](https://arxiv.org/pdf/2509.02723)
- [Generative design of crystal structures via point cloud diffusion — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11763582/)
- [MatLLMSearch — arXiv:2502.20933](https://arxiv.org/pdf/2502.20933)
- [LLM Meets Diffusion for Crystal Material Generation — arXiv:2510.23040](https://arxiv.org/html/2510.23040v1)
- [AI-for-Crystal-Materials resource list — GitHub](https://github.com/WanyuGroup/AI-for-crystal-materials/)

**JEPA:**
- [What is Joint Embedding Predictive Architecture (JEPA)? — Turing Post](https://www.turingpost.com/p/jepa)
- [JEPA: A Complete In-Depth Guide — Nextwaves](https://nextwaves.com/blog/joint-embedding-predictive-architecture-jepa-a-complete-in-depth-guide)
- [Introducing V-JEPA 2 — Meta AI](https://ai.meta.com/research/vjepa/)
- [Demo-JEPA — arXiv:2605.20811](https://arxiv.org/html/2605.20811v1)
- [A Generalization Theory for JEPA-Based World Models — arXiv:2606.27014](https://arxiv.org/pdf/2606.27014)
- [VL-JEPA — arXiv:2512.10942](https://arxiv.org/pdf/2512.10942)
- [Var-JEPA — arXiv:2603.20111](https://arxiv.org/pdf/2603.20111)
- [ACT-JEPA — arXiv:2501.14622](https://arxiv.org/abs/2501.14622)
- [Phys-JEPA — arXiv:2606.16076](https://arxiv.org/pdf/2606.16076)
- [GeoJEPA — arXiv:2503.05774](https://arxiv.org/pdf/2503.05774)
- [Value-guided action planning with JEPA world models — arXiv:2601.00844](https://arxiv.org/pdf/2601.00844)
- [Yann LeCun's AMI Labs seed round — AI2Work](https://ai2.work/blog/yann-lecun-s-ami-labs-lands-europe-s-largest-seed-for-world-models)
- [JEPA benchmark brittleness finding — Tech Times](https://www.techtimes.com/articles/317452/20260531/yann-lecuns-world-model-earns-formal-proof-benchmark-finds-current-models-brittle.htm)
