# Revue : matériaux computationnels pour supraconducteurs, et JEPA

*Préparé le 2026-07-01*

## Partie 1 — Découverte computationnelle de matériaux pour supraconducteurs

### Où en est le domaine

La découverte de supraconducteurs a longtemps été freinée par le coût des calculs ab initio : prédire une température critique (Tc) à partir des premiers principes nécessite des calculs DFT de couplage électron-phonon (théorie d'Eliashberg, formule de McMillan-Allen-Dynes), suffisamment coûteux pour que seuls quelques milliers de supraconducteurs aient été caractérisés numériquement. Depuis 2023, ce goulot d'étranglement est attaqué sur trois fronts en parallèle : des modèles de substitution (surrogate) plus rapides pour la Tc, des pipelines de criblage DFT à haut débit, et des modèles génératifs qui proposent directement de nouvelles structures candidates.

### Modèles de substitution pour la Tc et le couplage électron-phonon

Plusieurs équipes ont entraîné des modèles pour reproduire ou approximer les résultats DFT de type Eliashberg/Allen-Dynes à une fraction du coût :

- **BEE-NET** (Bootstrapped Ensemble of Equivariant Graph Neural Networks) prédit la fonction spectrale d'Eliashberg et la Tc avec une erreur absolue moyenne de 0,87 K par rapport à la DFT, et un taux de vrais négatifs de 99,4 % pour filtrer les non-supraconducteurs — utilisé comme étape de préfiltrage avant confirmation DFT coûteuse ([npj Computational Materials](https://www.nature.com/articles/s41524-026-01964-8)).
- Les modèles basés sur **ALIGNN** prédisent la température de Debye, la densité d'états, la Tc et le couplage électron-phonon directement à partir de la structure cristalline ([infrastructure JARVIS](https://arxiv.org/pdf/2503.04133)).
- Des approches de deep learning entraînées uniquement sur les structures de bandes électroniques, sans calcul explicite des phonons ([arXiv:2409.07721](https://arxiv.org/pdf/2409.07721)).
- **S2SNet**, un réseau pré-entraîné spécifiquement pour la découverte de supraconductivité ([arXiv:2306.16270](https://arxiv.org/pdf/2306.16270)).
- Une approche de deep learning « tempéré » pour la fonction spectrale électron-phonon directement ([arXiv:2401.16611](https://arxiv.org/pdf/2401.16611)).
- Des méthodes de recalibrage de la densité d'états pour la prédiction de Tc à haut débit ([arXiv:2508.18371](https://arxiv.org/pdf/2508.18371)).

Un nouveau benchmark, **HTSC-2025**, rassemble les supraconducteurs à haute Tc à pression ambiante rapportés en 2024–2025 (systèmes X₂YH₆, pérovskites MXH₃, fluorites M₃XH₈, structures en cage dérivées de LaH₁₀, nids d'abeilles 2D dérivés de MgB₂) spécifiquement pour standardiser l'évaluation des prédictions de Tc pilotées par IA ([arXiv:2506.03837](https://arxiv.org/pdf/2506.03837)).

### Pipelines de criblage à haut débit

Des pipelines complets enchaînent désormais préfiltrage ML et confirmation DFT à grande échelle :

- Un flux de travail accéléré par IA a réduit 1,3 million de structures candidates à 741 composés stables dynamiquement et thermodynamiquement, avec Tc confirmée par DFT > 5 K ([npj Computational Materials](https://www.nature.com/articles/s41524-026-01964-8)).
- Le criblage d'environ 1 000 matériaux 2D via la formule de McMillan-Allen-Dynes a identifié 34 structures dynamiquement stables avec Tc > 5 K, dont un composé Mg₂B₄N₂ jusqu'alors non répertorié, à 21,8 K.
- Criblages à haut débit spécifiques aux borures et aux matériaux 2D ([arXiv:2401.13211](https://arxiv.org/pdf/2401.13211), [Nano Letters](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.2c04420)).
- Le **consortium SuperC** a utilisé un préfiltrage ML pour identifier deux nouveaux supraconducteurs (YRu₃B₂, LuRu₃B₂) et affiche un objectif de supraconducteur pratique à température ambiante d'ici 2033 ([phys.org](https://phys.org/news/2026-06-superconductors-yield-thousands.html)).

### Supraconducteurs hydrures (haute pression)

Les hydrures restent la classe à plus haute Tc (LaH₁₀ et apparentés), et constituent un axe majeur de recherche pilotée par ML car leur espace chimique est immense et une recherche DFT seule est impraticable :

- Un « Large Atomic Model » — un potentiel interatomique de deep learning entraîné sur des données ab initio issues de plus de 200 000 structures d'hydrures — a permis d'explorer environ 36 millions de structures d'hydrures ternaires sur 29 éléments, identifiant 144 candidats avec Tc prédite > 200 K à 200 GPa ([National Science Review](https://academic.oup.com/nsr/article/13/6/nwag030/8427328), [arXiv:2502.16558](https://arxiv.org/pdf/2502.16558)).
- Les potentiels interatomiques appris par ML (MLIP), combinés à l'approximation harmonique auto-cohérente stochastique (SSCHA), sont utilisés pour capturer les effets d'anharmonicité et les effets quantiques nucléaires dans les hydrures à atomes légers, où la DFT harmonique standard échoue ([Quantum Zeitgeist](https://quantumzeitgeist.com/150-prediction-advances-crystal-structure-unlock-superconducting/), [npj Computational Materials](https://www.nature.com/articles/s41524-025-01553-1)).
- Une revue plus large des MLIP situe ce travail dans le cadre du « centenaire » de la mécanique quantique, présentant les MLIP comme le pont entre précision quantique et simulation à l'échelle classique ([Nature Computational Science](https://www.nature.com/articles/s43588-025-00930-6)).
- Criblage accéléré par IA pour les composés à liaison σ métallisée (motif structurel associé à la supraconductivité conventionnelle à haute Tc) — [arXiv:2606.21251](https://arxiv.org/pdf/2606.21251).

### Approches génératives / conception inverse

Plutôt que de cribler une liste fixe de candidats, des modèles génératifs proposent désormais de nouvelles structures cristallines conditionnées sur des propriétés cibles :

- Modèles de diffusion pour la génération de structures cristallines, y compris les représentations en nuage de points, et variantes contraintes par valence/symétrie comme **CrysVCD** (85 % de stabilité thermodynamique, 68 % de stabilité phononique dans les structures générées) et **DiffCSP++** (conditionné par groupe d'espace) ([revue, arXiv:2509.02723](https://arxiv.org/pdf/2509.02723) ; [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11763582/)).
- Approches évolutionnaires guidées par LLM et hybrides LLM+diffusion (**MatLLMSearch**, **LLEMA**, hybrides LLM+diffusion) traitant la génération de structures comme un problème de recherche ou de génération de séquence ([arXiv:2502.20933](https://arxiv.org/pdf/2502.20933), [arXiv:2510.23040](https://arxiv.org/html/2510.23040v1)).
- **AtomGPT** applique une génération de type NLP spécifiquement à la conception de supraconducteurs.
- Une revue plus large de l'IA générative pour les structures cristallines est disponible sur [arXiv:2509.02723](https://arxiv.org/pdf/2509.02723), et une liste de ressources sélectionnées sur le [GitHub AI-for-Crystal-Materials](https://github.com/WanyuGroup/AI-for-crystal-materials/).

### Enseignements pertinents pour un pipeline de scoring heuristique

Pour un projet comme le vôtre, qui fait tourner un pipeline de scoring heuristique offline (chimie → tc_model → graphe → surprise → motif → classement, d'après la mémoire du projet), le domaine s'est largement orienté vers : (1) des modèles de substitution appris qui approximent Eliashberg/Tc à coût quasi nul en tant que préfiltre, (2) la DFT réservée à une étape finale de confirmation sur une short-list, et (3) les modèles génératifs/de conception inverse comme voie complémentaire pour proposer des candidats plutôt que de simplement cribler une liste fixe. Aucun des pipelines publiés actuellement ne semble intégrer une étape de scoring de « surprise » ou de nouveauté comme le fait le vôtre — c'est un point de différenciation plausible, à mettre en avant explicitement si cela doit déboucher sur une publication.

---

## Partie 2 — JEPA (Joint Embedding Predictive Architecture)

### Idée centrale

JEPA est l'alternative proposée par Yann LeCun à l'apprentissage auto-supervisé génératif (reconstruction de pixels/tokens). Au lieu de prédire les observations brutes futures, un JEPA prédit la *représentation latente* d'une cible (patch masqué, image future, etc.) à partir de la représentation latente du contexte, via un réseau prédicteur placé entre deux encodeurs. L'avantage revendiqué est que le modèle n'est pas contraint de modéliser des détails imprévisibles et non pertinents au niveau du pixel — il doit seulement être juste sur ce qui est prévisible dans l'espace de représentation ([aperçu Turing Post](https://www.turingpost.com/p/jepa), [guide approfondi](https://nextwaves.com/blog/joint-embedding-predictive-architecture-jepa-a-complete-in-depth-guide)).

### Chronologie

- **2022** — JEPA proposé conceptuellement dans le position paper de LeCun sur le « chemin vers l'intelligence machine autonome ».
- **2023** — **I-JEPA**, la version image, démontre l'approche sur des images statiques.
- **2024** — **V-JEPA**, étendu à la vidéo, apprenant la dynamique temporelle dans l'espace latent.
- **2025** — **V-JEPA 2**, un world model vidéo bien plus grand combinant environ un million d'heures de vidéo issue d'internet avec une petite quantité de trajectoires robotiques ; le premier world model entraîné sur vidéo à rapporter des résultats état de l'art en compréhension, prédiction *et* planification zero-shot pour le contrôle de robots dans de nouveaux environnements ([Meta AI](https://ai.meta.com/research/vjepa/)).
- **2026** — **V-JEPA 2.1** (mars 2026) affine la recette pour des caractéristiques denses plus cohérentes temporellement. L'approche a aussi ancré la nouvelle entreprise de LeCun, **AMI Labs**, qui a levé ce qui est présenté comme le plus gros tour de seed d'Europe pour poursuivre des world models basés sur cette architecture ([AI2Work](https://ai2.work/blog/yann-lecun-s-ami-labs-lands-europe-s-largest-seed-for-world-models)).

### Variantes et extensions (2025–2026)

L'architecture a été adaptée bien au-delà de la vision :

- **VL-JEPA** — prédiction conjointe vision-langage ([arXiv:2512.10942](https://arxiv.org/pdf/2512.10942)).
- **Var-JEPA** — une formulation variationnelle faisant le pont entre apprentissage prédictif (façon JEPA) et génératif ([arXiv:2603.20111](https://arxiv.org/pdf/2603.20111)).
- **ACT-JEPA** — pour l'apprentissage de représentations d'action/comportement ([arXiv:2501.14622](https://arxiv.org/abs/2501.14622)).
- **Demo-JEPA** — imitation cross-embodiment en un coup (one-shot) pour la robotique ([arXiv:2605.20811](https://arxiv.org/html/2605.20811v1)).
- **Phys-JEPA** — world models latents informés par la physique pour la prévision de séries temporelles multivariées, pertinent si vous envisagez des approches façon JEPA hors vision ([arXiv:2606.16076](https://arxiv.org/pdf/2606.16076)).
- **GeoJEPA** — apprentissage géospatial multimodal, traitant les biais d'augmentation/échantillonnage ([arXiv:2503.05774](https://arxiv.org/pdf/2503.05774)).
- Planification d'action guidée par la valeur, construite sur des world models JEPA ([arXiv:2601.00844](https://arxiv.org/pdf/2601.00844)).

### Fondements théoriques

Un article de 2026 propose **« A Generalization Theory for JEPA-Based World Models »**, caractérisant formellement les conditions sous lesquelles un JEPA peut apprendre un modèle fidèle de l'environnement et — point important — dans quelle mesure les implémentations actuelles restent en deçà de ce standard ([arXiv:2606.27014](https://arxiv.org/pdf/2606.27014)).

### Limites et critiques connues

- **Hypothèse de centralisation** : le JEPA hiérarchique (H-JEPA) suppose implicitement qu'un système unique peut acquérir tout ce dont il a besoin pour simuler son environnement ; cette centralisation est identifiée comme une limite structurelle pour les contextes multi-agents, le modèle résultant ne faisant qu'approximer l'état réel de l'environnement plutôt que d'y accéder directement.
- **Fragilité face aux changements de distribution** : un article de benchmark de 2026 montre que les world models JEPA actuels s'effondrent sous de légères perturbations visuelles ([Tech Times](https://www.techtimes.com/articles/317452/20260531/yann-lecuns-world-model-earns-formal-proof-benchmark-finds-current-models-brittle.htm)).
- **Risque d'effondrement de représentation** : un mode de défaillance connu de l'entraînement prédictif par embeddings conjoints en général (le prédicteur trouve une solution triviale/dégénérée), toujours une préoccupation active pour les applications de planification.
- **Critique externe** : Eric Xing (MBZUAI) a avancé que « la prédiction latente abstraite sans validateur génératif est essentiellement de la méditation dans une pièce fermée » — c'est-à-dire que sans mécanisme pour vérifier les prédictions latentes par rapport à la réalité terrain, les modèles de type JEPA risquent de perdre le contact avec la réalité, une critique visant spécifiquement l'absence de signal génératif/de vérification.
- Malgré cela, LeCun continue de parier sur cette architecture, la considérant comme la voie la plus scalable comparée au pré-entraînement génératif autorégressif.

### Pertinence pour un projet combinant cadrage JEPA et scoring de matériaux

Si SUPRA-JEPA utilise « JEPA » davantage comme une inspiration architecturale (prédiction dans l'espace latent de « ce qui vient ensuite » dans une représentation chimie/graphe) plutôt qu'un JEPA vidéo/vision au sens littéral, l'article sur la théorie de généralisation ([arXiv:2606.27014](https://arxiv.org/pdf/2606.27014)) mérite une lecture attentive — c'est la source la plus directe pour ce qui distingue formellement un « vrai » world model JEPA d'un scoreur latent heuristique, une distinction qui compte si vous voulez revendiquer l'étiquette JEPA de façon défendable.

---

## Sources

**Supraconducteurs / matériaux computationnels :**
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

**JEPA :**
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
