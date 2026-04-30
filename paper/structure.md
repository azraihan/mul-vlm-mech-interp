# Paper Structure
# "Do VLMs See in English? Mechanistic Analysis of the Language Pivot in Multilingual Vision-Language Models"
# Target: MeLLM @ ACL 2026 | Deadline: May 1, 2026 | Format: 6–8 pages ACL

## Title
*"Do VLMs See in English? Mechanistic Analysis of the Language Pivot in Multilingual Vision-Language Models"*

## Abstract (~150 words)
Fill in X (accuracy gain) and Y–Z (pivot layer range) from actual results.

> Multilingual vision-language models (VLMs) consistently underperform on non-English visual queries, yet the internal mechanism behind this disparity is unknown. We apply logit lens analysis and activation patching to LLaVA-1.5-7B and show that non-English visual queries are processed through an English-biased concept space in middle layers (layers Y–Z), extending the English-pivot phenomenon of Wendler et al. (2024) to the multimodal setting for the first time. We further show that the visual modality partially suppresses but does not eliminate this pivot. Building on our mechanistic findings, we propose a training-free steering intervention that adds a language-direction vector to the residual stream at the identified pivot layers, improving average non-English VQA accuracy by X points on MMMB and xGQA without any fine-tuning. Our findings suggest the English pivot is a structural property inherited from the LLM backbone, not mitigated by multimodal training.

## §1 Introduction (0.5 pages)
- Para 1: Deploy hook — multilingual VLMs used globally; non-English performance gap is documented (cite PLAST Fan et al. 2025, "What Is Missing in Multilingual Visual Reasoning" Song et al. NAACL 2025).
- Para 2: The English-pivot in text-only LLMs — cite Wendler et al. ACL 2024 (three phases), Ferrando & Costa-jussà EMNLP 2024 (circuit similarity across languages). Neither studies VLMs.
- Para 3: Gap statement — nobody knows whether or how the visual modality interacts with this pivot.
- Para 4: Contributions (3 bullet points):
  - First causal mechanistic analysis of the English pivot in a multilingual VLM
  - Finding: visual modality partially but not fully suppresses the pivot (cite Figure 3)
  - Training-free steering intervention improving non-English VQA by X% on MMMB/xGQA (cite Table 1)

## §2 Related Work (0.75 pages)
**2.1 Mechanistic Interpretability of Multilingual LLMs**
Cite: Wendler et al. 2024; Dumas et al. 2024; Ferrando & Costa-jussà 2024; Resck et al. EMNLP 2025; Brinkmann et al. NAACL 2025.
Key contrast: all prior mech-interp multilingual work is text-only.

**2.2 Multilingual Vision-Language Models**
Cite: PLAST (Fan et al. EMNLP 2025); CulturalPangea (Nyandwi et al. EMNLP 2025); Song et al. NAACL 2025.
Key contrast: our work is causal (activation patching) and training-free.

**2.3 Mechanistic Interpretability for VLMs**
Cite: Golovanevsky et al. NAACL 2025; Zou et al. 2023 (representation engineering).

## §3 Background (0.4 pages)
- LLaVA-1.5 architecture (3 sentences)
- Logit lens
- Activation patching / AIE
- Wendler et al. three phases

## §4 Experiments and Results (3.5 pages)
**4.1 Experimental Setup** — Model, probing set, languages, benchmarks, baselines
**4.2 The English Pivot in VLMs** → Figure 1
**4.3 Causal Localization** → Figures 2 and 3
**4.4 Training-Free Steering** → Table 1
**4.5 Ablations** → Tables 2, 3, 4 and Figure 4

## §5 Discussion (0.5 pages)
- Why does the pivot exist?
- Connection to multilingual curse
- Limitations and future work

## §6 Conclusion (0.2 pages)

## References
Required citations:
- wendler-etal-2024-llamas (ACL 2024)
- ferrando-costa-jussa-2024-similarity (EMNLP Findings 2024)
- dumas-etal-2024-llamas-process (arXiv)
- fan-etal-2025-plast (EMNLP Findings 2025)
- resck-etal-2025-explainability (EMNLP 2025)
- golovanevsky-etal-2025-notice (NAACL 2025)
- brinkmann-etal-2025-large (NAACL 2025)
- song-etal-2025-missing (NAACL 2025)
- nyandwi-etal-2025-grounding (EMNLP 2025)
- liu-etal-2024-improved (LLaVA-1.5)
- zou-etal-2023-representation (arXiv 2023)
- meng-etal-2022-locating (NeurIPS 2022)
