# Cover Letter
**Manuscript:** Calibration Objectives and Evaluation Protocols in 4-bit Edge LLM Deployment
**Journal:** Neurocomputing (Elsevier)
**Article type:** Regular article

---

Dear Editor-in-Chief,

We are pleased to submit our manuscript, *"Calibration Objectives and Evaluation Protocols in 4-bit Edge LLM Deployment,"* for consideration as a Regular article in *Neurocomputing*.

Quantization is the default route to running large language models on edge NPUs, yet the calibration step that precedes deployment is almost always validated in a floating-point simulation rather than on the target device. This paper is a controlled, reproducible measurement study of that disconnect. We fix one W4A16 pipeline, one 1.5B model, and one edge SoC (Rockchip RK3576), and vary the calibration objective across five methods (min-max, GPTQ, AWQ, NAC-v1, NAC-v2). Three results we believe are of direct interest to the neurocomputing and edge-deployment communities:

1. **The protocol, not the calibration method, separates the numbers.** Four quantized calibration objectives that are separated by perplexity and multiple-choice accuracy are statistically indistinguishable on long-form GSM8K arithmetic under a contamination-free protocol (paired McNemar p ≥ 0.31). An earlier apparent 4× gap was an artefact of scoring untruncated generations.
2. **A generation cap reverses the verdict.** A 32-token cap — a common harness default — drives even the vendor reference model to 0.0000 on GSM8K; at 512 tokens the same artifacts score 0.535–0.565 and become statistically tied with the vendor. This is a measurement-validity finding with direct consequences for published edge-deployment numbers.
3. **Calibration effects are conditional on a hyperparameter.** The sign of a calibration effect is reversed by group size (NAC-v1: −7.43 pp of MMLU at g64, +6.71 pp at g128).

We make no state-of-the-art claim; the paper's contribution is a controlled demonstration that, on a real edge deployment stack, the effect of a calibration objective is frequently *unidentifiable* unless the evaluation protocol and the build path are fixed. We want to be explicit about what this manuscript does and does not claim, so that it is not mistaken for a benchmark report: this paper does **not** propose a new calibration algorithm, and it does **not** report a ranking of methods. Instead it proves a methodological point that bears directly on how learning systems are compared on edge hardware — that the conclusions drawn from floating-point simulation about the relative quality of quantized variants do not transfer to the device, because the protocol and the build path dominate the measurement and can hide or even reverse a method effect. We provide a reproducible discrimination protocol, a quantified account of the two protocol artefacts, a build-reproducibility result, and a protocol checklist backed by measured failures.

These results address *Neurocomputing*'s scope on two fronts: they advance the understanding of post-training quantization of neural networks, and they provide a reproducible evaluation framework for deployment-oriented learning systems, where recent literature has demonstrated that off-device conclusions do not transfer to accelerated hardware.

We confirm that this work is original, has not been published elsewhere, and is not under consideration by any other venue.

Thank you for your consideration.

Sincerely,

**YIBO ZHANG**
Corresponding author
Universiti Utara Malaysia
06010 UUM Sintok, Kedah, Malaysia
E-mail: zhang_yibo@ahsgs.uum.edu.my

On behalf of co-authors:
Muhammad Syafiq bin Mohd Pozi, Osman B Ghazali (Universiti Utara Malaysia)