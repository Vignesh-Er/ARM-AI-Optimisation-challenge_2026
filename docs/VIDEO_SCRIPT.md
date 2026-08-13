# PACI-Arm: 2-Minute Video Script

**Target**: Arm AI Optimization Challenge 2026
**Format**: 2-minute screen recording presentation.
**Word Count**: ~255 words (approx. 100-115 seconds spoken)

---

## 0:00 - 0:25 | The Problem (25s)
*(Visual: Slide showing an industrial machine with a battery-powered IMU sensor attached. A graph showing the battery draining rapidly due to constant AI inference)*

**Voiceover:** 
"In industrial IoT, running continuous deep learning models on battery-powered sensors drains power incredibly fast. But if we don't, we might miss critical machine failures. We need complex AI capabilities, but without the constant energy tax. How do we get the best of both worlds?"

## 0:25 - 0:50 | Our Solution: PACI Cascade (25s)
*(Visual: Architecture diagram of PACI (Physics-Aware Cascade Inference), highlighting the 3 tiers: Tier 0 (Physics EKF), Tier 1 (INT4 1D-CNN), and Tier 2 (INT8 1D-CNN))*

**Voiceover:** 
"Enter PACI. We built a three-tier inference cascade that only computes what it absolutely needs to. We start with a hyper-efficient Physics Extended Kalman Filter—our Tier 0. It continuously tracks machine health using just 465 nanoseconds per step. Only when it detects a statistical anomaly—a threshold breach—do we wake up our neural networks."

## 0:50 - 1:20 | The Hard Work: Neural Networks & Bias Budget (30s)
*(Visual: Terminal showing the CMSIS-NN convolutions, transitioning to our "Bias Budget Plot" showing the measured vs. predicted margin degradation from `outputs/plots/bias_predicted_vs_measured.png`)*

**Voiceover:** 
"If an anomaly is found, we escalate to our Tier 1 INT4 CNN, acting as a fast screener. If it confirms the issue, Tier 2—our heavier INT8 CNN—wakes up for detailed failure classification. Building this on CMSIS-NN wasn't trivial. We rigorously mapped the quantization layers and mathematically bounded the rounding bias to exactly 1.25 LSBs. This guarantees absolute mathematical stability and zero false decision flips across our dynamic precision scaling."

## 1:20 - 1:45 | Results: Live Demo vs. Canonical Benchmark (25s)
*(Visual: Split screen. Left side: The 2,000-step canonical benchmark report. Right side: The live terminal demo running the 300-step trace)*

**Voiceover:** 
"The results speak for themselves. In our live 300-step interactive demo, Tier 0 remained at the physics gate—with 169 out of 300 samples requiring no CNN escalation at all, netting an 88.91% compute reduction. And on our full 2,000-step canonical benchmark baseline, we achieved a staggering 90.58% compute reduction, which is a 10.62x speedup compared to running a standard DNN."

## 1:45 - 2:00 | Conclusion (15s)
*(Visual: GitHub repository page showing the instructions, native C code, and GitHub Actions CI passing automatically)*

**Voiceover:** 
"We've implemented this natively in C for Arm Cortex-M devices, backed by fully reproducible CI pipelines. PACI delivers the accuracy of deep learning at the power budget of a simple physics filter. Thank you."
