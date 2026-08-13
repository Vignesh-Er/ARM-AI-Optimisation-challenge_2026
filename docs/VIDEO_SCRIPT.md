# PACI-Arm: 3-Minute Video Script

**Target**: Arm AI Optimization Challenge 2026 — Phase 6 Deliverable  
**Format**: 3-minute screen recording presentation.

---

## 0:00 - 0:20 | The Problem (20s)
*(Visual: Slide showing IMU sensor attached to industrial machinery, sending raw data to the cloud)*
**Speaker:** "Continuous vibration monitoring on battery-powered IoT nodes drains power rapidly if we run deep neural networks 100% of the time, or if we stream raw data over RF. We need complex AI capabilities, but we can't afford the constant energy tax."

## 0:20 - 1:00 | The Ladder (40s)
*(Visual: Slide showing the 3-Tier PACI Cascade Architecture)*
**Speaker:** "Enter PACI. We've built an evidence-proportional inference cascade that only computes what it needs to. 
- Tier 0 is a hyper-efficient Physics EKF. It tracks the machine's state using just 20 microseconds per step.
- Only when the EKF detects a statistical anomaly—an NIS threshold breach—do we wake Tier 1. 
- Tier 1 is an ultra-fast INT4 1D-CNN. It acts as a cheap anomaly screener. 
- If Tier 1 confirms the anomaly, we finally trigger Tier 2, our heavier INT8 1D-CNN for detailed failure classification. 
This means we achieve DNN-level accuracy while spending 90% of our time in a micro-watt physics tracker."

## 1:00 - 2:00 | Live Terminal Demo (60s)
*(Visual: Split screen. Left: Presentation. Right: Live Terminal running `python demo_live.py`)*
**Speaker:** "Let's see it in action. I'm running our live terminal demo. 
*(Run `python demo_live.py`)*
Watch the real-time stream. You can see the EKF coasting cleanly through normal noise—that's the green dot. It costs almost nothing. 
Suddenly, a transient shock hits. The EKF detects the divergence, and Tier 1 wakes up (Yellow `T1`). Tier 1 isn't sure, so it wakes Tier 2 (Red `T2` and `!!!`), which successfully classifies a bearing fault. 
Look at the summary at the bottom: Out of 1000 steps, we only woke the neural networks a fraction of the time, dropping our effective compute cost by over 90%."

## 2:00 - 2:40 | The Bias Budget Plot (40s)
*(Visual: Full screen showing `outputs/plots/bias_predicted_vs_measured.png` from `docs/BIAS_BUDGET.md`)*
**Speaker:** "A key challenge in INT8/INT4 cascade inference is ensuring CMSIS-NN rounding errors don't accidentally flip our decision gates. We did the math. We proved the worst-case single-rounding bias is strictly bounded at 1.25 LSBs. 
We didn't just model this—we verified it. Here is our predicted vs. measured margin degradation plot. As you drop precision down to 4 bits, the empirical measured degradation stays comfortably below our theoretical worst-case bound, and well within our minimum 4-LSB security margin. Zero false decisions."

## 2:40 - 3:00 | Reproduce It Yourself (20s)
*(Visual: GitHub repository page showing the instructions and CI status)*
**Speaker:** "And you don't just have to take our word for it. Every plot, every metric, and every trace is entirely reproducible. The GitHub Actions CI compiles the C code, runs the test suite, generates the JSON traces, and builds the bias budget report automatically. Clone the repo, run the CMake build, and verify it yourself on Arm Corstone-300. Thank you."
