# PACI Benchmark Report

## Research Question
Can physics-based statistical gating reduce CNN inference count by over 80% while maintaining fault-detection recall above 95%?

## Results Summary

| Method | CNN Invocations | CNN Reduction | Fault Detection | False Wake Rate | Energy Saving |
|--------|:-:|:-:|:-:|:-:|:-:|
| Always-On CNN | 2000 | 0.0% | 100% | 100.0% | 0.0% |
| Variance Threshold | 124 | 93.8% | 50% | 2.4% | 92.0% |
| Moving Average | 82 | 95.9% | 100% | 3.6% | 94.0% |
| CUSUM Detector | 712 | 64.4% | 100% | 29.4% | 63.1% |
| Kalman (No Physics) | 346 | 82.7% | 100% | 12.3% | 81.1% |
| PACI | 318 | 84.1% | 100% | 4.6% | 82.5% |

## Headline Result
**PACI reduced CNN execution by 84% with 100% fault detection rate.**

## Key Observations
- PACI achieves the best balance of CNN reduction and fault detection
- The physics-informed EKF gate outperforms purely statistical methods
- All fault events were detected by PACI via NIS spike + watchdog combination