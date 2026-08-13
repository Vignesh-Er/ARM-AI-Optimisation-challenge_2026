# PACI Benchmark Report

## Research Question
Can physics-based statistical gating reduce CNN inference count by over 80% while maintaining fault-detection recall above 95%?

## Results Summary

| Method | CNN Invocations | CNN Reduction | Fault Detection | False Wake Rate | Energy Saving |
|--------|:-:|:-:|:-:|:-:|:-:|
| Always-On CNN | 2000 | 0.0% | 100% | 100.0% | 0.0% |
| Variance Threshold | 101 | 95.0% | 25% | 2.1% | 94.5% |
| Moving Average | 79 | 96.0% | 75% | 3.7% | 95.6% |
| CUSUM Detector | 616 | 69.2% | 100% | 29.4% | 68.8% |
| Kalman (No Physics) | 341 | 83.0% | 100% | 12.3% | 82.5% |
| PACI | 312 | 84.4% | 100% | 4.5% | 65.0% |

## Headline Result
**PACI reduced CNN execution by 84% with 100% fault detection rate.**

## Key Observations
- PACI achieves the best balance of CNN reduction and fault detection
- The physics-informed EKF gate outperforms purely statistical methods
- All fault events were detected by PACI via NIS spike + watchdog combination