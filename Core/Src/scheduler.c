#include "scheduler.h"

void Scheduler_Init(Scheduler_t *sched) {
    sched->step_count = 0;
    sched->cycles_since_cnn = 0;
}

SchedulerDecision_t Scheduler_Step(Scheduler_t *sched, float nis) {
    sched->step_count++;
    
    // 1. Burn-in period check (wait for EKF to stabilize)
    if (sched->step_count <= BURN_IN_STEPS) {
        sched->cycles_since_cnn++;
        return DECISION_SLEEP;
    }
    
    // 2. Anomaly detected (NIS exceeds threshold)
    if (nis > CHI2_THRESHOLD) {
        sched->cycles_since_cnn = 0;
        return DECISION_WAKE_CNN;
    }
    
    // 3. Watchdog timeout (Force CNN to check for slow drifts)
    if (sched->cycles_since_cnn >= WATCHDOG_INTERVAL) {
        sched->cycles_since_cnn = 0;
        return DECISION_WAKE_CNN;
    }
    
    // 4. Default state: sleep
    sched->cycles_since_cnn++;
    return DECISION_SLEEP;
}
