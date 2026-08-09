#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Scheduler Constants
#define CHI2_THRESHOLD 3.841f  // 95% confidence threshold for 1 DOF
#define WATCHDOG_INTERVAL 50   // Maximum steps between CNN calls
#define BURN_IN_STEPS 30       // Initial period where CNN stays asleep to let EKF converge

typedef enum {
    DECISION_SLEEP = 0,
    DECISION_WAKE_CNN = 1
} SchedulerDecision_t;

typedef struct {
    uint32_t step_count;
    uint32_t cycles_since_cnn;
} Scheduler_t;

// Initialize the scheduler
void Scheduler_Init(Scheduler_t *sched);

// Process the latest NIS and decide whether to wake the CNN
SchedulerDecision_t Scheduler_Step(Scheduler_t *sched, float nis);

#ifdef __cplusplus
}
#endif

#endif // SCHEDULER_H
