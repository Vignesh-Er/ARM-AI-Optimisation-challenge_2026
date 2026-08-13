// SPDX-License-Identifier: Apache-2.0
#ifndef BENCH_JSON_H
#define BENCH_JSON_H

#include <stdint.h>
#include "bench_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

// A fixed-shape writer for outputs/bench/<target>.json's schema v2 (Task
// 3.3) — not a general JSON library, since the schema is fully known ahead
// of time and never varies at runtime beyond the values themselves.

typedef struct {
    bench_stats_t hot;
    bench_stats_t cold;
    int has_cold;
    int32_t scratch_bytes; // -1 if not applicable (tier0_ekf)
} bench_unit_result_t;

typedef struct {
    uint64_t total_ns;
    uint64_t always_on_ns;
    uint32_t n1;
    uint32_t n2;
    uint32_t steps;
} bench_cascade_result_t;

typedef struct {
    char target[32];
    char cpu[64];
    char compiler[64];
    char compiler_version[32];
    char build_type[32];
    char flags[128];
    char metric[32];
    char note[256];
    char git_commit[64];
    char timestamp[64];
    char cmsis_nn_version[32];
    int batches;
    char tier1_artifact[64];
    char tier2_artifact[64];
    bench_unit_result_t tier0_ekf;
    bench_unit_result_t tier1_int4;
    bench_unit_result_t tier2_int8;
    bench_cascade_result_t cascade_trace;
} bench_report_t;

// Returns 0 on success, nonzero on failure to open/write the file.
int bench_write_json(const char *path, const bench_report_t *report);

#ifdef __cplusplus
}
#endif
#endif // BENCH_JSON_H
