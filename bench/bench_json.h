// SPDX-License-Identifier: Apache-2.0
#ifndef BENCH_JSON_H
#define BENCH_JSON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A fixed-shape writer for outputs/bench/<target>.json's schema v2 (Task
// 3.3) — not a general JSON library, since the schema is fully known ahead
// of time and never varies at runtime beyond the values themselves.

typedef struct {
    double hot_value_ns;
    double hot_mad_ns;
    int has_cold;          // 0 for tier0_ekf, which has no cold variant
    double cold_value_ns;
    double cold_mad_ns;
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
    const char *target;
    const char *cpu;
    const char *compiler;
    const char *flags;
    const char *metric;
    const char *note;
    int batches;
    const char *tier1_artifact;
    const char *tier2_artifact;
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
