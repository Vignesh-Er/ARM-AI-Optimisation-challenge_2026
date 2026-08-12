// SPDX-License-Identifier: Apache-2.0
#ifndef BENCH_TIMING_H
#define BENCH_TIMING_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// CLOCK_MONOTONIC_RAW is a Linux/glibc extension (not in POSIX). On
// platforms lacking it (confirmed: this codebase's Windows/MinGW dev
// environment, used only for local smoke-testing bench_main's logic before
// CI runs it for real on aarch64 Linux) it's mapped to CLOCK_MONOTONIC.
// bench_now_ns() records which one is actually in effect so the JSON `note`
// field can say so.
uint64_t bench_now_ns(void);
bool bench_used_raw_clock(void);

// Sorts a COPY of `values` (does not mutate the caller's array) and writes
// the median and median-absolute-deviation to *out_median / *out_mad.
void bench_median_mad(const double *values, int n, double *out_median, double *out_mad);

// `sched_setaffinity` to pin this process to one core. Linux-only (the
// syscall does not exist elsewhere); logs and returns false rather than
// failing hard if pinning isn't available, per the master spec's "if it
// fails, log and continue".
bool bench_pin_to_core(int core_id);

// Probes `perf stat -e cycles true` at startup. Logs
// "perf: unavailable, wall-clock only" and returns false on any failure —
// never blocks or aborts on a missing/unusable perf.
bool bench_probe_perf(void);

// Cold-cache eviction: walks a large thrash buffer at a fixed stride with a
// data-dependent accumulate (so it cannot be optimised away), evicting
// weights/scratch/window from cache before a COLD measurement. Returns the
// accumulated sum so callers can fold it into their own volatile sink and
// guarantee the whole walk survives optimisation.
uint64_t bench_thrash_cache(void);

// Returns true if `scaled` grew by at least `min_ratio` relative to `base`
// (e.g. min_ratio=1.3 requires at least 30% growth). Used to catch dead-
// code elimination: if doubling the iteration count of a timed loop does
// not measurably increase wall time, the loop body was optimised away.
bool bench_scaling_ok(uint64_t base_ns, uint64_t scaled_ns, double min_ratio);

#ifdef __cplusplus
}
#endif
#endif // BENCH_TIMING_H
