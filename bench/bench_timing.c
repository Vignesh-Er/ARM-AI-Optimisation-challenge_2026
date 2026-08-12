// SPDX-License-Identifier: Apache-2.0
#include "bench_timing.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__linux__)
#define _GNU_SOURCE
#include <sched.h>
#include <unistd.h>
#define BENCH_HAVE_LINUX_SCHED 1
#endif

// CLOCK_MONOTONIC_RAW is a glibc/Linux extension, not POSIX — falls back to
// CLOCK_MONOTONIC on platforms lacking it (this repo's Windows/MinGW dev
// box, used only to smoke-test bench_main's control-flow logic locally
// before CI measures for real on aarch64 Linux, where CLOCK_MONOTONIC_RAW
// is always available).
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#define BENCH_NO_RAW_CLOCK 1
#endif

uint64_t bench_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

bool bench_used_raw_clock(void) {
#ifdef BENCH_NO_RAW_CLOCK
    return false;
#else
    return true;
#endif
}

static int bench_compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double bench_median_of_sorted(const double *sorted, int n) {
    if (n % 2 == 1) {
        return sorted[n / 2];
    }
    return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

#define BENCH_MAX_SAMPLES 256

void bench_median_mad(const double *values, int n, double *out_median, double *out_mad) {
    double sorted[BENCH_MAX_SAMPLES];
    double deviations[BENCH_MAX_SAMPLES];
    int count = (n > BENCH_MAX_SAMPLES) ? BENCH_MAX_SAMPLES : n;

    for (int i = 0; i < count; i++) {
        sorted[i] = values[i];
    }
    qsort(sorted, (size_t)count, sizeof(double), bench_compare_double);
    double median = bench_median_of_sorted(sorted, count);

    for (int i = 0; i < count; i++) {
        double d = values[i] - median;
        deviations[i] = (d < 0.0) ? -d : d;
    }
    qsort(deviations, (size_t)count, sizeof(double), bench_compare_double);

    *out_median = median;
    *out_mad = bench_median_of_sorted(deviations, count);
}

bool bench_pin_to_core(int core_id) {
#ifdef BENCH_HAVE_LINUX_SCHED
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core_id, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "bench: sched_setaffinity failed, continuing unpinned\n");
        return false;
    }
    return true;
#else
    (void)core_id;
    fprintf(stderr, "bench: sched_setaffinity not available on this platform, continuing unpinned\n");
    return false;
#endif
}

bool bench_probe_perf(void) {
#ifdef __linux__
    int rc = system("perf stat -e cycles true > /dev/null 2>&1");
    if (rc != 0) {
        fprintf(stderr, "perf: unavailable, wall-clock only\n");
        return false;
    }
    return true;
#else
    fprintf(stderr, "perf: unavailable, wall-clock only\n");
    return false;
#endif
}

#define BENCH_THRASH_BYTES (64u * 1024u * 1024u)
#define BENCH_THRASH_STRIDE 64u

static unsigned char *bench_thrash_buffer(void) {
    static unsigned char *buf = NULL;
    if (buf == NULL) {
        buf = (unsigned char *)malloc(BENCH_THRASH_BYTES);
        if (buf != NULL) {
            for (uint32_t i = 0; i < BENCH_THRASH_BYTES; i++) {
                buf[i] = (unsigned char)(i * 2654435761u);
            }
        }
    }
    return buf;
}

uint64_t bench_thrash_cache(void) {
    unsigned char *buf = bench_thrash_buffer();
    if (buf == NULL) {
        return 0;
    }
    uint64_t acc = 0;
    for (uint32_t i = 0; i < BENCH_THRASH_BYTES; i += BENCH_THRASH_STRIDE) {
        acc += buf[i];
        buf[i] = (unsigned char)(acc & 0xFFu);  // data-dependent write: survives DCE/hoisting
    }
    return acc;
}

bool bench_scaling_ok(uint64_t base_ns, uint64_t scaled_ns, double min_ratio) {
    if (base_ns == 0) {
        return false;
    }
    double ratio = (double)scaled_ns / (double)base_ns;
    return ratio >= min_ratio;
}
