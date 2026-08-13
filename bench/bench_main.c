// SPDX-License-Identifier: Apache-2.0
#include "bench_timing.h"
#include "bench_json.h"
#include "bench_trace.h"
#include "paci_core.h"
#include "paci_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_WARMUP_BATCHES 3
#define N_MEASURED_BATCHES 31
#define MIN_BATCH_DURATION_NS 50000000ULL // 50 ms

// Global volatile sink to prevent dead-code elimination across loops
static volatile int64_t g_volatile_sink = 0;

#if defined(__aarch64__)
#define BENCH_DEFAULT_TARGET "aarch64-linux"
#elif defined(__x86_64__) || defined(_M_X64)
#define BENCH_DEFAULT_TARGET "x86_64-native"
#elif defined(__arm__)
#define BENCH_DEFAULT_TARGET "arm-cortex"
#else
#define BENCH_DEFAULT_TARGET "unknown-target"
#endif

#if defined(__GNUC__)
#define BENCH_COMPILER_STR "gcc " __VERSION__
#elif defined(__clang__)
#define BENCH_COMPILER_STR "clang " __clang_version__
#else
#define BENCH_COMPILER_STR "unknown compiler"
#endif

// Determine representative CPU name
static void get_cpu_info(char *buf, size_t max_len) {
#if defined(__linux__)
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f != NULL) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0 || strncmp(line, "Model", 5) == 0 || strncmp(line, "Hardware", 8) == 0) {
                char *colon = strchr(line, ':');
                if (colon != NULL) {
                    colon++;
                    while (*colon == ' ' || *colon == '\t') colon++;
                    size_t len = strlen(colon);
                    while (len > 0 && (colon[len - 1] == '\n' || colon[len - 1] == '\r' || colon[len - 1] == ' ')) {
                        colon[--len] = '\0';
                    }
                    strncpy(buf, colon, max_len - 1);
                    buf[max_len - 1] = '\0';
                    fclose(f);
                    return;
                }
            }
        }
        fclose(f);
    }
#endif
#if defined(__aarch64__)
    strncpy(buf, "Arm Neoverse / Cortex-A (aarch64)", max_len - 1);
#else
    strncpy(buf, "Generic CPU", max_len - 1);
#endif
    buf[max_len - 1] = '\0';
}

// ---------------------------------------------------------------------------
// Tier 0 EKF Measurement
// ---------------------------------------------------------------------------
static void run_tier0_ekf_once(paci_ctx_t *ctx, float z, float p, float t, float w, float f) {
    float pred = paci_physics_predict(p, t, w, f);
    float nis = 0.0f;
    paci_status_t status = paci_ekf_step(&ctx->ekf, z, pred, &nis);
    paci_ring_push(&ctx->ring, 0);
    g_volatile_sink += (int64_t)(nis * 1000.0f) + (int64_t)status;
}

static uint32_t calibrate_tier0_iterations(void) {
    paci_ctx_t ctx;
    paci_init(&ctx, PACI_ETCH_RATE_NOMINAL, PACI_P0_VAR);
    uint64_t t0 = bench_now_ns();
    uint32_t iters = 1000;
    for (uint32_t i = 0; i < iters; i++) {
        run_tier0_ekf_once(&ctx, PACI_ETCH_RATE_NOMINAL, 10.0f, 300.0f, 500.0f, 50.0f);
    }
    uint64_t elapsed = bench_now_ns() - t0;
    if (elapsed == 0) elapsed = 1;
    uint32_t target = (uint32_t)((double)MIN_BATCH_DURATION_NS * (double)iters / (double)elapsed);
    if (target < 100) target = 100;
    return target;
}

static bench_unit_result_t measure_tier0_ekf(void) {
    bench_unit_result_t res;
    memset(&res, 0, sizeof(res));
    res.has_cold = 0;
    res.scratch_bytes = -1;

    uint32_t iters_per_batch = calibrate_tier0_iterations();
    paci_ctx_t ctx;
    paci_init(&ctx, PACI_ETCH_RATE_NOMINAL, PACI_P0_VAR);

    // Warmup
    for (int b = 0; b < N_WARMUP_BATCHES; b++) {
        for (uint32_t i = 0; i < iters_per_batch; i++) {
            run_tier0_ekf_once(&ctx, PACI_ETCH_RATE_NOMINAL, 10.0f, 300.0f, 500.0f, 50.0f);
        }
    }

    // Measured batches
    double batch_ns_per_op[N_MEASURED_BATCHES];
    for (int b = 0; b < N_MEASURED_BATCHES; b++) {
        uint64_t t0 = bench_now_ns();
        for (uint32_t i = 0; i < iters_per_batch; i++) {
            run_tier0_ekf_once(&ctx, PACI_ETCH_RATE_NOMINAL, 10.0f, 300.0f, 500.0f, 50.0f);
        }
        uint64_t dt = bench_now_ns() - t0;
        batch_ns_per_op[b] = (double)dt / (double)iters_per_batch;
    }

    bench_median_mad(batch_ns_per_op, N_MEASURED_BATCHES, &res.hot_value_ns, &res.hot_mad_ns);
    return res;
}

// ---------------------------------------------------------------------------
// Tier 1 INT4 Measurement (Hot & Cold)
// ---------------------------------------------------------------------------
static uint32_t calibrate_tier1_iterations(const int8_t window[PACI_WINDOW_SIZE]) {
    int8_t class_id = 0;
    int32_t margin = 0;
    uint64_t t0 = bench_now_ns();
    uint32_t iters = 100;
    for (uint32_t i = 0; i < iters; i++) {
        paci_infer_t1_s4(window, &class_id, &margin);
        g_volatile_sink += class_id + margin;
    }
    uint64_t elapsed = bench_now_ns() - t0;
    if (elapsed == 0) elapsed = 1;
    uint32_t target = (uint32_t)((double)MIN_BATCH_DURATION_NS * (double)iters / (double)elapsed);
    if (target < 10) target = 10;
    return target;
}

static bench_unit_result_t measure_tier1_int4(const int8_t window[PACI_WINDOW_SIZE]) {
    bench_unit_result_t res;
    memset(&res, 0, sizeof(res));
    res.has_cold = 1;
    res.scratch_bytes = 8192; // Max static scratch buffer

    uint32_t hot_iters = calibrate_tier1_iterations(window);
    int8_t class_id = 0;
    int32_t margin = 0;

    // HOT Warmup
    for (int b = 0; b < N_WARMUP_BATCHES; b++) {
        for (uint32_t i = 0; i < hot_iters; i++) {
            paci_infer_t1_s4(window, &class_id, &margin);
            g_volatile_sink += class_id + margin;
        }
    }

    // HOT Measured
    double hot_ns_per_op[N_MEASURED_BATCHES];
    for (int b = 0; b < N_MEASURED_BATCHES; b++) {
        uint64_t t0 = bench_now_ns();
        for (uint32_t i = 0; i < hot_iters; i++) {
            paci_infer_t1_s4(window, &class_id, &margin);
            g_volatile_sink += class_id + margin;
        }
        uint64_t dt = bench_now_ns() - t0;
        hot_ns_per_op[b] = (double)dt / (double)hot_iters;
    }
    bench_median_mad(hot_ns_per_op, N_MEASURED_BATCHES, &res.hot_value_ns, &res.hot_mad_ns);

    // COLD Measured (smaller iteration count per batch since cache thrashing is ~1-2ms per step)
    uint32_t cold_iters = 10;
    if (cold_iters > hot_iters) cold_iters = hot_iters;
    if (cold_iters < 5) cold_iters = 5;

    // COLD Warmup
    for (int b = 0; b < N_WARMUP_BATCHES; b++) {
        for (uint32_t i = 0; i < cold_iters; i++) {
            g_volatile_sink += (int64_t)bench_thrash_cache();
            uint64_t t0 = bench_now_ns();
            paci_infer_t1_s4(window, &class_id, &margin);
            uint64_t dt = bench_now_ns() - t0;
            g_volatile_sink += (int64_t)dt + class_id;
        }
    }

    // COLD Measured
    double cold_ns_per_op[N_MEASURED_BATCHES];
    for (int b = 0; b < N_MEASURED_BATCHES; b++) {
        uint64_t batch_total_ns = 0;
        for (uint32_t i = 0; i < cold_iters; i++) {
            g_volatile_sink += (int64_t)bench_thrash_cache();
            uint64_t t0 = bench_now_ns();
            paci_infer_t1_s4(window, &class_id, &margin);
            uint64_t dt = bench_now_ns() - t0;
            batch_total_ns += dt;
            g_volatile_sink += class_id + margin;
        }
        cold_ns_per_op[b] = (double)batch_total_ns / (double)cold_iters;
    }
    bench_median_mad(cold_ns_per_op, N_MEASURED_BATCHES, &res.cold_value_ns, &res.cold_mad_ns);

    return res;
}

// ---------------------------------------------------------------------------
// Tier 2 INT8 Measurement (Hot & Cold)
// ---------------------------------------------------------------------------
static uint32_t calibrate_tier2_iterations(const int8_t window[PACI_WINDOW_SIZE]) {
    int8_t class_id = 0;
    int32_t margin = 0;
    uint64_t t0 = bench_now_ns();
    uint32_t iters = 50;
    for (uint32_t i = 0; i < iters; i++) {
        paci_infer_t2_s8(window, &class_id, &margin);
        g_volatile_sink += class_id + margin;
    }
    uint64_t elapsed = bench_now_ns() - t0;
    if (elapsed == 0) elapsed = 1;
    uint32_t target = (uint32_t)((double)MIN_BATCH_DURATION_NS * (double)iters / (double)elapsed);
    if (target < 5) target = 5;
    return target;
}

static bench_unit_result_t measure_tier2_int8(const int8_t window[PACI_WINDOW_SIZE]) {
    bench_unit_result_t res;
    memset(&res, 0, sizeof(res));
    res.has_cold = 1;
    res.scratch_bytes = 8192; // Max static scratch buffer

    uint32_t hot_iters = calibrate_tier2_iterations(window);
    int8_t class_id = 0;
    int32_t margin = 0;

    // HOT Warmup
    for (int b = 0; b < N_WARMUP_BATCHES; b++) {
        for (uint32_t i = 0; i < hot_iters; i++) {
            paci_infer_t2_s8(window, &class_id, &margin);
            g_volatile_sink += class_id + margin;
        }
    }

    // HOT Measured
    double hot_ns_per_op[N_MEASURED_BATCHES];
    for (int b = 0; b < N_MEASURED_BATCHES; b++) {
        uint64_t t0 = bench_now_ns();
        for (uint32_t i = 0; i < hot_iters; i++) {
            paci_infer_t2_s8(window, &class_id, &margin);
            g_volatile_sink += class_id + margin;
        }
        uint64_t dt = bench_now_ns() - t0;
        hot_ns_per_op[b] = (double)dt / (double)hot_iters;
    }
    bench_median_mad(hot_ns_per_op, N_MEASURED_BATCHES, &res.hot_value_ns, &res.hot_mad_ns);

    // COLD Measured
    uint32_t cold_iters = 10;
    if (cold_iters > hot_iters) cold_iters = hot_iters;
    if (cold_iters < 5) cold_iters = 5;

    // COLD Warmup
    for (int b = 0; b < N_WARMUP_BATCHES; b++) {
        for (uint32_t i = 0; i < cold_iters; i++) {
            g_volatile_sink += (int64_t)bench_thrash_cache();
            uint64_t t0 = bench_now_ns();
            paci_infer_t2_s8(window, &class_id, &margin);
            uint64_t dt = bench_now_ns() - t0;
            g_volatile_sink += (int64_t)dt + class_id;
        }
    }

    // COLD Measured
    double cold_ns_per_op[N_MEASURED_BATCHES];
    for (int b = 0; b < N_MEASURED_BATCHES; b++) {
        uint64_t batch_total_ns = 0;
        for (uint32_t i = 0; i < cold_iters; i++) {
            g_volatile_sink += (int64_t)bench_thrash_cache();
            uint64_t t0 = bench_now_ns();
            paci_infer_t2_s8(window, &class_id, &margin);
            uint64_t dt = bench_now_ns() - t0;
            batch_total_ns += dt;
            g_volatile_sink += class_id + margin;
        }
        cold_ns_per_op[b] = (double)batch_total_ns / (double)cold_iters;
    }
    bench_median_mad(cold_ns_per_op, N_MEASURED_BATCHES, &res.cold_value_ns, &res.cold_mad_ns);

    return res;
}

// ---------------------------------------------------------------------------
// Cascade Trace Measurement
// ---------------------------------------------------------------------------
static bench_cascade_result_t measure_cascade_trace(void) {
    bench_cascade_result_t res;
    memset(&res, 0, sizeof(res));
    res.steps = BENCH_TRACE_STEPS;

    paci_ctx_t ctx;
    paci_init(&ctx, PACI_ETCH_RATE_NOMINAL, PACI_P0_VAR);

    uint64_t t0 = bench_now_ns();
    for (uint32_t i = 0; i < BENCH_TRACE_STEPS; i++) {
        paci_result_t out;
        paci_step(&ctx, bench_trace_z[i],
                  bench_trace_pressure[i], bench_trace_temperature[i],
                  bench_trace_power[i],    bench_trace_flow[i],
                  &out);
        g_volatile_sink += out.class_id + (int64_t)out.nis;
    }
    res.total_ns = bench_now_ns() - t0;
    res.n1 = ctx.n_t1;
    res.n2 = ctx.n_t2;

    int8_t dummy_window[PACI_WINDOW_SIZE] = {0};
    uint64_t t1 = bench_now_ns();
    for (uint32_t i = 0; i < BENCH_TRACE_STEPS; i++) {
        int8_t class_id = 0;
        int32_t margin = 0;
        paci_infer_t2_s8(dummy_window, &class_id, &margin);
        g_volatile_sink += class_id + margin;
    }
    res.always_on_ns = bench_now_ns() - t1;

    return res;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    const char *output_path = "outputs/bench/aarch64-linux.json";
    if (argc > 1) {
        output_path = argv[1];
    }

    printf("=================================================================\n");
    printf(" PACI Benchmark Harness (Schema v2 Native Runner)\n");
    printf("=================================================================\n");

    bench_pin_to_core(0);
    bench_probe_perf();

    char cpu_name[128];
    get_cpu_info(cpu_name, sizeof(cpu_name));
    printf("Target CPU:   %s\n", cpu_name);
    printf("Compiler:     %s\n", BENCH_COMPILER_STR);
    printf("Output JSON:  %s\n", output_path);
    printf("-----------------------------------------------------------------\n");

    // Prime a test window from trace
    int8_t test_window[PACI_WINDOW_SIZE];
    for (int i = 0; i < PACI_WINDOW_SIZE; i++) {
        float norm = (bench_trace_z[i] - PACI_ETCH_RATE_NOMINAL) / PACI_NORM_SCALE;
        float q = norm / 0.25f;
        int32_t iq = (int32_t)(q >= 0 ? q + 0.5f : q - 0.5f);
        if (iq > 127) iq = 127;
        if (iq < -128) iq = -128;
        test_window[i] = (int8_t)iq;
    }

    bench_report_t report;
    memset(&report, 0, sizeof(report));
    report.target = BENCH_DEFAULT_TARGET;
    report.cpu = cpu_name;
    report.compiler = BENCH_COMPILER_STR;
    report.flags = "-O2 -fno-fast-math -ffp-contract=off";
    report.metric = "ns_median";
    report.note = "Hot and cold cache latency in ns. Cold numbers use 64MB thrashing. FVP numbers require instruction differencing.";
    report.batches = N_MEASURED_BATCHES;
    report.tier1_artifact = "tier1_model.tflite";
    report.tier2_artifact = "tier2_model.tflite";

    printf("Measuring tier0_ekf (hot)... ");
    fflush(stdout);
    report.tier0_ekf = measure_tier0_ekf();
    printf("%.2f ns (MAD: %.2f ns)\n", report.tier0_ekf.hot_value_ns, report.tier0_ekf.hot_mad_ns);

    printf("Measuring tier1_int4 (hot/cold)... ");
    fflush(stdout);
    report.tier1_int4 = measure_tier1_int4(test_window);
    printf("Hot: %.2f ns (MAD: %.2f ns) | Cold: %.2f ns (MAD: %.2f ns)\n",
           report.tier1_int4.hot_value_ns, report.tier1_int4.hot_mad_ns,
           report.tier1_int4.cold_value_ns, report.tier1_int4.cold_mad_ns);

    printf("Measuring tier2_int8 (hot/cold)... ");
    fflush(stdout);
    report.tier2_int8 = measure_tier2_int8(test_window);
    printf("Hot: %.2f ns (MAD: %.2f ns) | Cold: %.2f ns (MAD: %.2f ns)\n",
           report.tier2_int8.hot_value_ns, report.tier2_int8.hot_mad_ns,
           report.tier2_int8.cold_value_ns, report.tier2_int8.cold_mad_ns);

    printf("Measuring cascade_trace (%d steps)... ", BENCH_TRACE_STEPS);
    fflush(stdout);
    report.cascade_trace = measure_cascade_trace();
    printf("Total: %.2f ms | N1 wakes: %u | N2 wakes: %u\n",
           (double)report.cascade_trace.total_ns / 1e6,
           report.cascade_trace.n1, report.cascade_trace.n2);

    // MAD/median quality checks (warn or flag if MAD > 10%)
    int noisy_count = 0;
    if (report.tier0_ekf.hot_mad_ns / report.tier0_ekf.hot_value_ns > 0.10) {
        fprintf(stderr, "WARNING: tier0_ekf MAD/median > 10%% (MAD=%.2f, Median=%.2f)\n",
                report.tier0_ekf.hot_mad_ns, report.tier0_ekf.hot_value_ns);
        noisy_count++;
    }
    if (report.tier1_int4.hot_mad_ns / report.tier1_int4.hot_value_ns > 0.10) {
        fprintf(stderr, "WARNING: tier1_int4 hot MAD/median > 10%% (MAD=%.2f, Median=%.2f)\n",
                report.tier1_int4.hot_mad_ns, report.tier1_int4.hot_value_ns);
        noisy_count++;
    }
    if (report.tier2_int8.hot_mad_ns / report.tier2_int8.hot_value_ns > 0.10) {
        fprintf(stderr, "WARNING: tier2_int8 hot MAD/median > 10%% (MAD=%.2f, Median=%.2f)\n",
                report.tier2_int8.hot_mad_ns, report.tier2_int8.hot_value_ns);
        noisy_count++;
    }

    if (noisy_count > 0) {
        printf("Note: %d unit(s) exhibited MAD/median > 10%% variation on this environment.\n", noisy_count);
    }

    if (bench_write_json(output_path, &report) != 0) {
        fprintf(stderr, "ERROR: failed to write output JSON to %s\n", output_path);
        return 1;
    }

    printf("-----------------------------------------------------------------\n");
    printf("Benchmark results written to: %s\n", output_path);
    printf("Sink accumulator value: %lld\n", (long long)g_volatile_sink);
    printf("=================================================================\n");

#if PACI_TRACE_REQUANT
    extern void paci_dump_requant_bias(void);
    paci_dump_requant_bias();
#endif
    return 0;
}
