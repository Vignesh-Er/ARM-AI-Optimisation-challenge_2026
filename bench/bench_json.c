// SPDX-License-Identifier: Apache-2.0
#include "bench_json.h"

#include <stdio.h>

static void write_unit(FILE *f, const char *name, const bench_unit_result_t *u, int trailing_comma) {
    fprintf(f, "    \"%s\": {\n", name);
    fprintf(f, "      \"hot\": {\"value\": %.3f, \"mad\": %.3f}", u->hot_value_ns, u->hot_mad_ns);
    if (u->has_cold) {
        fprintf(f, ",\n      \"cold\": {\"value\": %.3f, \"mad\": %.3f}", u->cold_value_ns, u->cold_mad_ns);
    }
    if (u->scratch_bytes >= 0) {
        fprintf(f, ",\n      \"scratch_bytes\": %d", u->scratch_bytes);
    }
    fprintf(f, "\n    }%s\n", trailing_comma ? "," : "");
}

int bench_write_json(const char *path, const bench_report_t *report) {
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        return 1;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"schema_version\": 2,\n");
    fprintf(f, "  \"target\": \"%s\",\n", report->target);
    fprintf(f, "  \"cpu\": \"%s\",\n", report->cpu);
    fprintf(f, "  \"compiler\": \"%s\",\n", report->compiler);
    fprintf(f, "  \"flags\": \"%s\",\n", report->flags);
    fprintf(f, "  \"metric\": \"%s\",\n", report->metric);
    fprintf(f, "  \"note\": \"%s\",\n", report->note);
    fprintf(f, "  \"batches\": %d,\n", report->batches);
    fprintf(f, "  \"model_artifacts\": {\"tier1\": \"%s\", \"tier2\": \"%s\"},\n",
            report->tier1_artifact, report->tier2_artifact);
    fprintf(f, "  \"units\": {\n");
    write_unit(f, "tier0_ekf", &report->tier0_ekf, 1);
    write_unit(f, "tier1_int4", &report->tier1_int4, 1);
    write_unit(f, "tier2_int8", &report->tier2_int8, 1);
    fprintf(f, "    \"cascade_trace\": {\"total\": %llu, \"always_on\": %llu, \"n1\": %u, \"n2\": %u, \"steps\": %u}\n",
            (unsigned long long)report->cascade_trace.total_ns,
            (unsigned long long)report->cascade_trace.always_on_ns,
            report->cascade_trace.n1, report->cascade_trace.n2, report->cascade_trace.steps);
    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}
