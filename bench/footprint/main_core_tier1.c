// SPDX-License-Identifier: Apache-2.0
#include "paci_core.h"
#include "paci_internal.h"
#include <stdio.h>

volatile float g_sink = 0.0f;

int main(void) {
    paci_ctx_t ctx;
    paci_init(&ctx, 1.0f, 0.1f);
    paci_result_t res;
    paci_step(&ctx, 1.05f, 10.0f, 300.0f, 100.0f, 50.0f, &res);
    int8_t window[PACI_WINDOW_SIZE] = {0};
    int8_t class_id = 0;
    int32_t margin = 0;
    paci_infer_t1_s4(window, &class_id, &margin);
    g_sink += res.nis + (float)class_id + (float)margin;
    return 0;
}
