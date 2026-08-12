// SPDX-License-Identifier: Apache-2.0
#include "paci_core.h"
#include <stdio.h>

volatile float g_sink = 0.0f;

int main(void) {
    paci_ctx_t ctx;
    paci_init(&ctx, 1.0f, 0.1f);
    paci_result_t res;
    paci_step(&ctx, 1.05f, 10.0f, 300.0f, 100.0f, 50.0f, &res);
    g_sink += res.nis;
    return 0;
}
