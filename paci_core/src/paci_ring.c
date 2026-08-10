// SPDX-License-Identifier: Apache-2.0
#include "paci_internal.h"

// D2 fix: the original Core/Src/main_cascade.c wrote
// sensor_buffer[buffer_index % WINDOW_SIZE] and then passed sensor_buffer
// straight to inference, so unless buffer_index % 32 == 0 the window handed
// to the CNN was rotated (time-scrambled). paci_ring_push only advances the
// write head; linearisation into chronological order happens exclusively in
// paci_ring_read below.
void paci_ring_push(paci_ring_t *ring, int8_t sample) {
    ring->buf[ring->head] = sample;
    ring->head = (uint8_t)((ring->head + 1) % PACI_WINDOW_SIZE);
    ring->total++;
}

// D3 fix: total is uint32_t (never wraps in practice at one sample/step),
// so "primed" is a real comparison rather than a uint8_t that silently
// wraps at 256 and skips 32 consecutive windows.
paci_status_t paci_ring_read(const paci_ring_t *r, int8_t out[PACI_WINDOW_SIZE]) {
    if (r == NULL || out == NULL) {
        return PACI_E_NULL;
    }
    if (r->total < PACI_WINDOW_SIZE) {
        return PACI_E_UNPRIMED;
    }

    // Once the ring is full, r->head points at the slot that will be
    // overwritten next — i.e. the oldest live sample.
    uint8_t oldest = r->head;
    for (uint32_t i = 0; i < PACI_WINDOW_SIZE; i++) {
        out[i] = r->buf[(oldest + i) % PACI_WINDOW_SIZE];
    }
    return PACI_OK;
}
