// SPDX-License-Identifier: Apache-2.0
#include "paci_internal.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"  // arm_nn_requantize (paci_global_avgpool_s8)
#include <string.h>

#if __has_include("tier2_weights.h")
#include "tier2_weights.h"
#define PACI_HAVE_TIER2 1
#else
#define PACI_HAVE_TIER2 0
#endif

#if __has_include("tier1_weights.h")
#include "tier1_weights.h"
#define PACI_HAVE_TIER1 1
#else
#define PACI_HAVE_TIER1 0
#endif

// G6: static scratch buffer, no dynamic allocation anywhere in paci_core/.
// Sized generously for this architecture's tiny per-layer channel counts
// (<=32) and checked against each layer's *_get_buffer_size() at runtime
// (G6) rather than trusting this constant alone. 16-byte alignment is
// required for MVE (Helium) builds; harmless on host/DSP builds.
// Overridable via -DPACI_SCRATCH_BYTES=N (see paci_core_tiny_scratch in
// CMakeLists.txt) so tests/test_infer_t2.py can build a deliberately
// undersized variant and confirm PACI_E_BUFSIZE actually fires, per G6's
// "tested by deliberately under-sizing a buffer" requirement — this is not
// reachable by feeding paci_infer_t2_s8 a bad *argument*, since the buffer
// isn't caller-supplied.
#ifndef PACI_SCRATCH_BYTES
#define PACI_SCRATCH_BYTES 8192
#endif
static int8_t paci_scratch[PACI_SCRATCH_BYTES] __attribute__((aligned(16)));

#if PACI_HAVE_TIER1 || PACI_HAVE_TIER2

// Shared by both tiers: conv1 in both architectures is plain INT8
// (arm_convolve_wrapper_s8 — the generic wrapper, per section 5's guidance
// to prefer it over the MVE-only variants that return NO_IMPL_ERROR on a
// non-MVE build).
static paci_status_t paci_conv1d_s8(const int8_t *input, int8_t *output,
                                     int32_t in_w, int32_t in_c, int32_t out_c, int32_t k,
                                     const int8_t *weight, const int32_t *bias,
                                     const int32_t *multiplier, const int32_t *shift,
                                     int32_t input_offset, int32_t output_offset) {
    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    conv_params.input_offset = input_offset;
    conv_params.output_offset = output_offset;
    conv_params.stride.w = 1;
    conv_params.stride.h = 1;
    conv_params.padding.w = (k - 1) / 2;
    conv_params.padding.h = 0;
    conv_params.dilation.w = 1;
    conv_params.dilation.h = 1;
    conv_params.activation.min = -128;
    conv_params.activation.max = 127;

    quant_params.multiplier = (int32_t *)multiplier;
    quant_params.shift = (int32_t *)shift;

    input_dims.n = 1; input_dims.h = 1; input_dims.w = in_w; input_dims.c = in_c;
    filter_dims.n = out_c; filter_dims.h = 1; filter_dims.w = k; filter_dims.c = in_c;
    bias_dims.n = 1; bias_dims.h = 1; bias_dims.w = 1; bias_dims.c = out_c;
    output_dims.n = 1; output_dims.h = 1; output_dims.w = in_w; output_dims.c = out_c;

    int32_t required = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    if (required > (int32_t)PACI_SCRATCH_BYTES) {
        return PACI_E_BUFSIZE;
    }
    ctx.buf = paci_scratch;
    ctx.size = PACI_SCRATCH_BYTES;

    arm_cmsis_nn_status status = arm_convolve_wrapper_s8(
        &ctx, &conv_params, &quant_params,
        &input_dims, input, &filter_dims, weight, &bias_dims, bias, &output_dims, output);

    return (status == ARM_CMSIS_NN_SUCCESS) ? PACI_OK : PACI_E_QUANT;
}

// arm_avgpool_s8's signature (cmsis_nn_pool_params: stride/padding/activation
// only) has no input/output offset or multiplier/shift, i.e. it assumes
// input and output share one scale. The actual trained models'
// GlobalAveragePooling1D lowers to a quantized MEAN op whose input/output
// tensors have INDEPENDENTLY DIFFERENT scales (confirmed by inspecting the
// converted .tflite directly — docs/STATUS.md GATE 2.1), so arm_avgpool_s8
// is not a valid substitute here. This reproduces TFLite's actual int8 Mean
// kernel instead: accumulate the raw sum, requantize once with a
// multiplier/shift derived from input_scale/(W*output_scale), add
// output_offset — the same accumulate-then-arm_nn_requantize pattern
// CMSIS-NN's own kernels use throughout, just not packaged as a kernel
// CMSIS-NN ships for this specific rescaling case. Matches
// tools/ref_cmsisnn.py:global_average_pool_s8 exactly (verified together
// against the real TFLite interpreter, 200/200 exact, before this port).
static void paci_global_avgpool_s8(const int8_t *input, int8_t *output,
                                    int32_t w, int32_t c,
                                    int32_t multiplier, int32_t shift,
                                    int32_t input_offset, int32_t output_offset) {
    for (int32_t ch = 0; ch < c; ch++) {
        int64_t acc = 0;
        for (int32_t i = 0; i < w; i++) {
            acc += input[i * c + ch];
        }
        acc += (int64_t)input_offset * w;

        int32_t requantized = arm_nn_requantize((int32_t)acc, multiplier, shift);
        int32_t val = requantized + output_offset;
        if (val < -128) val = -128;
        if (val > 127) val = 127;
        output[ch] = (int8_t)val;
    }
}

static void paci_top2_margin(const int8_t *logits, int32_t n, int8_t *class_id, int32_t *margin) {
    int32_t best = logits[0];
    int32_t second = INT32_MIN;
    int8_t best_idx = 0;
    for (int32_t i = 1; i < n; i++) {
        if (logits[i] > best) {
            second = best;
            best = logits[i];
            best_idx = (int8_t)i;
        } else if (logits[i] > second) {
            second = logits[i];
        }
    }
    *class_id = best_idx;
    *margin = best - second;
}

#endif  // PACI_HAVE_TIER1 || PACI_HAVE_TIER2

#if PACI_HAVE_TIER2

static paci_status_t paci_dense_s8(const int8_t *input, int8_t *output,
                                    int32_t in_c, int32_t out_c,
                                    const int8_t *weight, const int32_t *bias,
                                    const int32_t *multiplier, const int32_t *shift,
                                    int32_t input_offset, int32_t output_offset,
                                    int32_t activation_min, int32_t activation_max) {
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    fc_params.input_offset = input_offset;
    fc_params.filter_offset = 0;
    fc_params.output_offset = output_offset;
    fc_params.activation.min = activation_min;
    fc_params.activation.max = activation_max;

    quant_params.multiplier = (int32_t *)multiplier;
    quant_params.shift = (int32_t *)shift;

    input_dims.n = 1; input_dims.h = 1; input_dims.w = 1; input_dims.c = in_c;
    filter_dims.n = in_c; filter_dims.h = 1; filter_dims.w = 1; filter_dims.c = out_c;
    bias_dims.n = 1; bias_dims.h = 1; bias_dims.w = 1; bias_dims.c = out_c;
    output_dims.n = 1; output_dims.h = 1; output_dims.w = 1; output_dims.c = out_c;

    int32_t required = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    if (required > (int32_t)PACI_SCRATCH_BYTES) {
        return PACI_E_BUFSIZE;
    }
    ctx.buf = paci_scratch;
    ctx.size = PACI_SCRATCH_BYTES;

    arm_cmsis_nn_status status = arm_fully_connected_per_channel_s8(
        &ctx, &fc_params, &quant_params,
        &input_dims, input, &filter_dims, weight, &bias_dims, bias, &output_dims, output);

    return (status == ARM_CMSIS_NN_SUCCESS) ? PACI_OK : PACI_E_QUANT;
}

paci_status_t paci_infer_t2_s8(const int8_t window[PACI_WINDOW_SIZE], int8_t *class_id, int32_t *margin) {
    if (window == NULL || class_id == NULL || margin == NULL) {
        return PACI_E_NULL;
    }

    static int8_t conv1_out[PACI_WINDOW_SIZE * TIER2_CONV1_N_CHANNELS];
    static int8_t conv2_out[PACI_WINDOW_SIZE * TIER2_CONV2_N_CHANNELS];
    static int8_t pooled[TIER2_CONV2_N_CHANNELS];
    static int8_t dense1_out[TIER2_DENSE1_N_CHANNELS];
    static int8_t logits_out[TIER2_LOGITS_N_CHANNELS];

    paci_status_t status;

    status = paci_conv1d_s8(window, conv1_out, PACI_WINDOW_SIZE, 1, TIER2_CONV1_N_CHANNELS, 5,
                             tier2_conv1_weight, tier2_conv1_bias, tier2_conv1_multiplier, tier2_conv1_shift,
                             TIER2_CONV1_INPUT_OFFSET, TIER2_CONV1_OUTPUT_OFFSET);
    if (status != PACI_OK) return status;

    status = paci_conv1d_s8(conv1_out, conv2_out, PACI_WINDOW_SIZE, TIER2_CONV1_N_CHANNELS, TIER2_CONV2_N_CHANNELS, 3,
                             tier2_conv2_weight, tier2_conv2_bias, tier2_conv2_multiplier, tier2_conv2_shift,
                             TIER2_CONV2_INPUT_OFFSET, TIER2_CONV2_OUTPUT_OFFSET);
    if (status != PACI_OK) return status;

    paci_global_avgpool_s8(conv2_out, pooled, PACI_WINDOW_SIZE, TIER2_CONV2_N_CHANNELS,
                            TIER2_GAP_MULTIPLIER, TIER2_GAP_SHIFT, TIER2_GAP_INPUT_OFFSET, TIER2_GAP_OUTPUT_OFFSET);

    status = paci_dense_s8(pooled, dense1_out, TIER2_CONV2_N_CHANNELS, TIER2_DENSE1_N_CHANNELS,
                            tier2_dense1_weight, tier2_dense1_bias, tier2_dense1_multiplier, tier2_dense1_shift,
                            TIER2_DENSE1_INPUT_OFFSET, TIER2_DENSE1_OUTPUT_OFFSET, -128, 127);
    if (status != PACI_OK) return status;

    // Final logits layer has no activation clamp (raw accumulator range) —
    // matches the frozen architecture (section 2): Dense(5), no ReLU.
    status = paci_dense_s8(dense1_out, logits_out, TIER2_DENSE1_N_CHANNELS, TIER2_LOGITS_N_CHANNELS,
                            tier2_logits_weight, tier2_logits_bias, tier2_logits_multiplier, tier2_logits_shift,
                            TIER2_LOGITS_INPUT_OFFSET, TIER2_LOGITS_OUTPUT_OFFSET, -128, 127);
    if (status != PACI_OK) return status;

    paci_top2_margin(logits_out, TIER2_LOGITS_N_CHANNELS, class_id, margin);
    return PACI_OK;
}

#else  // !PACI_HAVE_TIER2

paci_status_t paci_infer_t2_s8(const int8_t window[PACI_WINDOW_SIZE], int8_t *class_id, int32_t *margin) {
    (void)window;
    if (class_id == NULL || margin == NULL) {
        return PACI_E_NULL;
    }
    *class_id = -1;
    *margin = 0;
    return PACI_E_QUANT;
}

#endif  // PACI_HAVE_TIER2

#if PACI_HAVE_TIER1

// tier1_conv2's weights are packed 4-bit, per-CHANNEL quantized
// (arm_convolve_s4 takes cmsis_nn_per_channel_quant_params, confirmed by
// reading Include/arm_nnfunctions.h directly rather than assumed).
static paci_status_t paci_conv1d_s4(const int8_t *input, int8_t *output,
                                     int32_t in_w, int32_t in_c, int32_t out_c, int32_t k,
                                     const int8_t *packed_weight, const int32_t *bias,
                                     const int32_t *multiplier, const int32_t *shift,
                                     int32_t input_offset, int32_t output_offset) {
    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    conv_params.input_offset = input_offset;
    conv_params.output_offset = output_offset;
    conv_params.stride.w = 1;
    conv_params.stride.h = 1;
    conv_params.padding.w = (k - 1) / 2;
    conv_params.padding.h = 0;
    conv_params.dilation.w = 1;
    conv_params.dilation.h = 1;
    conv_params.activation.min = -128;
    conv_params.activation.max = 127;

    quant_params.multiplier = (int32_t *)multiplier;
    quant_params.shift = (int32_t *)shift;

    input_dims.n = 1; input_dims.h = 1; input_dims.w = in_w; input_dims.c = in_c;
    filter_dims.n = out_c; filter_dims.h = 1; filter_dims.w = k; filter_dims.c = in_c;
    bias_dims.n = 1; bias_dims.h = 1; bias_dims.w = 1; bias_dims.c = out_c;
    output_dims.n = 1; output_dims.h = 1; output_dims.w = in_w; output_dims.c = out_c;

    int32_t required = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    if (required > (int32_t)PACI_SCRATCH_BYTES) {
        return PACI_E_BUFSIZE;
    }
    ctx.buf = paci_scratch;
    ctx.size = PACI_SCRATCH_BYTES;

    arm_cmsis_nn_status status = arm_convolve_s4(
        &ctx, &conv_params, &quant_params,
        &input_dims, input, &filter_dims, packed_weight, &bias_dims, bias, &output_dims, output);

    return (status == ARM_CMSIS_NN_SUCCESS) ? PACI_OK : PACI_E_QUANT;
}

// tier1_logits' weights are packed 4-bit, PER-TENSOR quantized: there is no
// per-channel INT4 fully-connected kernel in this CMSIS-NN checkout
// (arm_fully_connected_s4 takes cmsis_nn_per_tensor_quant_params, a single
// multiplier/shift — confirmed by reading the header, not assumed; see
// tools/export_cmsisnn.py's docstring and docs/STATUS.md).
static paci_status_t paci_dense_s4(const int8_t *input, int8_t *output,
                                    int32_t in_c, int32_t out_c,
                                    const int8_t *packed_weight, const int32_t *bias,
                                    int32_t multiplier, int32_t shift,
                                    int32_t input_offset, int32_t output_offset) {
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    fc_params.input_offset = input_offset;
    fc_params.filter_offset = 0;
    fc_params.output_offset = output_offset;
    fc_params.activation.min = -128;
    fc_params.activation.max = 127;

    quant_params.multiplier = multiplier;
    quant_params.shift = shift;

    input_dims.n = 1; input_dims.h = 1; input_dims.w = 1; input_dims.c = in_c;
    filter_dims.n = in_c; filter_dims.h = 1; filter_dims.w = 1; filter_dims.c = out_c;
    bias_dims.n = 1; bias_dims.h = 1; bias_dims.w = 1; bias_dims.c = out_c;
    output_dims.n = 1; output_dims.h = 1; output_dims.w = 1; output_dims.c = out_c;

    int32_t required = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    if (required > (int32_t)PACI_SCRATCH_BYTES) {
        return PACI_E_BUFSIZE;
    }
    ctx.buf = paci_scratch;
    ctx.size = PACI_SCRATCH_BYTES;

    arm_cmsis_nn_status status = arm_fully_connected_s4(
        &ctx, &fc_params, &quant_params,
        &input_dims, input, &filter_dims, packed_weight, &bias_dims, bias, &output_dims, output);

    return (status == ARM_CMSIS_NN_SUCCESS) ? PACI_OK : PACI_E_QUANT;
}

paci_status_t paci_infer_t1_s4(const int8_t window[PACI_WINDOW_SIZE], int8_t *class_id, int32_t *margin) {
    if (window == NULL || class_id == NULL || margin == NULL) {
        return PACI_E_NULL;
    }

    static int8_t conv1_out[PACI_WINDOW_SIZE * TIER1_CONV1_N_CHANNELS];
    static int8_t conv2_out[PACI_WINDOW_SIZE * TIER1_CONV2_N_CHANNELS];
    static int8_t pooled[TIER1_CONV2_N_CHANNELS];
    static int8_t logits_out[2];  // Tier-1 is always a 2-class screen (section 4)

    paci_status_t status;

    status = paci_conv1d_s8(window, conv1_out, PACI_WINDOW_SIZE, 1, TIER1_CONV1_N_CHANNELS, 5,
                             tier1_conv1_weight, tier1_conv1_bias, tier1_conv1_multiplier, tier1_conv1_shift,
                             TIER1_CONV1_INPUT_OFFSET, TIER1_CONV1_OUTPUT_OFFSET);
    if (status != PACI_OK) return status;

    status = paci_conv1d_s4(conv1_out, conv2_out, PACI_WINDOW_SIZE, TIER1_CONV1_N_CHANNELS, TIER1_CONV2_N_CHANNELS, 3,
                             tier1_conv2_weight, tier1_conv2_bias, tier1_conv2_multiplier, tier1_conv2_shift,
                             TIER1_CONV2_INPUT_OFFSET, TIER1_CONV2_OUTPUT_OFFSET);
    if (status != PACI_OK) return status;

    paci_global_avgpool_s8(conv2_out, pooled, PACI_WINDOW_SIZE, TIER1_CONV2_N_CHANNELS,
                            TIER1_GAP_MULTIPLIER, TIER1_GAP_SHIFT, TIER1_GAP_INPUT_OFFSET, TIER1_GAP_OUTPUT_OFFSET);

    // No trailing softmax and no activation clamp beyond the int8 range —
    // matches the frozen architecture (section 2): Dense(2), raw logits.
    status = paci_dense_s4(pooled, logits_out, TIER1_CONV2_N_CHANNELS, 2,
                            tier1_logits_weight, tier1_logits_bias, TIER1_LOGITS_MULTIPLIER, TIER1_LOGITS_SHIFT,
                            TIER1_LOGITS_INPUT_OFFSET, TIER1_LOGITS_OUTPUT_OFFSET);
    if (status != PACI_OK) return status;

    paci_top2_margin(logits_out, 2, class_id, margin);
    return PACI_OK;
}

#else  // !PACI_HAVE_TIER1

// ASSUMPTION: reports PACI_E_QUANT ("missing export"), the same "not built"
// signal PACI_HAVE_TIER2's #else branch uses, rather than a fake success
// path — a fake PACI_OK here would be indistinguishable from a real result
// to any caller. Run tools/export_cmsisnn.py --prefix tier1 first.
paci_status_t paci_infer_t1_s4(const int8_t window[PACI_WINDOW_SIZE], int8_t *class_id, int32_t *margin) {
    (void)window;
    if (class_id == NULL || margin == NULL) {
        return PACI_E_NULL;
    }
    *class_id = -1;
    *margin = 0;
    return PACI_E_QUANT;
}

#endif  // PACI_HAVE_TIER1
