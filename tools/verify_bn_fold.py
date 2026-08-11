# SPDX-License-Identifier: Apache-2.0
"""Task 2.1 gate check: BatchNorm can only be folded into the preceding
convolution by the TFLite converter when it sits BEFORE the activation
(Conv1D(activation=None) -> BatchNormalization() -> ReLU()). If it's folded
correctly, the converted graph never has a separate MUL/ADD op immediately
after a CONV_2D op (the scale/shift either vanishes into the conv's
weights/bias, or activation is fused directly into the conv op's
fused_activation_function field) — CMSIS-NN has no kernel for a bare
mul/add chain, so this must hold for both Tier-1 and Tier-2.

Usage: python tools/verify_bn_fold.py <model.tflite> [<model2.tflite> ...]
Also importable: from tools.verify_bn_fold import assert_bn_folded
"""
import sys

from tensorflow.lite.python import schema_py_generated as schema_fb

_BUILTIN_NAME_BY_CODE = {
    getattr(schema_fb.BuiltinOperator, name): name
    for name in dir(schema_fb.BuiltinOperator)
    if not name.startswith("_")
}

_DISALLOWED_AFTER_CONV = {"ADD", "MUL"}


def _op_name(model, op):
    opcode = model.OperatorCodes(op.OpcodeIndex())
    code = opcode.DeprecatedBuiltinCode()
    # TFLite schema >= v3 uses BuiltinCode() for opcodes that don't fit in
    # the deprecated int8 field; DeprecatedBuiltinCode() saturates at 127.
    if code == 127 and opcode.BuiltinCode() is not None:
        code = opcode.BuiltinCode()
    return _BUILTIN_NAME_BY_CODE.get(code, f"UNKNOWN({code})")


def op_sequence(tflite_path):
    with open(tflite_path, "rb") as f:
        buf = f.read()
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)
    return [_op_name(model, subgraph.Operators(i)) for i in range(subgraph.OperatorsLength())]


def assert_bn_folded(tflite_path):
    ops = op_sequence(tflite_path)
    for i, name in enumerate(ops[:-1]):
        if name == "CONV_2D" and ops[i + 1] in _DISALLOWED_AFTER_CONV:
            raise AssertionError(
                f"{tflite_path}: unfused BatchNorm detected — CONV_2D at op index {i} "
                f"is immediately followed by {ops[i + 1]}, not folded into the conv's "
                f"weights/bias or fused_activation_function. Full op sequence: {ops}"
            )
    return ops


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        ops = assert_bn_folded(path)
        print(f"{path}: OK, no unfused BatchNorm. Ops: {ops}")


if __name__ == "__main__":
    main()
