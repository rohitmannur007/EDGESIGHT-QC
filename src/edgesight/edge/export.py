"""Model export and quantization with explicit capability outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OperationResult:
    supported: bool
    operation: str
    artifact: Path | None = None
    reason: str | None = None
    model: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifact"] = str(self.artifact) if self.artifact else None
        value.pop("model", None)
        return value


def _torch_or_result(operation: str) -> tuple[Any | None, OperationResult | None]:
    try:
        import torch
    except ImportError:
        return None, OperationResult(False, operation, reason="PyTorch is not installed")
    return torch, None


def export_model(
    model: Any,
    example_input: Any,
    destination: str | Path,
    *,
    format: str = "torchscript",
    verify: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> OperationResult:
    torch, unavailable = _torch_or_result(f"export:{format}")
    if unavailable:
        return unavailable
    assert torch is not None
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = model.eval()
        if format == "torchscript":
            traced = torch.jit.trace(model, example_input)
            traced.save(str(path))
        elif format == "onnx":
            try:
                import onnx  # noqa: F401
            except ImportError:
                return OperationResult(False, "export:onnx", reason="ONNX package is not installed")
            torch.onnx.export(
                model,
                example_input,
                str(path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
            )
        elif format == "coreml":
            try:
                import coremltools as ct
            except ImportError:
                return OperationResult(
                    False,
                    "export:coreml",
                    reason=(
                        "coremltools is not installed; install it explicitly on macOS "
                        "and verify the selected compute units"
                    ),
                )
            traced = torch.jit.trace(model, example_input)
            converted = ct.convert(
                traced,
                inputs=[ct.TensorType(name="input", shape=tuple(example_input.shape))],
            )
            converted.save(str(path))
        else:
            return OperationResult(
                False, f"export:{format}", reason=f"unsupported export format: {format}"
            )
        agreement: dict[str, Any] = {}
        if verify:
            with torch.inference_mode():
                expected = model(example_input)
                if format == "torchscript":
                    actual = torch.jit.load(str(path))(example_input)
                elif format == "onnx":
                    try:
                        import onnxruntime as ort
                    except ImportError:
                        return OperationResult(
                            False,
                            "export:onnx",
                            artifact=path,
                            reason=(
                                "ONNX Runtime is not installed; native/ONNX agreement not verified"
                            ),
                            metadata={"status": "NOT_SUPPORTED"},
                        )
                    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                    actual = session.run(None, {"input": example_input.detach().cpu().numpy()})[0]
                    actual = torch.as_tensor(actual)
                else:
                    actual = None
                if actual is not None:
                    expected_tensor = _first_tensor(expected, torch).detach().cpu()
                    actual_tensor = _first_tensor(actual, torch).detach().cpu()
                    maximum = float((expected_tensor - actual_tensor).abs().max())
                    agreed = bool(
                        torch.allclose(expected_tensor, actual_tensor, rtol=rtol, atol=atol)
                    )
                    agreement = {"agreement": agreed, "max_abs_error": maximum}
                    if not agreed:
                        return OperationResult(
                            False,
                            f"export:{format}",
                            artifact=path,
                            reason="exported output does not agree with native output",
                            metadata=agreement,
                        )
    except Exception as exc:
        return OperationResult(False, f"export:{format}", reason=str(exc))
    return OperationResult(
        True,
        f"export:{format}",
        artifact=path,
        metadata={"status": "OK", **agreement},
    )


def _first_tensor(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value
    if isinstance(value, tuple | list) and value:
        return _first_tensor(value[0], torch)
    if isinstance(value, dict) and value:
        return _first_tensor(next(iter(value.values())), torch)
    raise TypeError("model output does not contain a tensor suitable for agreement checking")


def quantize_model(model: Any, *, mode: str = "dynamic") -> OperationResult:
    torch, unavailable = _torch_or_result(f"quantize:{mode}")
    if unavailable:
        return unavailable
    assert torch is not None
    if mode != "dynamic":
        return OperationResult(
            False,
            f"quantize:{mode}",
            reason="only dynamic CPU quantization is implemented",
        )
    try:
        linear_count = sum(isinstance(module, torch.nn.Linear) for module in model.modules())
        if linear_count == 0:
            return OperationResult(
                False,
                "quantize:dynamic",
                reason=(
                    "NOT_SUPPORTED: dynamic int8 quantization only transforms Linear "
                    "layers; this architecture has none"
                ),
                metadata={"status": "NOT_SUPPORTED", "linear_layers": 0},
            )
        engines = list(getattr(torch.backends.quantized, "supported_engines", []))
        if not engines:
            return OperationResult(
                False,
                "quantize:dynamic",
                reason="NOT_SUPPORTED: this PyTorch build has no quantized CPU engine",
                metadata={"status": "NOT_SUPPORTED"},
            )
        quantized = torch.ao.quantization.quantize_dynamic(
            model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8
        )
    except Exception as exc:
        return OperationResult(False, "quantize:dynamic", reason=str(exc))
    return OperationResult(
        True,
        "quantize:dynamic",
        model=quantized,
        metadata={"status": "OK", "linear_layers": linear_count, "engines": engines},
    )
