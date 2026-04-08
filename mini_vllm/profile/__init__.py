from __future__ import annotations

from importlib import import_module

__all__ = [
    "BatchSpec",
    "RequestSpec",
    "BatchSampler",
    "EnergyMeter",
    "GpuFrequencyController",
    "BatchProfiler",
    "ProfileModeler",
]


_SYMBOL_TO_MODULE = {
    "BatchSpec": ".batch_sampler",
    "RequestSpec": ".batch_sampler",
    "BatchSampler": ".batch_sampler",
    "EnergyMeter": ".energy_meter",
    "GpuFrequencyController": ".energy_meter",
    "BatchProfiler": ".profiler",
    "ProfileModeler": ".modeler",
}


def __getattr__(name: str):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
