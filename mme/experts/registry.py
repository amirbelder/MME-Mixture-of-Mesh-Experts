"""Expert registry: decorator + entry-point discovery.

Two ways to register an expert:

1. Decorator (in-process):

    from mme.experts.registry import register_expert
    from mme.experts.torch_expert import TorchExpert

    @register_expert("my_expert")
    class MyExpert(TorchExpert):
        ...

2. Entry point (third-party packages, in ``pyproject.toml``):

    [project.entry-points."mme.experts"]
    my_expert = "my_package.experts:MyExpert"

Both are surfaced via ``list_experts()``. Entry points are loaded lazily on
the first call to ``list_experts()``, ``get_expert()``, or
``load_entry_point_experts()``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type

_REGISTRY: Dict[str, Type] = {}
_ENTRY_POINTS_LOADED = False


def register_expert(name: str) -> Callable[[Type], Type]:
    """Decorator that registers an Expert class under ``name``.

    Raises:
        ValueError: if the name is already registered (prevents accidental clobber).
    """

    def _decorate(cls: Type) -> Type:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"expert {name!r} is already registered ({_REGISTRY[name]!r})"
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _decorate


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover — py<3.10 fallback is same
        return
    try:
        eps = entry_points(group="mme.experts")  # py>=3.10
    except TypeError:  # py3.9
        eps = entry_points().get("mme.experts", [])
    for ep in eps:
        if ep.name in _REGISTRY:
            continue
        try:
            cls = ep.load()
        except Exception as e:
            # Skip broken plugins with a soft warning rather than crashing all callers.
            import warnings

            warnings.warn(
                f"failed to load expert entry point {ep.name!r}: {e}", stacklevel=2
            )
            continue
        cls.name = ep.name
        _REGISTRY[ep.name] = cls


def get_expert(name: str, **kwargs):
    """Instantiate a registered expert by name, forwarding ``**kwargs`` to its ctor."""
    _load_entry_points()
    if name not in _REGISTRY:
        raise KeyError(f"unknown expert {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_experts() -> List[str]:
    """Return sorted names of all registered experts (decorator + entry points)."""
    _load_entry_points()
    return sorted(_REGISTRY)


def _reset_registry_for_tests() -> None:
    """Clear the registry. Used by unit tests only."""
    global _ENTRY_POINTS_LOADED
    _REGISTRY.clear()
    _ENTRY_POINTS_LOADED = False
