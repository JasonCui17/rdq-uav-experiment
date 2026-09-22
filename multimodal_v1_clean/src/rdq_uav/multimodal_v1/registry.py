"""Small explicit component registry for Multimodal V1 YAML construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Map stable configuration names to constructors.

    Duplicate names are rejected so importing a module cannot silently replace
    the implementation selected by an experiment configuration.
    """

    def __init__(self, label: str = "component") -> None:
        self.label = label
        self._constructors: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[T], T]:
        if not name:
            raise ValueError("registry name must be non-empty")

        def decorator(constructor: T) -> T:
            if name in self._constructors:
                raise KeyError(f"duplicate {self.label} registration: {name}")
            if not callable(constructor):
                raise TypeError(f"registered {self.label} must be callable: {name}")
            self._constructors[name] = constructor  # type: ignore[assignment]
            return constructor

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._constructors[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._constructors)) or "<none>"
            raise KeyError(f"unknown {self.label} {name!r}; available: {choices}") from exc

    def build(self, spec: str | Mapping[str, Any], **overrides: Any) -> Any:
        if isinstance(spec, str):
            name, kwargs = spec, {}
        elif isinstance(spec, Mapping):
            if "name" not in spec:
                raise KeyError(f"{self.label} spec requires a 'name' field")
            name = str(spec["name"])
            kwargs = {key: value for key, value in spec.items() if key != "name"}
        else:
            raise TypeError(f"{self.label} spec must be a name or mapping")
        kwargs.update(overrides)
        return self.get(name)(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._constructors))


COMPONENTS = Registry("multimodal component")
