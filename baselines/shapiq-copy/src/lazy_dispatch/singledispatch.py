from __future__ import annotations

from functools import update_wrapper
from typing import Any, Callable, TypeAlias


LazyType: TypeAlias = type | str | tuple[type | str, ...]


def _qualified_names(cls: type) -> set[str]:
    return {f"{base.__module__}.{base.__qualname__}" for base in cls.__mro__}


class _LazyDispatcher:
    def __init__(self, default: Callable) -> None:
        self.default = default
        self._type_registry: list[tuple[type, Callable]] = []
        self._name_registry: list[tuple[str, Callable]] = []
        self._delayed: list[tuple[tuple[type | str, ...], Callable]] = []
        self._loading: set[Callable] = set()
        update_wrapper(self, default)

    @staticmethod
    def _items(cls: LazyType) -> tuple[type | str, ...]:
        return cls if isinstance(cls, tuple) else (cls,)

    def register(self, cls: LazyType = object, func: Callable | None = None):
        def decorator(callback: Callable) -> Callable:
            for item in self._items(cls):
                if isinstance(item, str):
                    self._name_registry.append((item, callback))
                elif isinstance(item, type):
                    self._type_registry.append((item, callback))
                else:
                    raise TypeError(f"Unsupported lazy dispatch key: {item!r}")
            return callback

        return decorator if func is None else decorator(func)

    def delayed_register(self, cls: LazyType):
        def decorator(loader: Callable) -> Callable:
            self._delayed.append((self._items(cls), loader))
            return loader

        return decorator

    @staticmethod
    def _matches(item: type | str, value: object, names: set[str]) -> bool:
        if isinstance(item, str):
            return item in names
        return isinstance(value, item)

    def _resolve(self, value: object) -> Callable | None:
        names = _qualified_names(type(value))
        for cls, callback in reversed(self._type_registry):
            if isinstance(value, cls):
                return callback
        for name, callback in reversed(self._name_registry):
            if name in names:
                return callback
        for keys, loader in list(self._delayed):
            if loader in self._loading:
                continue
            if any(self._matches(item, value, names) for item in keys):
                self._loading.add(loader)
                try:
                    loader(type(value))
                finally:
                    self._loading.remove(loader)
                    self._delayed = [entry for entry in self._delayed if entry[1] is not loader]
                return self._resolve(value)
        return None

    def __call__(self, *args: Any, **kwargs: Any):
        if not args:
            return self.default(*args, **kwargs)
        callback = self._resolve(args[0]) or self.default
        return callback(*args, **kwargs)


def lazydispatch(function: Callable) -> _LazyDispatcher:
    return _LazyDispatcher(function)
