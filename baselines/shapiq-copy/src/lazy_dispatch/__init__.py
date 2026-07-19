"""Small local fallback for the lazy-dispatch API used by this shapiq checkout.

The upstream dependency is optional in this repository snapshot.  This module
implements only the public surface used by shapiq: eager registrations by type
or qualified type name, plus delayed import callbacks.
"""

from .singledispatch import LazyType, lazydispatch

__all__ = ["LazyType", "lazydispatch"]
