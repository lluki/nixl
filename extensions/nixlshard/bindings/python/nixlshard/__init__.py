"""Native NIXLShard API.

Registered addresses must reference CPU memory kept alive by the caller until
all transfers using the registration have finished and their handles are
released. Completion is a list of per-object status names; ``poll`` returns
``None`` while a request is pending. Explicitly close agents before interpreter
shutdown, or use them as context managers.
"""
from ._bindings import Agent, MetadataServer, build_git

__build_marker__ = build_git
__all__ = ["Agent", "MetadataServer", "build_git", "__build_marker__"]
