"""Static label-leakage checker used by tests/test_no_label_leakage.py.

Builds the import graph of src/ and reports every restricted module that can reach a
label-bearing module, directly or transitively. It understands:
  import a.b.c [as x] | from a.b import c | from .. import x | from ..a import b
  importlib.import_module("...") / __import__("...") with a literal string
The spec's reference test inspected only `node.module` for ImportFrom. That misses
`from src.evaluation import sealed_store`, where node.module is "src.evaluation",
so this checker also resolves the imported names.
"""
from __future__ import annotations

import ast
import pathlib

LABEL_BEARING = ("src.evaluation", "src.data.loader", "src.pipeline")
RESTRICTED_PACKAGES = ("src.observer", "src.detection", "src.decision", "src.classifier")
RESTRICTED_MODULES = ("src.data.replay",)


def module_name(root: pathlib.Path, f: pathlib.Path) -> str:
    rel = f.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(mod: str, is_pkg: bool, level: int, target: str | None) -> str:
    base = mod.split(".") if is_pkg else mod.split(".")[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join(base + ([target] if target else []))


def imports_of(root: pathlib.Path, f: pathlib.Path) -> set[str]:
    mod = module_name(root, f)
    is_pkg = f.name == "__init__.py"
    out: set[str] = set()
    for node in ast.walk(ast.parse(f.read_text())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative(mod, is_pkg, node.level, node.module) if node.level else (node.module or "")
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names if a.name != "*")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                out.add(node.args[0].value)
    return out


def build_graph(root: pathlib.Path) -> dict[str, set[str]]:
    src = root / "src"
    return {module_name(root, f): imports_of(root, f) for f in src.rglob("*.py")}


def _is_label_bearing(name: str) -> bool:
    return any(name == lb or name.startswith(lb + ".") for lb in LABEL_BEARING) or "sealed_store" in name


def _is_restricted(name: str) -> bool:
    return name in RESTRICTED_MODULES or any(name == p or name.startswith(p + ".") for p in RESTRICTED_PACKAGES)


def violations(root: pathlib.Path) -> list[str]:
    g = build_graph(root)
    bad = []
    for mod in sorted(g):
        if not _is_restricted(mod):
            continue
        seen, stack, path = set(), [(mod, [mod])], None
        while stack:
            cur, trail = stack.pop()
            for imp in g.get(cur, ()):
                if _is_label_bearing(imp):
                    bad.append(" -> ".join(trail + [imp]))
                    continue
                # follow imports of modules inside src (also parents of submodules)
                for cand in (imp, imp.rsplit(".", 1)[0]):
                    if cand in g and cand not in seen:
                        seen.add(cand)
                        stack.append((cand, trail + [cand]))
    return bad
