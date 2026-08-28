"""Catch attributes a class reads but never initialises.

This exists because a run of string-based edits left IBKRPollingSource
reading `self._lag_warned` while its constructor set different names. The
result crashed only once a bar arrived — minutes into a live session, after
warmup had already succeeded.

Static analysis finds it in a second instead.

Usage:
    python scripts/check_attributes.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Attributes legitimately set outside __init__ or inherited from a base class.
ALLOWED: dict[str, set[str]] = {
    "PaperBroker": {"_entry_fills", "_entry_commission", "params"},
    "IBKRBroker": {"_entry_fills", "_entry_commission", "params",
                   "limit_offset_bps", "fill_timeout", "use_limit_orders"},
    "Engine": {"broker", "source", "model", "config", "state"},
}


def _self_targets(targets) -> set[str]:
    """Attribute names assigned on self, unpacking tuple targets.

    `self._url, self._key = url, key` is a Tuple target and is easy to miss.
    """
    names: set[str] = set()
    stack = list(targets)
    while stack:
        target = stack.pop()
        if isinstance(target, (ast.Tuple, ast.List)):
            stack.extend(target.elts)
        elif (isinstance(target, ast.Attribute)
              and isinstance(target.value, ast.Name)
              and target.value.id == "self"):
            names.add(target.attr)
    return names


def assigned_names(cls: ast.ClassDef) -> set[str]:
    """Every `self.x = ...` anywhere in the class, plus class-level names."""
    names: set[str] = set()
    for node in ast.walk(cls):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            names.update(_self_targets(targets))
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(stmt.name)
    return names


def class_level_names(cls: ast.ClassDef) -> set[str]:
    """Methods and class variables. These are never set in __init__."""
    names: set[str] = set()
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            names.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
    return names


def init_names(cls: ast.ClassDef) -> set[str]:
    """Attributes assigned specifically in __init__."""
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
            names = set()
            for node in ast.walk(stmt):
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = (node.targets if isinstance(node, ast.Assign)
                               else [node.target])
                    names.update(_self_targets(targets))
            return names
    return set()


def read_names(cls: ast.ClassDef) -> dict[str, int]:
    """Every `self.x` read, with the line it appears on."""
    reads: dict[str, int] = {}
    for node in ast.walk(cls):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            reads.setdefault(node.attr, node.lineno)
    return reads


def check(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    classes = {n.name: n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef)}
    problems = []
    for cls in classes.values():
        assigned = assigned_names(cls)
        # A subclass legitimately uses attributes its base sets, so walk the
        # inheritance chain within this module.
        pending = [b.id for b in cls.bases if isinstance(b, ast.Name)]
        seen = set()
        while pending:
            name = pending.pop()
            if name in seen or name not in classes:
                continue
            seen.add(name)
            base = classes[name]
            assigned |= assigned_names(base)
            pending.extend(b.id for b in base.bases if isinstance(b, ast.Name))
        initialised = init_names(cls)
        allowed = ALLOWED.get(cls.name, set())
        bases = {b.id for b in cls.bases if isinstance(b, ast.Name)}
        inherited_init = set()
        class_level = class_level_names(cls)
        for name in seen:
            inherited_init |= init_names(classes[name])
            class_level |= class_level_names(classes[name])

        for attr, line in read_names(cls).items():
            # Methods and class constants are resolved on the class, not the
            # instance, so the constructor rule does not apply to them.
            if attr in allowed or attr in class_level or attr.startswith("__"):
                continue
            if attr not in assigned:
                if bases and not attr.startswith("_"):
                    continue    # probably inherited from outside this module
                problems.append(
                    f"{path.relative_to(ROOT)}:{line}  {cls.name}.{attr} is "
                    f"read but never assigned")
            elif (initialised or inherited_init) and \
                    attr not in initialised and attr not in inherited_init:
                # Assigned somewhere, but not in any constructor. Reading it
                # before that assignment runs raises AttributeError at
                # runtime — exactly the IBKRPollingSource._lag_warned crash,
                # which only surfaced once a bar arrived.
                problems.append(
                    f"{path.relative_to(ROOT)}:{line}  {cls.name}.{attr} is "
                    f"read but only assigned outside __init__ "
                    f"(set it in the constructor)")
    return problems


def main() -> None:
    targets = sorted((ROOT / "engine").glob("*.py"))
    all_problems: list[str] = []
    for path in targets:
        found = check(path)
        status = "FAIL" if found else "ok"
        print(f"  {status:<5} {path.relative_to(ROOT)}")
        all_problems.extend(found)

    if all_problems:
        print(f"\n{len(all_problems)} problem(s):")
        for p in all_problems:
            print(f"  {p}")
        sys.exit(1)
    print("\nNo uninitialised attribute reads.")


if __name__ == "__main__":
    main()
