#!/usr/bin/env python3
"""Reject G1 Arm SDK authority-release paths outside the central handover."""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "src/lerobot/robots/unitree_g1/g1_arm_sdk.py"
REFERENCE_CONTROLLER = ROOT / "src/lerobot/robots/unitree_g1/g1_arm_unitree.py"
SEARCH_ROOTS = (
    ROOT / "src/lerobot/robots/unitree_g1",
    ROOT / "examples/unitree_g1",
)
ALLOWED_ZERO_WEIGHT_FUNCTIONS = {"activate", "return_to_stock"}


def _is_zero(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        node = node.operand
    return isinstance(node, ast.Constant) and not isinstance(node.value, bool) and node.value == 0


class HandoverVisitor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.functions: list[str] = []
        self.errors: list[str] = []

    @property
    def function(self) -> str | None:
        return self.functions[-1] if self.functions else None

    def fail(self, node: ast.AST, message: str) -> None:
        self.errors.append(f"{self.path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}: {message}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "return_to_stock" and self.path != CONTROLLER:
            self.fail(node, "return_to_stock must have exactly one central implementation")
        if node.name == "_release_authority":
            self.fail(node, "direct authority-release helpers are forbidden; use return_to_stock")
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name == "return_to_stock" and self.path != CONTROLLER:
            self.fail(node, "return_to_stock must have exactly one central implementation")
        if node.name == "_release_authority":
            self.fail(node, "direct authority-release helpers are forbidden; use return_to_stock")
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.id if isinstance(node.func, ast.Name) else None
        allowed_caller = (
            self.path == CONTROLLER and self.function in ALLOWED_ZERO_WEIGHT_FUNCTIONS | {"_step"}
        ) or (self.path == REFERENCE_CONTROLLER and self.function == "_step")
        if name == "arm_command" and not allowed_caller:
            self.fail(node, "Arm SDK publishing is only allowed in activate, _step, and return_to_stock")
        weight = (
            node.args[2]
            if len(node.args) >= 3
            else next((keyword.value for keyword in node.keywords if keyword.arg == "weight"), None)
        )
        if (
            name == "arm_command"
            and weight is not None
            and _is_zero(weight)
            and (self.path != CONTROLLER or self.function not in ALLOWED_ZERO_WEIGHT_FUNCTIONS)
        ):
            self.fail(node, "zero Arm SDK weight is only allowed in the central handover")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_release_authority":
            self.fail(node, "direct authority release is forbidden; use return_to_stock")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "release_weight":
                self.fail(node, "spoke release weights are forbidden; use return_to_stock")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value == "rt/arm_sdk" and self.path != CONTROLLER:
            self.fail(node, "rt/arm_sdk publishing is centralized in g1_arm_sdk.py")


def main() -> int:
    errors: list[str] = []
    paths = sorted(path for root in SEARCH_ROOTS if root.exists() for path in root.rglob("*.py"))
    for path in paths:
        visitor = HandoverVisitor(path)
        visitor.visit(ast.parse(path.read_text(), filename=str(path)))
        errors.extend(visitor.errors)
    if errors:
        print("Unsafe G1 Arm SDK handover architecture:", file=sys.stderr)
        print("\n".join(f"  {error}" for error in errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
