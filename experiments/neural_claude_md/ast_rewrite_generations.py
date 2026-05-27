from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

from experiments.neural_claude_md.common import DEFAULT_OUTPUT_DIR, extract_code_block, read_jsonl, write_jsonl


class PrintToLoggerTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.replaced = 0

    def visit_Call(self, node: ast.Call) -> Any:
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.replaced += 1
            return ast.copy_location(
                ast.Call(
                    func=ast.Attribute(value=ast.Name(id="logger", ctx=ast.Load()), attr="info", ctx=ast.Load()),
                    args=self._logger_args(node),
                    keywords=[],
                ),
                node,
            )
        return node

    def _logger_args(self, node: ast.Call) -> list[ast.expr]:
        if not node.args:
            return [ast.Constant(value="")]
        if len(node.args) == 1:
            return node.args
        tuple_expr = ast.Tuple(elts=list(node.args), ctx=ast.Load())
        return [
            ast.Call(
                func=ast.Attribute(
                    value=ast.Constant(value=" "),
                    attr="join",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Call(
                        func=ast.Name(id="map", ctx=ast.Load()),
                        args=[ast.Name(id="str", ctx=ast.Load()), tuple_expr],
                        keywords=[],
                    )
                ],
                keywords=[],
            )
        ]


def has_import_logging(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "logging" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module == "logging":
            return True
    return False


def has_logger_binding(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "logger" for target in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "logger":
            return True
    return False


def insertion_index_after_imports(tree: ast.Module) -> int:
    idx = 0
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        if isinstance(tree.body[0].value.value, str):
            idx = 1
    while idx < len(tree.body) and isinstance(tree.body[idx], (ast.Import, ast.ImportFrom)):
        idx += 1
    return idx


def rewrite_code(code: str) -> tuple[str, dict[str, Any]]:
    meta = {
        "rewrite_attempted": False,
        "rewrite_applied": False,
        "rewrite_parse_ok": False,
        "replaced_print_calls": 0,
        "added_import_logging": False,
        "added_logger_binding": False,
        "rewrite_error": None,
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        meta["rewrite_error"] = str(exc)
        return code, meta

    meta["rewrite_attempted"] = True
    meta["rewrite_parse_ok"] = True
    transformer = PrintToLoggerTransformer()
    tree = transformer.visit(tree)
    assert isinstance(tree, ast.Module)
    meta["replaced_print_calls"] = transformer.replaced

    if transformer.replaced == 0:
        return code, meta

    insert_at = insertion_index_after_imports(tree)
    if not has_import_logging(tree):
        tree.body.insert(insert_at, ast.Import(names=[ast.alias(name="logging")]))
        insert_at += 1
        meta["added_import_logging"] = True
    if not has_logger_binding(tree):
        logger_assign = ast.Assign(
            targets=[ast.Name(id="logger", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="logging", ctx=ast.Load()), attr="getLogger", ctx=ast.Load()),
                args=[ast.Name(id="__name__", ctx=ast.Load())],
                keywords=[],
            ),
        )
        tree.body.insert(insert_at, logger_assign)
        meta["added_logger_binding"] = True

    ast.fix_missing_locations(tree)
    rewritten = ast.unparse(tree)
    meta["rewrite_applied"] = True
    return rewritten, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default=str(DEFAULT_OUTPUT_DIR / "generations" / "generations_gated_test.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "generations" / "generations_ast_rewrite.jsonl"))
    parser.add_argument("--source-condition", default=None)
    parser.add_argument("--condition-name", default="ast_rewrite")
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(args.generations):
        if args.source_condition and row.get("condition") != args.source_condition:
            continue
        code = extract_code_block(row["generation"])
        rewritten, meta = rewrite_code(code)
        rows.append(
            {
                **row,
                "source_condition": row.get("condition"),
                "condition": args.condition_name if args.source_condition else f"{args.condition_name}_{row.get('condition')}",
                "generation": f"```python\n{rewritten}\n```",
                **meta,
            }
        )

    write_jsonl(args.output, rows)
    print(f"Saved {len(rows)} rewritten generations to {args.output}")


if __name__ == "__main__":
    main()
