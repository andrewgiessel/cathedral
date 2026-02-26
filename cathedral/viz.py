"""Trace visualization for Cathedral.

Provides text-based and graphviz-based visualization of individual traces
and posterior-level analysis of trace collections.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from cathedral.trace import Choice, Trace

if TYPE_CHECKING:
    from cathedral.model import Posterior


def print_trace(trace: Trace, *, show_dist: bool = False, show_log_prob: bool = True) -> None:
    """Print a trace as an indented tree using scope_path grouping.

    Args:
        trace: The trace to visualize.
        show_dist: If True, show the distribution for each choice.
        show_log_prob: If True, show log-probabilities.
    """
    print(format_trace(trace, show_dist=show_dist, show_log_prob=show_log_prob))


def format_trace(trace: Trace, *, show_dist: bool = False, show_log_prob: bool = True) -> str:
    """Format a trace as an indented tree string using scope_path grouping.

    Args:
        trace: The trace to visualize.
        show_dist: If True, show the distribution for each choice.
        show_log_prob: If True, show log-probabilities.

    Returns:
        A formatted string representation.
    """
    tree = _build_scope_tree(trace)
    lines: list[str] = []

    header = f"Trace (result={trace.result!r}, log_joint={trace.log_joint:.4f}"
    if trace.log_score != 0.0:
        header += f", log_score={trace.log_score:.4f}"
    header += f", {len(trace)} choices)"
    lines.append(header)

    _render_tree(tree, lines, prefix="", show_dist=show_dist, show_log_prob=show_log_prob)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal tree construction
# ---------------------------------------------------------------------------

_SCOPE_NODE = "__scope__"


def _build_scope_tree(trace: Trace) -> list[tuple[str, Any]]:
    """Build a nested tree structure from choices grouped by scope_path.

    Returns a list of items, where each item is either:
    - ("choice", Choice) for a leaf choice
    - ("scope", name, children) for a scope group
    """
    root: list[tuple[str, Any]] = []
    scope_map: dict[tuple[str, ...], list] = {}
    scope_map[()] = root

    for choice in trace.choices.values():
        path = choice.scope_path
        _ensure_scope_path(scope_map, path)
        scope_map[path].append(("choice", choice))

    return root


def _ensure_scope_path(scope_map: dict[tuple[str, ...], list], path: tuple[str, ...]) -> None:
    """Ensure all ancestor scope nodes exist in the tree."""
    for depth in range(1, len(path) + 1):
        prefix = path[:depth]
        if prefix not in scope_map:
            parent = path[: depth - 1]
            children: list = []
            scope_map[prefix] = children
            scope_map[parent].append(("scope", prefix[-1], children))


def _render_tree(
    items: list,
    lines: list[str],
    prefix: str,
    show_dist: bool,
    show_log_prob: bool,
) -> None:
    """Recursively render tree items with box-drawing connectors."""
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        child_prefix = prefix + ("    " if is_last else "\u2502   ")

        if item[0] == "choice":
            choice: Choice = item[1]
            line = f"{prefix}{connector}{choice.address} = {choice.value!r}"
            if show_log_prob:
                line += f"  ({choice.log_prob:.4f})"
            if show_dist:
                line += f"  {choice.distribution}"
            lines.append(line)

        elif item[0] == "scope":
            scope_name: str = item[1]
            children: list = item[2]
            lines.append(f"{prefix}{connector}[{scope_name}]")
            _render_tree(children, lines, child_prefix, show_dist, show_log_prob)


# ---------------------------------------------------------------------------
# Posterior-level analysis
# ---------------------------------------------------------------------------


def structure_summary(posterior: Posterior) -> str:
    """Summarize the structural variation across traces in a posterior.

    Reports how many distinct address sets exist, which addresses are
    always present, and which appear only in some traces.

    Args:
        posterior: A Posterior object from inference.

    Returns:
        A formatted string summary.
    """
    traces = posterior.traces
    address_sets = [frozenset(t.choices.keys()) for t in traces]
    unique_structures = Counter(address_sets)

    all_addresses: set[str] = set()
    for addr_set in address_sets:
        all_addresses |= addr_set

    n = len(traces)
    always_present = {a for a in all_addresses if sum(1 for s in address_sets if a in s) == n}
    sometimes_present = all_addresses - always_present

    lines: list[str] = []
    lines.append(f"Structure Summary ({n} traces)")
    lines.append("=" * 50)
    lines.append(f"  Distinct structures: {len(unique_structures)}")
    lines.append(f"  Total unique addresses: {len(all_addresses)}")

    if always_present:
        lines.append(f"  Always present ({len(always_present)}):")
        for addr in sorted(always_present):
            lines.append(f"    - {addr}")

    if sometimes_present:
        lines.append(f"  Variable ({len(sometimes_present)}):")
        for addr in sorted(sometimes_present):
            count = sum(1 for s in address_sets if addr in s)
            lines.append(f"    - {addr}  ({count}/{n} traces, {count/n:.1%})")

    if len(unique_structures) > 1:
        lines.append(f"  Structure distribution:")
        for struct, count in unique_structures.most_common(10):
            addrs = sorted(struct)
            lines.append(f"    {count}/{n} ({count/n:.1%}): {{{', '.join(addrs)}}}")
        if len(unique_structures) > 10:
            lines.append(f"    ... and {len(unique_structures) - 10} more")

    return "\n".join(lines)


def address_frequency(posterior: Posterior) -> dict[str, float]:
    """Compute the fraction of traces containing each address.

    Args:
        posterior: A Posterior object from inference.

    Returns:
        Dict mapping address names to their frequency (0.0 to 1.0).
    """
    traces = posterior.traces
    n = len(traces)
    if n == 0:
        return {}

    counts: Counter[str] = Counter()
    for t in traces:
        for addr in t.choices:
            counts[addr] += 1

    return {addr: count / n for addr, count in counts.most_common()}


def compare_traces(trace_a: Trace, trace_b: Trace) -> str:
    """Show the diff between two traces: shared, appeared, and disappeared sites.

    Args:
        trace_a: The "before" trace.
        trace_b: The "after" trace.

    Returns:
        A formatted string showing the structural diff.
    """
    addrs_a = set(trace_a.choices.keys())
    addrs_b = set(trace_b.choices.keys())

    shared = addrs_a & addrs_b
    appeared = addrs_b - addrs_a
    disappeared = addrs_a - addrs_b

    lines: list[str] = []
    lines.append(f"Trace Comparison (A: {len(trace_a)} choices, B: {len(trace_b)} choices)")
    lines.append("=" * 50)

    if shared:
        lines.append(f"  Shared ({len(shared)}):")
        for addr in sorted(shared):
            va = trace_a.choices[addr].value
            vb = trace_b.choices[addr].value
            marker = " *" if va != vb else ""
            lines.append(f"    {addr}: {va!r} -> {vb!r}{marker}")

    if appeared:
        lines.append(f"  Appeared in B ({len(appeared)}):")
        for addr in sorted(appeared):
            lines.append(f"    + {addr} = {trace_b.choices[addr].value!r}")

    if disappeared:
        lines.append(f"  Disappeared from A ({len(disappeared)}):")
        for addr in sorted(disappeared):
            lines.append(f"    - {addr} = {trace_a.choices[addr].value!r}")

    if not appeared and not disappeared:
        changed = sum(1 for a in shared if trace_a.choices[a].value != trace_b.choices[a].value)
        lines.append(f"  Same structure, {changed}/{len(shared)} values differ")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graphviz DOT output
# ---------------------------------------------------------------------------


def trace_to_dot(
    trace: Trace,
    *,
    show_dist: bool = False,
    show_log_prob: bool = True,
    label: str | None = None,
) -> str:
    """Generate a Graphviz DOT representation of a trace.

    Uses scope_path to create a hierarchical subgraph structure.
    Each choice is a node, grouped by scope.

    Args:
        trace: The trace to visualize.
        show_dist: If True, show the distribution in each node.
        show_log_prob: If True, show log-probabilities in each node.
        label: Optional graph label.

    Returns:
        A DOT-language string that can be rendered with Graphviz.
    """
    lines: list[str] = []
    graph_label = label or f"Trace (result={trace.result!r}, log_joint={trace.log_joint:.4f})"
    lines.append("digraph trace {")
    lines.append(f'  label="{_dot_escape(graph_label)}";')
    lines.append("  labelloc=t;")
    lines.append("  fontsize=14;")
    lines.append("  node [shape=record, style=filled, fillcolor=lightblue, fontsize=10];")
    lines.append("  edge [arrowsize=0.7];")
    lines.append("")

    scope_children: dict[tuple[str, ...], list] = {}
    scope_children[()] = []

    for choice in trace.choices.values():
        path = choice.scope_path
        _ensure_scope_path_dot(scope_children, path)
        scope_children[path].append(choice)

    _render_dot_scope(lines, (), scope_children, show_dist, show_log_prob, depth=1)

    prev_id: str | None = None
    for choice in trace.choices.values():
        node_id = _dot_node_id(choice.address)
        if prev_id is not None:
            lines.append(f"  {prev_id} -> {node_id} [style=dashed, color=gray];")
        prev_id = node_id

    lines.append("}")
    return "\n".join(lines)


def _ensure_scope_path_dot(scope_children: dict[tuple[str, ...], list], path: tuple[str, ...]) -> None:
    """Ensure all ancestor scopes exist in the scope_children map."""
    for depth in range(1, len(path) + 1):
        prefix = path[:depth]
        if prefix not in scope_children:
            scope_children[prefix] = []


def _render_dot_scope(
    lines: list[str],
    scope_path: tuple[str, ...],
    scope_children: dict[tuple[str, ...], list],
    show_dist: bool,
    show_log_prob: bool,
    depth: int,
) -> None:
    """Recursively render scope subgraphs and choice nodes."""
    indent = "  " * depth
    choices = scope_children.get(scope_path, [])

    child_scopes = [
        sp for sp in scope_children
        if len(sp) == len(scope_path) + 1 and sp[:len(scope_path)] == scope_path
    ]

    if scope_path:
        cluster_id = "_".join(scope_path).replace(" ", "_").replace("(", "").replace(")", "")
        lines.append(f"{indent}subgraph cluster_{cluster_id} {{")
        lines.append(f'{indent}  label="{_dot_escape(scope_path[-1])}";')
        lines.append(f"{indent}  style=rounded;")
        lines.append(f"{indent}  color=gray;")
        depth += 1
        indent = "  " * depth

    for choice in choices:
        node_id = _dot_node_id(choice.address)
        label_parts = [f"{choice.address} = {choice.value!r}"]
        if show_log_prob:
            label_parts.append(f"lp={choice.log_prob:.4f}")
        if show_dist:
            label_parts.append(str(choice.distribution))
        node_label = " | ".join(label_parts)
        lines.append(f'{indent}{node_id} [label="{_dot_escape(node_label)}"];')

    for child_scope in sorted(child_scopes):
        _render_dot_scope(lines, child_scope, scope_children, show_dist, show_log_prob, depth)

    if scope_path:
        lines.append(f"{'  ' * (depth - 1)}}}")


def _dot_node_id(address: str) -> str:
    """Convert a choice address to a valid DOT node ID."""
    return "n_" + address.replace(" ", "_").replace(".", "_").replace("-", "_").replace("/", "_").replace("[", "_").replace("]", "_")


def _dot_escape(s: str) -> str:
    """Escape a string for use in DOT labels."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
