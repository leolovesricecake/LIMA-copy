"""Pure-Python LightGBM dump conversion used when the optional C++ parser is unavailable."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from shapiq.tree.base import TreeModel


def _is_split_node(node: Mapping[str, Any]) -> bool:
    """Return whether one LightGBM dump node is an internal split."""

    return "split_index" in node


def _is_leaf_node(node: Mapping[str, Any]) -> bool:
    """Return whether one dump node is a leaf, including an unindexed root leaf."""

    return "leaf_value" in node and not _is_split_node(node)


def _collect_nodes(
    node: Mapping[str, Any],
    *,
    split_nodes: list[Mapping[str, Any]],
    leaf_nodes: list[Mapping[str, Any]],
) -> None:
    """Collect every split and leaf node from one nested LightGBM tree dump."""

    if _is_split_node(node):
        split_nodes.append(node)
        _collect_nodes(
            node["left_child"],
            split_nodes=split_nodes,
            leaf_nodes=leaf_nodes,
        )
        _collect_nodes(
            node["right_child"],
            split_nodes=split_nodes,
            leaf_nodes=leaf_nodes,
        )
        return
    if _is_leaf_node(node):
        leaf_nodes.append(node)
        return
    raise ValueError(
        "LightGBM tree dump contains a node that is neither a split nor a leaf; "
        f"available keys are {sorted(str(key) for key in node)}."
    )


def _validate_contiguous_indexes(
    nodes: list[Mapping[str, Any]],
    *,
    key: str,
) -> None:
    """Ensure dump indexes can be mapped exactly to the native converter layout."""

    indexes = sorted(int(node[key]) for node in nodes)
    expected = list(range(len(nodes)))
    if indexes != expected:
        raise ValueError(
            f"LightGBM {key} values must be contiguous; got {indexes}, expected {expected}."
        )


def _resolve_leaf_indexes(
    leaf_nodes: list[Mapping[str, Any]],
) -> dict[int, int]:
    """Assign contiguous indexes when LightGBM omits one for a root-only leaf."""

    n_leaves = len(leaf_nodes)
    indexes_by_identity: dict[int, int] = {}
    claimed_indexes: set[int] = set()
    for node in leaf_nodes:
        if "leaf_index" not in node:
            continue
        leaf_index = int(node["leaf_index"])
        if not 0 <= leaf_index < n_leaves:
            raise ValueError(
                f"LightGBM leaf_index {leaf_index} is outside [0, {n_leaves})."
            )
        if leaf_index in claimed_indexes:
            raise ValueError(f"LightGBM leaf_index {leaf_index} is duplicated.")
        indexes_by_identity[id(node)] = leaf_index
        claimed_indexes.add(leaf_index)

    available_indexes = iter(sorted(set(range(n_leaves)) - claimed_indexes))
    for node in leaf_nodes:
        if id(node) not in indexes_by_identity:
            indexes_by_identity[id(node)] = next(available_indexes)
    return indexes_by_identity


def _dump_node_id(
    node: Mapping[str, Any],
    *,
    n_internal: int,
    leaf_indexes: Mapping[int, int],
) -> int:
    """Map LightGBM split/leaf indexes to the native converter node ordering."""

    if _is_split_node(node):
        return int(node["split_index"])
    if not _is_leaf_node(node):
        raise ValueError("Cannot assign a node id to an invalid LightGBM dump node.")
    return int(n_internal) + int(leaf_indexes[id(node)])


def _sample_count(node: Mapping[str, Any], *, leaf: bool) -> float:
    """Read the count used by the native converter as node sample weight."""

    count_key = "leaf_count" if leaf else "internal_count"
    weight_key = "leaf_weight" if leaf else "internal_weight"
    if count_key in node:
        return float(node[count_key])
    if weight_key in node:
        return float(node[weight_key])
    raise ValueError(f"LightGBM tree dump node is missing {count_key!r}.")


def tree_model_from_lightgbm_dump(tree_structure: Mapping[str, Any]) -> TreeModel:
    """Convert one nested LightGBM tree dump into shapiq's array representation."""

    split_nodes: list[Mapping[str, Any]] = []
    leaf_nodes: list[Mapping[str, Any]] = []
    _collect_nodes(
        tree_structure,
        split_nodes=split_nodes,
        leaf_nodes=leaf_nodes,
    )
    _validate_contiguous_indexes(split_nodes, key="split_index")
    leaf_indexes = _resolve_leaf_indexes(leaf_nodes)

    n_internal = len(split_nodes)
    n_nodes = n_internal + len(leaf_nodes)
    children_left = np.full(n_nodes, -1, dtype=np.int64)
    children_right = np.full(n_nodes, -1, dtype=np.int64)
    children_missing = np.full(n_nodes, -1, dtype=np.int64)
    features = np.full(n_nodes, -1, dtype=np.int64)
    thresholds = np.zeros(n_nodes, dtype=np.float64)
    values = np.zeros(n_nodes, dtype=np.float64)
    node_sample_weight = np.zeros(n_nodes, dtype=np.float64)

    for node in split_nodes:
        node_id = int(node["split_index"])
        decision_type = str(node.get("decision_type", "<="))
        if decision_type != "<=":
            raise NotImplementedError(
                "ProxySPEX LightGBM conversion supports numerical '<=' splits only; "
                f"received decision_type={decision_type!r}."
            )
        left_id = _dump_node_id(
            node["left_child"],
            n_internal=n_internal,
            leaf_indexes=leaf_indexes,
        )
        right_id = _dump_node_id(
            node["right_child"],
            n_internal=n_internal,
            leaf_indexes=leaf_indexes,
        )
        children_left[node_id] = left_id
        children_right[node_id] = right_id
        children_missing[node_id] = left_id if bool(node.get("default_left", True)) else right_id
        features[node_id] = int(node["split_feature"])
        thresholds[node_id] = float(node["threshold"])
        node_sample_weight[node_id] = _sample_count(node, leaf=False)

    for node in leaf_nodes:
        node_id = n_internal + int(leaf_indexes[id(node)])
        values[node_id] = float(node["leaf_value"])
        node_sample_weight[node_id] = _sample_count(node, leaf=True)

    tree_model = TreeModel(
        children_left=children_left,
        children_right=children_right,
        children_missing=children_missing,
        features=features,
        thresholds=thresholds,
        values=values,
        node_sample_weight=node_sample_weight,
        decision_type="<=",
    )
    tree_model.conversion_backend = "lightgbm_python_dump"
    return tree_model


def _resolve_booster(model: object) -> object:
    """Resolve a fitted sklearn LightGBM wrapper to its native Booster."""

    for attribute in ("booster_", "_Booster"):
        booster = getattr(model, attribute, None)
        if booster is not None and hasattr(booster, "dump_model"):
            return booster
    if hasattr(model, "dump_model"):
        return model
    raise TypeError("Expected a fitted LightGBM model or Booster exposing dump_model().")


def convert_lightgbm_dump_model(
    model: object,
    class_label: int | None = None,
) -> list[TreeModel]:
    """Convert a fitted LightGBM model through its structured model dump."""

    booster = _resolve_booster(model)
    model_dump = booster.dump_model()
    if not isinstance(model_dump, Mapping):
        raise TypeError("LightGBM dump_model() must return a mapping.")
    tree_info = model_dump.get("tree_info")
    if not isinstance(tree_info, list):
        raise ValueError("LightGBM model dump is missing the tree_info list.")

    trees_per_iteration = int(model_dump.get("num_tree_per_iteration", 1))
    if class_label is not None and not 0 <= int(class_label) < trees_per_iteration:
        raise ValueError(
            f"class_label must be in [0, {trees_per_iteration}); got {class_label}."
        )

    converted = []
    for tree_index, tree_entry in enumerate(tree_info):
        if (
            class_label is not None
            and trees_per_iteration > 1
            and tree_index % trees_per_iteration != int(class_label)
        ):
            continue
        if not isinstance(tree_entry, Mapping):
            raise TypeError("Each LightGBM tree_info entry must be a mapping.")
        tree_structure = tree_entry.get("tree_structure")
        if not isinstance(tree_structure, Mapping):
            raise ValueError("LightGBM tree_info entry is missing tree_structure.")
        converted.append(tree_model_from_lightgbm_dump(tree_structure))
    return converted
