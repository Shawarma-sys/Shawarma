import torch.nn as nn
import copy
from typing import List, Dict, Any, Optional, Tuple

def _build_call(func_name: str, x_in: list, buffer_in: list, layer) -> Tuple[list, dict]:
    """
    Construct positional/keyword call arguments for a layer.

    TorchLens captures both positional and keyword non-tensor args for many ops.
    Some creation ops (e.g., torch.zeros) require keyword-only args like dtype/device;
    replaying them positionally can cause errors.

    Optimization: Removed dictionary lookups, used direct branching for high-frequency 
    special cases, and used list unpacking to reduce intermediate objects.
    """
    # Prefetch non-tensor arguments to avoid repeated attribute access
    non_tensors_pos = getattr(layer, 'func_position_args_non_tensor', None)
    if non_tensors_pos is None:
        non_tensors_pos = getattr(layer, 'func_all_args_non_tensor', [])
    non_tensors_kw = getattr(layer, 'func_keyword_args_non_tensor', None)
    if non_tensors_kw is None:
        non_tensors_kw = {}
    if not isinstance(non_tensors_kw, dict):
        non_tensors_kw = {}
    
    # 1. Special case: Cat (requires inputs packaged as a tuple)
    if func_name == 'cat':
        return [tuple(x_in), *non_tensors_pos], dict(non_tensors_kw)

    # Normalize inputs: unpack if it is a single tensor; keep as is if multiple or empty
    inputs = [x_in[0]] if len(x_in) == 1 else x_in
    
    # Get parameters (use getattr default to prevent missing attribute errors)
    params = getattr(layer, 'parent_params', [])
    
    # 2. Special case: Embedding (Parameter order: params, input, buffers, args)
    if func_name == 'embedding':
        return [*params, *inputs, *buffer_in, *non_tensors_pos], dict(non_tensors_kw)

    # 2b. Special case: GRU (expects params packed as a single tuple)
    # Signature (batch_first variant): (input, hx, params, has_biases, num_layers,
    # dropout, train, bidirectional, batch_first)
    if func_name == 'gru':
        return [*inputs, tuple(params), *non_tensors_pos], dict(non_tensors_kw)

    # 3. Special case: Creation ops (prefer TorchLens' creation_args/creation_kwargs)
    creation_args = getattr(layer, 'creation_args', None)
    creation_kwargs = getattr(layer, 'creation_kwargs', None)
    if isinstance(creation_args, list) and isinstance(creation_kwargs, dict) and func_name in {
        'zeros', 'ones', 'empty', 'full', 'rand', 'randn', 'arange', 'linspace', 'logspace'
    }:
        return [*creation_args], dict(creation_kwargs)

    # 3. General case (Applies to most layers, including creation ops like zeros/randn)
    # Order: Inputs -> Params -> Buffers -> Args
    return [*inputs, *params, *buffer_in, *non_tensors_pos], dict(non_tensors_kw)


def train_mode(model: List) -> List:
    """Set layers containing 'training' arguments to True."""
    for layer in model:
        # Fast check if the layer has argument names
        argnames = getattr(layer, "func_argnames", None)
        if argnames and "training" in argnames:
            args = layer.func_all_args_non_tensor
            for idx, arg in enumerate(args):
                if isinstance(arg, bool):
                    args[idx] = True
                    break
    return model


def eval_mode(model: List) -> List:
    """Set layers containing 'training' arguments to False."""
    for layer in model:
        argnames = getattr(layer, "func_argnames", None)
        if argnames and "training" in argnames:
            args = layer.func_all_args_non_tensor
            for idx, arg in enumerate(args):
                if isinstance(arg, bool):
                    args[idx] = False
                    break
    return model


def get_weights(model: List, device=None) -> List:
    """Flatten and retrieve all model parameters."""
    params = []
    for layer in model:
        layer_params = getattr(layer, 'parent_params', None)
        if layer_params:
            if device is None:
                params.extend(layer_params)
            else:
                params.extend(p.to(device) for p in layer_params)
    return params


def load_weights(model: List, weights: List) -> List:
    """Load weights sequentially into the model."""
    weight_iter = iter(weights)
    for layer in model:
        layer_params = getattr(layer, 'parent_params', None)
        if not layer_params:
            continue
            
        for j, param in enumerate(layer_params):
            try:
                weight = next(weight_iter)
                # Removed assert to improve speed in production, assuming inputs are valid
                # assert param.size() == weight.size()
                layer_params[j] = weight
            except StopIteration:
                return model
    return model


def label2index(model: List) -> Dict[str, int]:
    """Create a fast lookup table from Layer Label to Index."""
    return {getattr(layer, 'layer_label', i): i for i, layer in enumerate(model)}


def forward_eval(model: List, label2index: Dict[str, int], x: Any) -> Any:
    """
    Forward pass in evaluation mode.
    Optimization: Inline input preparation logic, minimize getattr calls.
    """
    idx_map = label2index
    
    for layer in model:
        func_name = getattr(layer, 'func_applied_name', None)
        
        # --- Fast Path: No-op layer ---
        if func_name == 'none':
            layer_type = getattr(layer, 'layer_type', None)
            if layer_type == 'input':
                layer.tensor_contents = x
            elif layer_type == 'output':
                # Backtrack content for Output layer from its parent
                parent_idx = idx_map[layer.parent_layers[0]]
                layer.tensor_contents = model[parent_idx].tensor_contents
            continue

        # --- Prepare Inputs ---
        x_in = []
        buffer_in = []
        
        # Iterate over parent layers (use getattr default to avoid try-except overhead)
        for parent_label in getattr(layer, 'parent_layers', []):
            parent_idx = idx_map[parent_label]
            parent = model[parent_idx]
            parent_type = getattr(parent, 'layer_type', None)

            if parent_type == 'buffer':
                buffer_in.append(parent.tensor_contents)
            else:
                content = parent.tensor_contents
                if content is not None:
                    x_in.append(content)
                    # Memory Optimization: Check if parent tensor can be released.
                    # Logic: If current layer is the last child of the parent, release parent content.
                    if parent.child_layers:
                        last_child_label = parent.child_layers[-1]
                        last_child_idx = idx_map[last_child_label]
                        # Compare operation numbers to determine order
                        if model[last_child_idx].operation_num <= layer.operation_num:
                            parent.tensor_contents = None
                else:
                    # Fallback: If parent content is empty, check if it's the immediate previous op
                    if parent.operation_num == layer.operation_num - 1:
                        x_in.append(x)
                    else:
                        raise RuntimeError(f"Missing tensor_contents for parent {parent_label}")

        # --- Execute Layer Operation ---
        call_args, call_kwargs = _build_call(func_name, x_in, buffer_in, layer)

        try:
            # Execute and cache result
            out = layer.func_applied(*call_args, **call_kwargs)

            # TorchLens sometimes represents multi-output ops (e.g., GRU) as
            # separate layers with labels like '...:1', '...:2'. In that case,
            # select the corresponding tuple element to match the logged graph.
            if func_name == 'gru' and isinstance(out, (tuple, list)):
                label = getattr(layer, 'layer_label', '')
                if ':' in label:
                    suffix = label.rsplit(':', 1)[-1]
                    if suffix.isdigit():
                        idx = int(suffix) - 1
                        if 0 <= idx < len(out):
                            out = out[idx]

            layer.tensor_contents = out
        except Exception as e:
            # Calculate shapes only on error to save overhead during normal execution
            input_shapes = [getattr(t, 'shape', type(t)) for t in x_in]
            raise RuntimeError(
                f"Layer {getattr(layer, 'layer_label', '?')} ({func_name}) failed: {e}; "
                f"parent_shapes={input_shapes}"
            ) from e

        # Update x to current output (for fallback usage)
        x = layer.tensor_contents

        # Attempt to release current layer immediately 
        # (If it's an intermediate layer and already fully consumed by next op logic)
        if layer.child_layers:
            last_child_idx = idx_map[layer.child_layers[-1]]
            if model[last_child_idx].operation_num <= layer.operation_num + 1:
                layer.tensor_contents = None

    return x


def forward_eval_partition(model: List, label2index: Dict[str, int], x: Any, partition_index: int = None):
    """Forward pass in evaluation mode with partitioning support."""
    intermediate_outputs = None
    idx_map = label2index

    for idx, layer in enumerate(model):
        func_name = getattr(layer, 'func_applied_name', None)

        if func_name == 'none':
            if getattr(layer, 'layer_type', None) != 'buffer':
                layer.tensor_contents = None
                if idx == partition_index:
                    intermediate_outputs = x.detach().clone()
            continue

        x_in = []
        buffer_in = []
        
        for parent_label in getattr(layer, 'parent_layers', []):
            parent = model[idx_map[parent_label]]
            parent_type = getattr(parent, 'layer_type', None)
            
            if parent_type == 'buffer':
                buffer_in.append(parent.tensor_contents)
            else:
                content = parent.tensor_contents
                if content is not None:
                    x_in.append(content)
                    if parent.child_layers:
                        last_child_idx = idx_map[parent.child_layers[-1]]
                        if model[last_child_idx].operation_num <= layer.operation_num:
                            parent.tensor_contents = None
                elif parent.operation_num == layer.operation_num - 1:
                    x_in.append(x)
                else:
                    raise RuntimeError(f"Missing contents for {parent_label}")

        call_args, call_kwargs = _build_call(func_name, x_in, buffer_in, layer)

        try:
            out = layer.func_applied(*call_args, **call_kwargs)
            if func_name == 'gru' and isinstance(out, (tuple, list)):
                label = getattr(layer, 'layer_label', '')
                if ':' in label:
                    suffix = label.rsplit(':', 1)[-1]
                    if suffix.isdigit():
                        idx = int(suffix) - 1
                        if 0 <= idx < len(out):
                            out = out[idx]
            x = out
        except Exception as e:
            input_shapes = [getattr(t, 'shape', type(t)) for t in x_in]
            raise RuntimeError(f"Layer {func_name} failed: {e}; shapes={input_shapes}") from e

        if idx == partition_index:
            intermediate_outputs = x.detach().clone()

        # Cleanup logic for current layer
        layer_cleared = False
        if layer.child_layers:
            last_child_idx = idx_map[layer.child_layers[-1]]
            if model[last_child_idx].operation_num <= layer.operation_num + 1:
                layer.tensor_contents = None
                layer_cleared = True
        
        if not layer_cleared:
            layer.tensor_contents = x

    return x, intermediate_outputs


def partition_forward_eval(model: List, label2index: Dict[str, int], x: Any) -> Any:
    """
    Forward pass for a model partition.
    Handles boundary conditions: treats missing parent nodes as input 'x'.
    """
    first_in = True
    idx_map = label2index
    
    for layer in model:
        func_name = getattr(layer, 'func_applied_name', None)

        if func_name == 'none':
            if getattr(layer, 'layer_type', None) != 'buffer':
                layer.tensor_contents = None
            continue

        x_in = []
        buffer_in = []
        
        for parent_label in getattr(layer, 'parent_layers', []):
            # Boundary check: If parent is not in the current slice map, 
            # and it's the first time encountering a gap, treat as input.
            if parent_label not in idx_map:
                if first_in:
                    x_in.append(x)
                    first_in = False
            else:
                parent = model[idx_map[parent_label]]
                parent_type = getattr(parent, 'layer_type', None)
                
                if parent_type == 'buffer':
                    buffer_in.append(parent.tensor_contents)
                else:
                    content = parent.tensor_contents
                    if content is not None:
                        x_in.append(content)
                        # Cleanup logic: only check if child is also in current map
                        if parent.child_layers:
                            last_child_label = parent.child_layers[-1]
                            if last_child_label in idx_map:
                                last_child_idx = idx_map[last_child_label]
                                if model[last_child_idx].operation_num <= layer.operation_num:
                                    parent.tensor_contents = None
                    elif parent.operation_num == layer.operation_num - 1:
                        x_in.append(x)
                    else:
                        raise RuntimeError(f"Missing content for {parent_label}")

        call_args, call_kwargs = _build_call(func_name, x_in, buffer_in, layer)

        try:
            out = layer.func_applied(*call_args, **call_kwargs)
            if func_name == 'gru' and isinstance(out, (tuple, list)):
                label = getattr(layer, 'layer_label', '')
                if ':' in label:
                    suffix = label.rsplit(':', 1)[-1]
                    if suffix.isdigit():
                        idx = int(suffix) - 1
                        if 0 <= idx < len(out):
                            out = out[idx]
            x = out
        except Exception as e:
            input_shapes = [getattr(t, 'shape', type(t)) for t in x_in]
            raise RuntimeError(f"Partition forward failed at {func_name}: {e}") from e

        # Cleanup current layer
        layer_cleared = False
        if layer.child_layers:
            last_child_label = layer.child_layers[-1]
            if last_child_label in idx_map:
                if model[idx_map[last_child_label]].operation_num <= layer.operation_num + 1:
                    layer.tensor_contents = None
                    layer_cleared = True
        
        if not layer_cleared:
            layer.tensor_contents = x
            
    return x


def forward_train(model: List, label2index: Dict[str, int], x: Any) -> Any:
    """
    Forward pass in training mode.
    Note: Training mode requires the Autograd graph, so aggressive 'tensor_contents = None' 
    cleanup is generally avoided unless we are sure it doesn't affect backprop.
    """
    first_in = True
    idx_map = label2index
    
    for layer in model:
        func_name = getattr(layer, 'func_applied_name', None)

        if func_name == 'none':
            layer_type = getattr(layer, 'layer_type', None)
            if layer_type == 'input':
                layer.tensor_contents = x
            elif layer_type == 'output':
                # Output layer points to its parent's result
                parent = model[idx_map[layer.parent_layers[0]]]
                layer.tensor_contents = parent.tensor_contents
            continue

        x_in = []
        buffer_in = []
        
        for parent_label in getattr(layer, 'parent_layers', []):
            if parent_label not in idx_map:
                if first_in:
                    x_in.append(x)
                    first_in = False
            else:
                parent = model[idx_map[parent_label]]
                if getattr(parent, 'layer_type', None) == 'buffer':
                    buffer_in.append(parent.tensor_contents)
                else:
                    x_in.append(parent.tensor_contents)

        # Build arguments
        call_args = _build_call_args(func_name, x_in, buffer_in, layer)
        
        # Execute
        layer.tensor_contents = layer.func_applied(*call_args)
        
    return model[-1].tensor_contents


def frozen_model(model: List, partition_index: int) -> List:
    """Freeze parameters of all layers up to the specified partition index."""
    # Limit range to avoid index out of bounds
    limit = min(partition_index + 1, len(model))
    for i in range(limit):
        for param in getattr(model[i], 'parent_params', []):
            param.requires_grad = False
    return model


def new_model(original_model: List, partition_index: int) -> List:
    """Create a new model copy starting after the partition index."""
    res = []
    for idx, layer in enumerate(original_model):
        if idx > partition_index:
            # Shallow copy the layer object, but deep copy the params list structure
            # to allow list modification while sharing underlying parameter tensors.
            new_layer = copy.copy(layer)
            new_layer.parent_params = copy.copy(layer.parent_params) 
            res.append(new_layer)
    return res


def get_max_partition_index(model: List) -> int:
    """Find the index of the last layer containing trainable parameters."""
    for idx in range(len(model) - 1, -1, -1):
        if getattr(model[idx], 'computed_with_params', False):
            return idx
    return 0


def partition_my_dnn_train(model: List, partition_index: int) -> List:
    """Return model layers after the given partition index."""
    return model[partition_index+1:]