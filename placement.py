"""
VLSI Cell Placement Optimization
Work in progress.
"""

"""
VLSI Cell Placement Optimization

This file builds a tiny synthetic chip placement problem I can optimize later.
I generate fake cells, fake pins inside each cell, and fake connections between pins.
Then I hand those tensors to the training loop and loss functions.
"""

import os
from enum import IntEnum

import torch
import torch.optim as optim


# I kept tensor column indices in enums so the code reads like English.
# Instead of remembering "column 4 is width", I just write CellFeatureIdx.WIDTH.
class CellFeatureIdx(IntEnum):
    """Column indices for the cell feature tensor."""
    AREA = 0
    NUM_PINS = 1
    X = 2
    Y = 3
    WIDTH = 4
    HEIGHT = 5


class PinFeatureIdx(IntEnum):
    """Column indices for the pin feature tensor."""
    CELL_IDX = 0
    PIN_X = 1  # This is inside the cell, relative x
    PIN_Y = 2  # This is inside the cell, relative y
    X = 3      # This will be the absolute x after we place the cell
    Y = 4      # This will be the absolute y after we place the cell
    WIDTH = 5
    HEIGHT = 6


# These constants control what kind of random placement problem we generate.
# Think of this as our synthetic dataset generator.
MIN_MACRO_AREA = 100.0
MAX_MACRO_AREA = 10000.0

STANDARD_CELL_AREAS = [1.0, 2.0, 3.0]
STANDARD_CELL_HEIGHT = 1.0

MIN_STANDARD_CELL_PINS = 3
MAX_STANDARD_CELL_PINS = 6

# Where we save plots and outputs later.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_placement_input(num_macros, num_std_cells):
    """
    Generates a random placement problem.

    You give it how many macros and how many standard cells you want.
    It returns three tensors:
    1. cell_features which stores cell sizes and positions
    2. pin_features which stores pins inside each cell
    3. edge_list which stores random pin to pin connections

    This is the data the optimizer will try to improve.
    """
    total_cells = num_macros + num_std_cells

    # Macros get random areas in a realistic range.
    macro_areas = torch.rand(num_macros) * (MAX_MACRO_AREA - MIN_MACRO_AREA) + MIN_MACRO_AREA

    # Standard cells pick from a small discrete set of areas.
    std_cell_areas = torch.tensor(STANDARD_CELL_AREAS)[
        torch.randint(0, len(STANDARD_CELL_AREAS), (num_std_cells,))
    ]

    # One unified area vector for all cells.
    areas = torch.cat([macro_areas, std_cell_areas])

    # Macros are square so width and height are sqrt(area).
    macro_widths = torch.sqrt(macro_areas)
    macro_heights = torch.sqrt(macro_areas)

    # Standard cells have fixed height and width equal to area (since height is 1).
    std_cell_widths = std_cell_areas / STANDARD_CELL_HEIGHT
    std_cell_heights = torch.full((num_std_cells,), STANDARD_CELL_HEIGHT)

    # Combine widths and heights for all cells.
    cell_widths = torch.cat([macro_widths, std_cell_widths])
    cell_heights = torch.cat([macro_heights, std_cell_heights])

    # Each cell gets a number of pins.
    # Macros get more pins as they get bigger.
    # Standard cells get a small random number of pins.
    num_pins_per_cell = torch.zeros(total_cells, dtype=torch.int)

    for i in range(num_macros):
        sqrt_area = int(torch.sqrt(macro_areas[i]).item())
        num_pins_per_cell[i] = torch.randint(sqrt_area, 2 * sqrt_area + 1, (1,)).item()

    num_pins_per_cell[num_macros:] = torch.randint(
        MIN_STANDARD_CELL_PINS, MAX_STANDARD_CELL_PINS + 1, (num_std_cells,)
    )

    # Cell features are stored as a tensor so they work nicely with PyTorch later.
    # Positions start at 0 and will be optimized later.
    cell_features = torch.zeros(total_cells, 6)
    cell_features[:, CellFeatureIdx.AREA] = areas
    cell_features[:, CellFeatureIdx.NUM_PINS] = num_pins_per_cell.float()
    cell_features[:, CellFeatureIdx.X] = 0.0
    cell_features[:, CellFeatureIdx.Y] = 0.0
    cell_features[:, CellFeatureIdx.WIDTH] = cell_widths
    cell_features[:, CellFeatureIdx.HEIGHT] = cell_heights

    # Now we generate pins.
    # Each pin lives inside its cell and we store both relative and absolute slots.
    total_pins = num_pins_per_cell.sum().item()
    pin_features = torch.zeros(total_pins, 7)

    # Small square pins.
    PIN_SIZE = 0.1

    pin_idx = 0
    for cell_idx in range(total_cells):
        n_pins = num_pins_per_cell[cell_idx].item()
        cell_width = cell_widths[cell_idx].item()
        cell_height = cell_heights[cell_idx].item()

        # We sample pin locations inside the cell with a small margin.
        margin = PIN_SIZE / 2
        if cell_width > 2 * margin and cell_height > 2 * margin:
            pin_x = torch.rand(n_pins) * (cell_width - 2 * margin) + margin
            pin_y = torch.rand(n_pins) * (cell_height - 2 * margin) + margin
        else:
            # If a cell is tiny, we just place pins in the center.
            pin_x = torch.full((n_pins,), cell_width / 2)
            pin_y = torch.full((n_pins,), cell_height / 2)

        # Store pin information.
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.CELL_IDX] = cell_idx
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_X] = pin_x
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_Y] = pin_y

        # Initially the cell is at (0, 0) so relative equals absolute.
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.X] = pin_x
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.Y] = pin_y

        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.WIDTH] = PIN_SIZE
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.HEIGHT] = PIN_SIZE

        pin_idx += n_pins

    # Finally we create random connectivity between pins.
    # This gives us a graph that wirelength loss can later try to shorten.
    edge_list = []

    pin_to_cell = torch.zeros(total_pins, dtype=torch.long)
    pin_idx = 0
    for cell_idx, n_pins in enumerate(num_pins_per_cell):
        pin_to_cell[pin_idx : pin_idx + n_pins] = cell_idx
        pin_idx += n_pins

    adjacency = [set() for _ in range(total_pins)]

    for pin_idx in range(total_pins):
        num_connections = torch.randint(1, 4, (1,)).item()

        for _ in range(num_connections):
            other_pin = torch.randint(0, total_pins, (1,)).item()

            if other_pin == pin_idx or other_pin in adjacency[pin_idx]:
                continue

            if pin_idx < other_pin:
                edge_list.append([pin_idx, other_pin])
            else:
                edge_list.append([other_pin, pin_idx])

            adjacency[pin_idx].add(other_pin)
            adjacency[other_pin].add(pin_idx)

    if edge_list:
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        edge_list = torch.unique(edge_list, dim=0)
    else:
        edge_list = torch.zeros((0, 2), dtype=torch.long)

    print("\nGenerated placement data:")
    print(f"  Total cells: {total_cells}")
    print(f"  Total pins: {total_pins}")
    print(f"  Total edges: {len(edge_list)}")
    print(f"  Average edges per pin: {2 * len(edge_list) / total_pins:.2f}")

    return cell_features, pin_features, edge_list


# ===================== OPTIMIZATION CODE ===================================

def wirelength_attraction_loss(cell_features, pin_features, edge_list):
    """Calculate loss based on total wirelength to minimize routing.

    This is a REFERENCE IMPLEMENTATION showing how to write a differentiable loss function.

    The loss computes the Manhattan distance between connected pins and minimizes
    the total wirelength across all edges.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges

    Returns:
        Scalar loss value
    """
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Update absolute pin positions based on cell positions
    cell_positions = cell_features[:, 2:4]  # [N, 2]
    cell_indices = pin_features[:, 0].long()

    # Calculate absolute pin positions
    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, 1]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, 2]

    # Get source and target pin positions for each edge
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()

    src_x = pin_absolute_x[src_pins]
    src_y = pin_absolute_y[src_pins]
    tgt_x = pin_absolute_x[tgt_pins]
    tgt_y = pin_absolute_y[tgt_pins]

    # Calculate smooth approximation of Manhattan distance
    # Using log-sum-exp approximation for differentiability
    alpha = 0.1  # Smoothing parameter
    dx = torch.abs(src_x - tgt_x)
    dy = torch.abs(src_y - tgt_y)

    # Smooth L1 distance with numerical stability
    smooth_manhattan = alpha * torch.logsumexp(
        torch.stack([dx / alpha, dy / alpha], dim=0), dim=0
    )

    # Total wirelength
    total_wirelength = torch.sum(smooth_manhattan)

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges

def overlap_repulsion_loss(cell_features, pin_features, edge_list, epoch_progress):
    """
    Best approach so far: Compute a differentiable overlap loss for VLSI cell placement using a
    soft-Coulomb repulsion field and smooth overlap barriers.

    This function penalizes cell overlaps (primary objective) while applying a
    mild global compression field to encourage compact placements. It operates
    in O(N²) time by computing pairwise interactions between all cells.

    Components:
    1. Smooth Overlap Barrier (Main Term):
       - Uses a differentiable approximation of overlap area between each pair of cells.
       - The overlap is measured along both x and y axes using `softplus` for
         smooth gradient behavior.
       - The penalty increases quadratically with overlap area, ensuring that
         even small overlaps receive a strong push apart.
       - The sharpness of the barrier (Beta) increases with training progress
         (`epoch_progress`) to transition from soft to hard separation.

    2. Soft Coulomb Field (Auxiliary Term):
       - Adds a gentle global repulsion field that decays proportionally to
         my decaying field where `r` is pairwise distance.
       - Helps spread cells evenly early in training, reducing large-scale clustering.
       - The smoothing factor sigma is annealed to shrink over time, allowing tighter
         packing as training progresses.

    Args:
        cell_features (torch.Tensor): Tensor of shape [N, 6] containing
            [area, num_pins, x, y, width, height] for each cell.
        pin_features (torch.Tensor): Unused in this function (for compatibility).
        edge_list (torch.Tensor): Unused in this function (for compatibility).
        epoch_progress (float or torch.Tensor): Scalar in [0, 1] controlling
            annealing sharpness and field smoothing.

    Returns:
        torch.Tensor: Scalar loss combining overlap and field penalties.
            Lower is better (zero indicates no overlap).
    """
    import torch
    import torch.nn.functional as F

    N = cell_features.shape[0]
    dev = cell_features.device

    # Handle trivial case with no or single cell
    if N <= 1:
        return torch.tensor(0.0, device=dev, requires_grad=True)

    # Extract cell geometry 
    x, y = cell_features[:, 2], cell_features[:, 3]  # positions
    w, h = cell_features[:, 4], cell_features[:, 5]  # sizes

    # Compute pairwise deltas (broadcasted) 
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    mask = 1.0 - torch.eye(N, device=dev)  # ignore self-pairs

    # Minimum required separations along each axis 
    minx = 0.5 * (w[:, None] + w[None, :])
    miny = 0.5 * (h[:, None] + h[None, :])

    # Smooth Overlap Barrier 
    ep = torch.as_tensor(epoch_progress, dtype=torch.float32, device=dev)
    beta = 0.1 + 4.0 * (ep ** 2)  # annealed sharpness for softplus
    ox = F.softplus(minx - dx.abs(), beta=beta)  # soft overlap extent in x
    oy = F.softplus(miny - dy.abs(), beta=beta)  # soft overlap extent in y

    # Quadratic penalty on smooth overlap area
    overlap_area = (ox * oy) ** 2
    overlap_loss = (overlap_area * mask).sum()

    # Soft Coulomb Field (Global Repulsion) 
    dist_sq = dx * dx + dy * dy + 1e-6  # pairwise squared distances
    sigma = (w.mean() + h.mean()) * (0.4 + 0.6 * (1 - ep))  # annealed smoothing width
    coulomb = (1.0 / (dist_sq + sigma**2)) * mask  # decaying field
    field_loss = coulomb.sum() * 1e-3  # small weighting for global spread

    # Total Loss 
    return overlap_loss + field_loss


def train_placement(
    id_str,
    cell_features,
    pin_features,
    edge_list,
    num_epochs=1000,
    lr=0.01,
    lambda_wirelength=1.0,
    lambda_overlap=10.0,
    verbose=True,
    log_interval=100,
):
    """Train the placement optimization using gradient descent.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        num_epochs: Number of optimization iterations
        lr: Learning rate for Adam optimizer
        lambda_wirelength: Weight for wirelength loss
        lambda_overlap: Weight for overlap loss
        verbose: Whether to print progress
        log_interval: How often to print progress

    Returns:
        Dictionary with:
            - final_cell_features: Optimized cell positions
            - initial_cell_features: Original cell positions (for comparison)
            - loss_history: Loss values over time
    """
    # Clone features and create learnable positions
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    # Make only cell positions require gradients
    # Turn off grad for everything except cell_positions
    cell_features = cell_features.clone().detach()
    cell_features.requires_grad_(False)
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    # Create optimizer
    optimizer = optim.Adam([cell_positions], lr=lr)

    # Track loss history
    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
    }

    # Training loop
    for epoch in range(num_epochs):
        if epoch < num_epochs / 10:
            lambda_overlap = 0
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2.0  # High starting learning rate

        elif epoch > 4 * num_epochs / 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.4  # Low ending learning rate
                lambda_overlap = 1
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5
            lambda_overlap = 4 * (epoch / num_epochs) ** 10
            
            
        optimizer.zero_grad()

        # Create cell_features with current positions
        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = cell_positions

        # Calculate losses
        wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list
        )
        overlap_loss = overlap_repulsion_loss(
            cell_features_current, pin_features, edge_list, epoch / num_epochs
        )
        
        # Combined loss
        total_loss = lambda_wirelength * wl_loss + lambda_overlap * overlap_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping to prevent extreme updates
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=5.0)

        # Update positions
        optimizer.step()
        
        # jitter the positions slightly to prevent getting stuck in local minima
        # cell_positions.data += torch.randn_like(cell_positions.data) * 0.01 * (1 - (epoch / num_epochs))

        # Record losses
        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss.item())

        # Log progress
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Wirelength Loss: {wl_loss.item():.6f}")
            print(f"  Overlap Loss: {overlap_loss.item():.6f}")

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            filename = f"vis/{id_str}/placement_epoch_{epoch}_wl_{wl_loss.item():.4f}_ol_{overlap_loss.item():.4f}.png"
            plot_placement(
                initial_cell_features=initial_cell_features,
                final_cell_features=cell_features_current,
                pin_features=pin_features,
                edge_list=edge_list,
                filename=filename,
            )
        

    # Create final cell features
    final_cell_features = cell_features.clone()
    final_cell_features[:, 2:4] = cell_positions.detach()

    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
    }