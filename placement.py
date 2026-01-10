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

