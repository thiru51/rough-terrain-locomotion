"""Terrain curriculum as a NumPy heightfield.

Reimplements the isaacgym.terrain_utils generators so terrain can be built and
tested without a simulator. Deterministic difficulty x type grid.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

TERRAIN_TYPES = ("smooth_slope", "rough_slope", "stairs_up", "stairs_down", "discrete_obstacles")


@dataclass
class TerrainConfig:
    num_rows: int = 10  # difficulty levels
    num_cols: int = 20  # terrain types
    proportions: tuple[float, ...] = (0.1, 0.1, 0.35, 0.25, 0.2)
    horizontal_scale: float = 0.1  # m per heightfield cell
    vertical_scale: float = 0.005  # m per height unit
    terrain_length: float = 16.0  # m
    terrain_width: float = 16.0  # m
    border_size: float = 25.0  # m
    platform_size: float = 3.0  # m of flat ground at the centre, for spawning
    slope_threshold: float = 0.75

    # Gentler than upstream legged_gym: a blind policy cannot climb its steps.
    max_slope: float = 0.4
    rough_amplitude: float = 0.05
    max_step_height: float = 0.10
    max_obstacle_height: float = 0.10

    @property
    def cells_per_env(self) -> int:
        return int(self.terrain_length / self.horizontal_scale)

    @property
    def cumulative_proportions(self) -> np.ndarray:
        return np.cumsum(self.proportions)


def _platform_mask(n: int, cfg: TerrainConfig) -> np.ndarray:
    """Boolean mask of the flat central platform, in cells."""
    half = int(cfg.platform_size / cfg.horizontal_scale / 2)
    centre = n // 2
    mask = np.zeros((n, n), dtype=bool)
    mask[centre - half : centre + half, centre - half : centre + half] = True
    return mask


def pyramid_slope(n: int, slope: float, cfg: TerrainConfig) -> np.ndarray:
    """Cone rising toward the centre; negative slope descends."""
    coords = np.arange(n) - n / 2
    x, y = np.meshgrid(coords, coords, indexing="ij")
    radial = np.maximum(np.abs(x), np.abs(y)) * cfg.horizontal_scale
    height = -slope * radial / cfg.vertical_scale
    return height.astype(np.int16)


def random_rough(
    n: int, amplitude: float, cfg: TerrainConfig, rng: np.random.Generator
) -> np.ndarray:
    """Uniform noise, generated at 0.2 m and upsampled so features exceed foot size."""
    step = max(1, int(0.2 / cfg.horizontal_scale))
    coarse_n = n // step + 2
    levels = int(amplitude / cfg.vertical_scale)
    coarse = rng.integers(-levels, levels + 1, size=(coarse_n, coarse_n))
    upsampled = np.kron(coarse, np.ones((step, step)))
    return upsampled[:n, :n].astype(np.int16)


def pyramid_stairs(n: int, step_height: float, cfg: TerrainConfig, step_width: float = 0.31):
    """Concentric square steps. Positive height ascends toward the centre."""
    field = np.zeros((n, n), dtype=np.int16)
    width_cells = max(1, int(step_width / cfg.horizontal_scale))
    height_units = int(step_height / cfg.vertical_scale)

    start, level = 0, 0
    while start < n // 2:
        stop = n - start
        level += height_units
        field[start:stop, start:stop] = level
        start += width_cells
    return field


def discrete_obstacles(
    n: int, height: float, cfg: TerrainConfig, rng: np.random.Generator, num_rects: int = 20
) -> np.ndarray:
    """Rectangular blocks of alternating sign, i.e. both steps up and holes."""
    field = np.zeros((n, n), dtype=np.int16)
    height_units = int(height / cfg.vertical_scale)
    min_size = max(1, int(1.0 / cfg.horizontal_scale))
    max_size = max(min_size + 1, int(2.0 / cfg.horizontal_scale))

    for _ in range(num_rects):
        w, h = rng.integers(min_size, max_size, size=2)
        i = rng.integers(0, max(1, n - w))
        j = rng.integers(0, max(1, n - h))
        field[i : i + w, j : j + h] = rng.choice([-height_units, height_units])
    return field


def make_sub_terrain(
    terrain_type: str, difficulty: float, cfg: TerrainConfig, rng: np.random.Generator
) -> np.ndarray:
    """One sub-terrain tile at a given difficulty in [0, 1)."""
    n = cfg.cells_per_env

    if terrain_type == "smooth_slope":
        field = pyramid_slope(n, cfg.max_slope * difficulty, cfg)
    elif terrain_type == "rough_slope":
        field = pyramid_slope(n, cfg.max_slope * difficulty, cfg)
        field = field + random_rough(n, cfg.rough_amplitude, cfg, rng)
    elif terrain_type == "stairs_up":
        field = pyramid_stairs(n, 0.05 + cfg.max_step_height * difficulty, cfg)
    elif terrain_type == "stairs_down":
        field = -pyramid_stairs(n, 0.05 + cfg.max_step_height * difficulty, cfg)
    elif terrain_type == "discrete_obstacles":
        field = discrete_obstacles(n, 0.05 + cfg.max_obstacle_height * difficulty, cfg, rng)
    else:
        raise ValueError(f"unknown terrain type: {terrain_type}")

    # Spawn platform must be flat and at the tile's reference height, or robots
    # start mid-obstacle and the curriculum measures nothing.
    mask = _platform_mask(n, cfg)
    field = field.astype(np.int16)
    field[mask] = field[n // 2, n // 2]
    return field


def select_type(col: int, cfg: TerrainConfig) -> str:
    choice = col / cfg.num_cols + 1e-3
    index = int(np.searchsorted(cfg.cumulative_proportions, choice))
    return TERRAIN_TYPES[min(index, len(TERRAIN_TYPES) - 1)]


@dataclass
class TerrainGrid:
    """The full curriculum: `num_rows` x `num_cols` tiles in one heightfield."""

    cfg: TerrainConfig
    seed: int = 0
    heightfield: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        n = self.cfg.cells_per_env
        border = int(self.cfg.border_size / self.cfg.horizontal_scale)

        self.heightfield = np.zeros(
            (self.cfg.num_rows * n + 2 * border, self.cfg.num_cols * n + 2 * border),
            dtype=np.int16,
        )
        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                tile = make_sub_terrain(
                    select_type(col, self.cfg), row / self.cfg.num_rows, self.cfg, rng
                )
                i, j = border + row * n, border + col * n
                self.heightfield[i : i + n, j : j + n] = tile

    def update_levels(self, levels, distance_walked, commanded_distance):
        """Promote or demote terrain rows after an episode.

        Solving the top row wraps to a random row rather than pinning there, so
        easy terrain stays in the distribution.
        """
        top = self.cfg.num_rows - 1
        levels = levels + (distance_walked > self.cfg.terrain_length / 2).long()
        levels -= (distance_walked < commanded_distance * 0.5).long()
        solved = levels > top
        levels = torch.where(solved, torch.randint_like(levels, 0, top + 1), levels)
        return levels.clip(0, top)

    def origin(self, row: int, col: int) -> tuple[float, float, float]:
        """World-frame spawn point at the centre of a tile, in metres."""
        n = self.cfg.cells_per_env
        border = int(self.cfg.border_size / self.cfg.horizontal_scale)
        i, j = border + row * n + n // 2, border + col * n + n // 2
        return (
            i * self.cfg.horizontal_scale,
            j * self.cfg.horizontal_scale,
            float(self.heightfield[i, j]) * self.cfg.vertical_scale,
        )
