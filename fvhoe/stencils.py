import numpy as np
from numba import njit


@njit
def get_fv_conservative_weights(p: int, pos: str) -> np.ndarray:
    """
    Get the finite volume conservative weights for a given stencil size and position.
    args:
        p (int) : polynomial degree
        pos (str) : position of the stencil, one of 'l', 'c', or 'r'
    returns:
        np.ndarray : weights for finite volume conservative reconstruction and interpolation
    """
    if pos not in "lcr":
        raise ValueError("pos must be one of 'l', 'c', or 'r'")
    match p:
        case 0:
            return np.array([1], dtype=np.float64)
        case 1:
            if pos == "l":
                return np.array([1 / 4, 1, -1 / 4], dtype=np.float64)
            elif pos == "c":
                return np.array([0, 1, 0], dtype=np.float64)
            elif pos == "r":
                return np.array([-1 / 4, 1, 1 / 4], dtype=np.float64)
        case 2:
            if pos == "l":
                return np.array([2 / 6, 5 / 6, -1 / 6], dtype=np.float64)
            elif pos == "c":
                return np.array([-1 / 24, 26 / 24, -1 / 24], dtype=np.float64)
            elif pos == "r":
                return np.array([-1 / 6, 5 / 6, 2 / 6], dtype=np.float64)
        case 3:
            if pos == "l":
                return np.array(
                    [-1 / 24, 10 / 24, 20 / 24, -6 / 24, 1 / 24], dtype=np.float64
                )
            elif pos == "c":
                return np.array([0, -1 / 24, 26 / 24, -1 / 24, 0], dtype=np.float64)
            elif pos == "r":
                return np.array(
                    [1 / 24, -6 / 24, 20 / 24, 10 / 24, -1 / 24], dtype=np.float64
                )
        case 4:
            if pos == "l":
                return np.array(
                    [-3 / 60, 27 / 60, 47 / 60, -13 / 60, 2 / 60], dtype=np.float64
                )
            elif pos == "c":
                return np.array(
                    [9 / 1920, -116 / 1920, 2134 / 1920, -116 / 1920, 9 / 1920],
                    dtype=np.float64,
                )
            elif pos == "r":
                return np.array(
                    [2 / 60, -13 / 60, 47 / 60, 27 / 60, -3 / 60], dtype=np.float64
                )
        case 5:
            if pos == "l":
                return np.array(
                    [1 / 120, -1 / 12, 59 / 120, 47 / 60, -31 / 120, 1 / 15, -1 / 120],
                    dtype=np.float64,
                )
            elif pos == "c":
                return np.array(
                    [0, 3 / 640, -29 / 480, 1067 / 960, -29 / 480, 3 / 640, 0],
                    dtype=np.float64,
                )
            elif pos == "r":
                return np.array(
                    [-1 / 120, 1 / 15, -31 / 120, 47 / 60, 59 / 120, -1 / 12, 1 / 120],
                    dtype=np.float64,
                )
        case 6:
            if pos == "l":
                return np.array(
                    [
                        1 / 105,
                        -19 / 210,
                        107 / 210,
                        319 / 420,
                        -101 / 420,
                        5 / 84,
                        -1 / 140,
                    ],
                    dtype=np.float64,
                )
            elif pos == "c":
                return np.array(
                    [
                        -5 / 7168,
                        159 / 17920,
                        -7621 / 107520,
                        30251 / 26880,
                        -7621 / 107520,
                        159 / 17920,
                        -5 / 7168,
                    ],
                    dtype=np.float64,
                )
            elif pos == "r":
                return np.array(
                    [
                        -1 / 140,
                        5 / 84,
                        -101 / 420,
                        319 / 420,
                        107 / 210,
                        -19 / 210,
                        1 / 105,
                    ],
                    dtype=np.float64,
                )
        case 7:
            if pos == "l":
                return np.array(
                    [
                        -1 / 560,
                        17 / 840,
                        -97 / 840,
                        449 / 840,
                        319 / 420,
                        -223 / 840,
                        71 / 840,
                        -1 / 56,
                        1 / 560,
                    ],
                    dtype=np.float64,
                )
            elif pos == "c":
                return np.array(
                    [
                        0,
                        -5 / 7168,
                        159 / 17920,
                        -7621 / 107520,
                        30251 / 26880,
                        -7621 / 107520,
                        159 / 17920,
                        -5 / 7168,
                        0,
                    ],
                    dtype=np.float64,
                )
            elif pos == "r":
                return np.array(
                    [
                        1 / 560,
                        -1 / 56,
                        71 / 840,
                        -223 / 840,
                        319 / 420,
                        449 / 840,
                        -97 / 840,
                        17 / 840,
                        -1 / 560,
                    ],
                    dtype=np.float64,
                )
        case 8:
            if pos == "l":
                return np.array(
                    [
                        -1 / 504,
                        11 / 504,
                        -61 / 504,
                        275 / 504,
                        1879 / 2520,
                        -641 / 2520,
                        199 / 2520,
                        -41 / 2520,
                        1 / 630,
                    ],
                    dtype=np.float64,
                )
            elif pos == "c":
                return np.array(
                    [
                        35 / 294912,
                        -425 / 258048,
                        31471 / 2580480,
                        -100027 / 1290240,
                        5851067 / 5160960,
                        -100027 / 1290240,
                        31471 / 2580480,
                        -425 / 258048,
                        35 / 294912,
                    ],
                    dtype=np.float64,
                )
            elif pos == "r":
                return np.array(
                    [
                        1 / 630,
                        -41 / 2520,
                        199 / 2520,
                        -641 / 2520,
                        1879 / 2520,
                        275 / 504,
                        -61 / 504,
                        11 / 504,
                        -1 / 504,
                    ],
                    dtype=np.float64,
                )
    raise ValueError(f"Weights not defined for polynomial degree p={p}.")


@njit
def get_transverse_reconstruction_weights(p: int) -> np.ndarray:
    """
    Get the transverse reconstruction weights for a given polynomial degree.
    args:
        p (int) : polynomial degree
    returns:
        np.ndarray : weights for transverse reconstruction
    """
    match p:
        case 0:
            return np.array([1], dtype=np.float64)
        case 1:
            return np.array([0, 1, 0], dtype=np.float64)
        case 2:
            return np.array([1 / 24, 22 / 24, 1 / 24], dtype=np.float64)
        case 3:
            return np.array([0, 1 / 24, 22 / 24, 1 / 24, 0], dtype=np.float64)
        case 4:
            return np.array(
                [-17 / 5760, 308 / 5760, 5178 / 5760, 308 / 5760, -17 / 5760],
                dtype=np.float64,
            )
        case 5:
            return np.array(
                [0, -17 / 5760, 308 / 5760, 5178 / 5760, 308 / 5760, -17 / 5760, 0],
                dtype=np.float64,
            )
        case 6:
            return np.array(
                [
                    367 / 967680,
                    -281 / 53760,
                    6361 / 107520,
                    215641 / 241920,
                    6361 / 107520,
                    -281 / 53760,
                    367 / 967680,
                ],
                dtype=np.float64,
            )
        case 7:
            return np.array(
                [
                    0,
                    367 / 967680,
                    -281 / 53760,
                    6361 / 107520,
                    215641 / 241920,
                    6361 / 107520,
                    -281 / 53760,
                    367 / 967680,
                    0,
                ],
                dtype=np.float64,
            )
        case 8:
            return np.array(
                [
                    -27859 / 464486400,
                    49879 / 58060800,
                    -801973 / 116121600,
                    3629953 / 58060800,
                    41208059 / 46448640,
                    3629953 / 58060800,
                    -801973 / 116121600,
                    49879 / 58060800,
                    -27859 / 464486400,
                ],
                dtype=np.float64,
            )
    raise ValueError(f"Weights not defined for polynomial degree p={p}.")
