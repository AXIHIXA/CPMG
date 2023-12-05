import os
import time
import typing

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse

import build.cpmg as cpmg
import torch


def plt_subplot(dic: dict[str, np.ndarray],
                suptitle: typing.Optional[str] = None,
                unit_size: int = 5,
                show: bool = True,
                dump: typing.Optional[str] = None,
                dpi: typing.Optional[int] = None,
                show_axis: bool = True) -> None:
    fig = plt.figure(figsize=(unit_size * len(dic), unit_size), dpi=dpi)

    if suptitle:
        plt.suptitle(suptitle)

    for i, (k, v) in enumerate(dic.items()):
        plt.subplot(1, len(dic), i + 1)
        plt.axis(show_axis)
        plt.title(k)

        if v is not None:
            plt.imshow(v.squeeze())

        plt.colorbar()

    if dump is not None:
        os.makedirs(dump[:dump.rfind('/')], exist_ok=True)
        plt.savefig(dump, bbox_inches='tight')

    # show must follow savefig otherwise the saved image would be blank
    if show:
        plt.show()

    plt.close(fig)


def initial_guess(bc_value: torch.Tensor, bc_mask: torch.Tensor, initialization: str) -> torch.Tensor:
    """
    Assemble the initial guess of solution.
    """
    if initialization == 'random':
        routine = torch.rand_like   # U[0, 1)
    elif initialization == 'zero':
        routine = torch.zeros_like
    else:
        raise NotImplementedError

    return (1 - bc_mask) * routine(bc_value) + bc_value


def main():
    size: int = 1025
    mask: np.ndarray = np.ones((size, size), dtype=np.uint8)
    cv2.circle(mask, [size // 2, size // 2], size // 3, 0, cv2.FILLED)
    mask = mask.astype(bool)

    f: np.ndarray = np.ones_like(mask, dtype=np.float32) * 0.0001
    f *= 1 - mask
    b: np.ndarray = np.ones_like(f)
    b[:, size // 2:] = 2.0
    b *= mask

    tm: float = 0.0
    terror: float = 0.0
    duplication: int = 20

    for _ in range(duplication):
        x = initial_guess(torch.from_numpy(b), torch.from_numpy(mask).float(), 'random').numpy()
        iters, error, time_elapsed, y = cpmg.amgcl_solve(mask, f, b, x, 1e-4)
        y.resize((size, size))
        tm += time_elapsed
        terror += error

    print(f'AMGCL CUDA PyBind11 Extension took {tm / duplication} ms.\n'
          f'AMGCL CUDA Averange relative error {terror / duplication}.')

    # print(f'Iterations:   {iters}\n'
    #       f'Error:        {error}\n'
    #       f'Time Elapsed: {time_elapsed} ms')
    # plt_subplot({'mask': mask, 'f': f, 'b': b, 'x': x})

    # size: int = 1000000000
    # runs: int = 2
    # res: float = 0.0  # placeholder
    # rng = np.random.default_rng()
    #
    # start: float = time.time()
    # arr: np.ndarray = rng.random(size, np.float64)  # float32 overflows for this test
    # end: float = time.time()
    # print(f'Python initialization took: {(end - start) * 1000} ms')
    #
    # start = time.time()
    # for _ in range(runs):
    #     res = np.sum(arr).item()
    # end = time.time()
    # print(f'Python built-in reduction took: {(end - start) * 1000} ms')
    # print(f'Python built-in reduction returns: {res}')
    #
    # start = time.time()
    # for _ in range(runs):
    #     res = cpx.cu(arr)
    # end = time.time()
    # print(f'C++ STL (TBB-based) parallel reduction took: {(end - start) * 1000} ms')
    # print(f'C++ STL (TBB-based) parallel reduction returns: {res}')


if __name__ == '__main__':
    main()
