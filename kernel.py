import typing

import torch
# noinspection PyPep8Naming
import torch.nn.functional as F


__use_cpu: bool = False


def get_device(use_cpu: bool = __use_cpu) -> torch.device:
    return torch.device('cuda') if (not use_cpu and torch.cuda.is_available()) else torch.device('cpu')


__device: torch.device = get_device()


"""
Masked discrete Poisson equation (with arbitray Dirchilet boundary condition): 
        (I - bc_mask) A x = (I - bc_mask) f
             bc_mask    x =        bc_value  
Note for simplicity, 
    we preprocess bc_value s.t. bc_value == bc_mask bc_value, 
    and we have the exterior band of x be zero (trivial boundary condition). 
 
Masked Jacobi update: Step matrix P = -4I
        x' = (I - bc_mask) ( (I - P^-1 A) x                        + P^-1 f ) + bc_value
           = (I - bc_mask) ( F.conv2d(x, jacobi_kernel, padding=1) - 0.25 f ) + bc_value
           = util.jacobi_step(x)
"""


jacobi_kernel = torch.tensor(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]],
        dtype=torch.float32
).view(1, 1, 3, 3).to(__device) / 4.0


laplace_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=torch.float32
).view(1, 1, 3, 3).to(__device)


# Half-weighting
restriction_kernel = torch.tensor(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]],
        dtype=torch.float32
).view(1, 1, 3, 3).to(__device) / 8.0


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


def jacobi_step(x: torch.Tensor, bc_value: torch.Tensor, bc_mask: torch.Tensor, f: typing.Optional[torch.Tensor]):
    """
    One iteration step of masked Jacobi iterative solver.
    """
    y = F.conv2d(x, jacobi_kernel, padding=1)

    if f is not None:
        y = y - 0.25 * f

    return (1 - bc_mask) * y + bc_value


# class MultigridIterator(torch.nn.Module):
#     def __init__(self, num_layers: int, num_pre_smoothing: int, num_post_smoothing: int):
#         super().__init__()
#         self.num_layers: int = num_layers
#         self.pre_smoothing: int = num_pre_smoothing
#         self.post_smoothing: int = num_post_smoothing
#
#     def __mg_step(self,
#                   x: torch.Tensor,
#                   bc_value: torch.Tensor,
#                   bc_mask: torch.Tensor,
#                   f: torch.Tensor,
#                   step: int) -> torch.Tensor:
#         """
#         One layer of multigrid. Recursive function.
#         Find solution x to Ax + b = 0.
#         """
#         # Pre smoothing
#         for i in range(self.pre_smoothing):
#             x = jacobi_step(x, bc_value, bc_mask, f)
#
#         # Multigrid
#         if step > 1:
#             # Downsample
#             if f is not None:
#                 f_sub = 4 * downsample2x(f)
#             else:
#                 f_sub = None
#
#             # Subsample geometry
#             bc_value_sub = downsample2x(bc_value)
#             bc_mask_sub = downsample2x(bc_mask)
#
#             x_sub = restrict(x, bc_value_sub, bc_mask_sub)
#
#             # Refine x_sub recursively
#             x_sub = self.__mg_step(x_sub, bc_value_sub, bc_mask_sub, f_sub, step - 1)
#
#             # Upsample
#             x = upsample2x(x_sub)
#
#         # Post smoothing
#         for i in range(self.post_smoothing):
#             x = jacobi_step(x, bc_value, bc_mask, f)
#
#         return x
#
#     def forward(self,
#                 x: torch.Tensor,
#                 bc_value: torch.Tensor,
#                 bc_mask: torch.Tensor,
#                 f: torch.Tensor, ) -> torch.Tensor:
#         """
#         x, f: size (batch_size x image_size x image_size)
#         return: same size
#         """
#         y = self.__mg_step(x, bc_value, bc_mask, f, self.num_layers)
#         return y
#
#
# def restrict(x: torch.Tensor, bc_value: torch.Tensor, bc_mask: torch.Tensor) -> torch.Tensor:
#     """
#     Multigrid restriction step of an image of size 2^N + 1.
#     E.g., 257 -> 129 -> 65 -> ...
#     """
#     y = F.conv2d(x, restriction_kernel, stride=2, padding=1)
#     y = y * (1 - bc_mask) + bc_value
#     return y


def restrict(x: torch.Tensor) -> torch.Tensor:
    """
    Half-weighting fine-to-coarse restriction of an image of size 2^N + 1.
    Favored over direct injection for the residue (NOT for boundary geometries!).
    E.g., 257 -> 129 -> 65 -> ...
    """
    y = F.conv2d(x, restriction_kernel, stride=2, padding=1)
    return y


def downsample2x(x: torch.Tensor) -> torch.Tensor:
    """
    Bilinear 2x-downsampling of an image of size 2^N + 1 is essentially direct injection.
    E.g., 257 -> 129 -> 65 -> ...

    Note: torch.nn.UpsamplingBilinear2d is deprecated testcase favor of interpolate.
    It is equivalent to nn.functional.interpolate(..., mode='bilinear', align_corners=True).
    """
    new_size = (x.size(-1) - 1) // 2 + 1
    y = F.interpolate(x, size=new_size, mode='bilinear', align_corners=True)
    return y


def upsample2x(x: torch.Tensor) -> torch.Tensor:
    """
    Bilinear 2x-upsampling of an image of size 2^N + 1.
    E.g., 65 -> 129 -> 257 -> ...

    Note: torch.nn.UpsamplingBilinear2d is deprecated testcase favor of interpolate.
    It is equivalent to nn.functional.interpolate(..., mode='bilinear', align_corners=True).
    """
    new_size = x.size(-1) * 2 - 1
    y = F.interpolate(x, size=new_size, mode='bilinear', align_corners=True)
    return y


def norm(x: torch.Tensor) -> torch.Tensor:
    """
    Vector norm on each batch.
    Note: We only deal with cases where channel == 1!
    :param x: (batch_size, channel, image_size, image_size)
    :return: (batch_size,)
    """
    y = x.view(x.size(0), -1)
    return (y * y).sum(dim=1).sqrt()


def absolute_residue(x: torch.Tensor,
                     bc_mask: torch.Tensor,
                     f: typing.Optional[torch.Tensor],
                     reduction: str = 'norm') -> torch.Tensor:
    """
    For a linear system Ax = f,
    the absolute residue is r = f - Ax,
    the absolute residual (norm) error eps = ||f - Ax||.
    """
    # eps of size (batch_size, channel (1), image_size, image_size)
    eps = F.conv2d(x, laplace_kernel, padding=1)

    if f is not None:
        eps = eps - f

    eps = eps * (1 - bc_mask)
    eps = eps.view(eps.size(0), -1)            # of size (batch_size, image_size ** 2)

    if reduction == 'norm':
        error = norm(eps)                      # of size (batch_size,)
    elif reduction == 'mean':
        error = torch.abs(eps).mean(dim=1)     # of size (batch_size,)
    elif reduction == 'max':
        error = torch.abs(eps).max(dim=1)[0]   # of size (batch_size,)
    elif reduction == 'none':
        error = -eps                           # of size (batch_size, image_size ** 2)
    else:
        raise NotImplementedError

    return error


def relative_residue(x: torch.Tensor,
                     bc_value: torch.Tensor,
                     bc_mask: torch.Tensor,
                     f: typing.Optional[torch.Tensor]) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    For a linear system Ax = f, the relative residual error eps = ||f - Ax|| / ||f||.
    :return: abs_residual_error, relative_residual_error
    """
    numerator: torch.Tensor = absolute_residue(x, bc_mask, f, reduction='norm')  # norm of size (batch_size,)

    denominator: torch.Tensor = bc_value                                         # (batch_size, image_size, image_size)

    if f is not None:
        denominator = denominator + f                                            # (batch_size, image_size, image_size)

    denominator = norm(denominator)

    return numerator, numerator / denominator                                    # (batch_size,)
