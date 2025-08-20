import numpy as np
from scipy.optimize import linprog
from scipy.special import softmax

def allocate_frames_linear_cost(cost_matrix, domain_totals, frame_capacities=None):
    """
    Allocate sample counts per frame to minimize linear cost.

    Parameters
    ----------
    cost_matrix : 2D array (num_domains x num_frames)
        Cost c_{dq} of allocating domain d in frame q.
    domain_totals : 1D array (num_domains,)
        Total sample size required for each domain d.
    frame_capacities : 1D array (num_frames,), optional
        Maximum samples that can be allocated in each frame.

    Returns
    -------
    allocation : 2D array (num_domains x num_frames)
        Optimal allocation y_{dq}.
    """
    num_domains, num_frames = cost_matrix.shape

    # Flatten the decision variables y_{dq} into a 1D vector
    c = cost_matrix.flatten()

    # Equality constraints: sum_q y_{dq} = domain_totals[d]
    A_eq = np.zeros((num_domains, num_domains * num_frames))
    for d in range(num_domains):
        A_eq[d, d*num_frames:(d+1)*num_frames] = 1
    b_eq = domain_totals

    # Inequality constraints: optional frame capacities sum_d y_{dq} <= frame_capacities[q]
    if frame_capacities is not None:
        A_ub = np.zeros((num_frames, num_domains * num_frames))
        for q in range(num_frames):
            for d in range(num_domains):
                A_ub[q, d*num_frames + q] = 1
        b_ub = frame_capacities
    else:
        A_ub = None
        b_ub = None

    # Bounds: allocations must be >= 0
    bounds = [(0, None) for _ in range(num_domains * num_frames)]

    # Solve linear program
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        raise ValueError("Linear programming failed: " + res.message)

    # Reshape solution into (num_domains x num_frames)
    allocation = res.x.reshape((num_domains, num_frames))
    return allocation