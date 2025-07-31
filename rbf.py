import torch


def rbf_kernel(X, Y, gamma=1.0):
    """
    Compute the RBF kernel matrix between two sets of data points.

    Args:
    - X (torch.Tensor): Tensor of shape (N1, D) representing the first set of data points.
    - Y (torch.Tensor): Tensor of shape (N2, D) representing the second set of data points.
    - gamma (float): Parameter for the RBF kernel.

    Returns:
    - kernel_matrix (torch.Tensor): Computed RBF kernel matrix of shape (N1, N2).
    """
    # Compute squared Euclidean distance
    dist_matrix = torch.cdist(X, Y, p=2) ** 2

    # Compute RBF kernel matrix
    kernel_matrix = torch.exp(-gamma * dist_matrix)

    return kernel_matrix


# Example usage
if __name__ == "__main__":
    # Generate random data points
    N1, N2, D = 100, 80, 5
    X = torch.randn(N1, D)
    Y = torch.randn(N2, D)

    # Compute RBF kernel matrix
    gamma_val = 0.1
    K = rbf_kernel(X, Y, gamma=gamma_val)

    print("Shape of RBF Kernel Matrix:", K.shape)
