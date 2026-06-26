import numpy as np

def calculate_topological_scaling(N, structural_type):
    """
    Illustrates the Section 3 Kuramoto scaling law numerically:
    kappa(N) = kappa_0 * (1 + gamma * lambda_max / N)
    """
    # 1. Build an idealized topological adjacency matrix A_ij for the network type
    A = np.zeros((N, N))
    
    if structural_type == "deuteron":
        # Simple single link between 2 loops
        A[0, 1] = A[1, 0] = 1.0
        
    elif structural_type == "alpha":
        # Fully connected micro-tetrahedron (4 nodes, every node links to all others)
        A = np.ones((N, N)) - np.eye(N)
        
    elif structural_type == "oxygen16":
        # Macro-tetrahedron of 4 alpha clusters (16 nodes total)
        # Internal alpha links are strong (1.0), cross-alpha cluster links are weaker (0.25)
        for i in range(16):
            for j in range(16):
                if i != j:
                    cluster_i = i // 4
                    cluster_j = j // 4
                    if cluster_i == cluster_j:
                        A[i, j] = 1.0  # Intense intra-alpha coupling
                    else:
                        A[i, j] = 0.25 # Distant inter-alpha coupling

    # 2. Extract the maximum eigenvalue (spectral radius) of the network topology
    eigenvalues = np.linalg.eigvals(A)
    lambda_max = np.max(np.real(eigenvalues))
    
    # 3. Apply the Section 3 Kuramoto scaling formula
    # Anchor values calibrated to base network complexity
    kappa_0 = 7.00  
    gamma = 0.5384
    
    kappa_predicted = kappa_0 * (1.0 + gamma * (lambda_max / N))
    return lambda_max, kappa_predicted

# Execute the comparative structural sweep
structures = [
    (2, "deuteron", "Linear Link"),
    (4, "alpha", "Micro-Tetrahedral Cluster"),
    (16, "oxygen16", "Macro-Tetrahedral Cluster")
]

print("================================================================")
print("NUMERICAL ILLUSTRATION OF THE TOPOLOGICAL KURAMOTO SCALING LAW")
print("================================================================")
for N, key, label in structures:
    l_max, k_pred = calculate_topological_scaling(N, key)
    print(f"Structure: {label:<26} | Nodes (N): {N:2d} | Max Eigenvalue: {l_max:.4f} | Predicted kappa: {k_pred:.4f}")
print("================================================================")
