import numpy as np
from scipy.optimize import minimize

# ==========================================
# 1. PHYSICAL CONSTANTS & WORKING VARIABLES
# ==========================================
e = 1.602176634e-19
m_p = 1.67262192369e-27
m_n = 1.67492749804e-27
c = 299792458.0
h = 6.62607015e-34
mu0 = 4e-7 * np.pi

# Network Currents (0.324 coherence deficit applied)
I_p = e * (m_p * c**2 / h)
I_n = e * (m_n * c**2 / h) * 0.676  

# ==========================================
# 2. CLOSED-LOOP NEUMANN MATRIX ENGINE
# ==========================================
def generate_loop_points(center, tilt, yaw, radius=0.85e-15, steps=16):
    t = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    base_points = np.zeros((steps, 3))
    base_points[:, 0] = radius * np.cos(t)
    base_points[:, 1] = radius * np.sin(t)
    
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    
    R_tilt = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_t, -sin_t],
        [0.0, sin_t, cos_t]
    ])
    R_yaw = np.array([
        [cos_y, -sin_y, 0.0],
        [sin_y, cos_y, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    return np.dot(base_points, np.dot(R_yaw, R_tilt).T) + center

def calculate_mutual_inductance(loop1, loop2):
    dl1 = np.empty_like(loop1)
    dl1[:-1] = loop1[1:] - loop1[:-1]
    dl1[-1] = loop1[0] - loop1[-1]  # Corrected vector array indexing [0]
    
    dl2 = np.empty_like(loop2)
    dl2[:-1] = loop2[1:] - loop2[:-1]
    dl2[-1] = loop2[0] - loop2[-1]  # Corrected vector array indexing [0]
    
    diff = loop1[:, np.newaxis, :] - loop2[np.newaxis, :, :]
    r12 = np.linalg.norm(diff, axis=2)
    
    core_buffer = 0.1e-15
    r12 = np.where(r12 < core_buffer, core_buffer, r12)
    
    return (mu0 / (4 * np.pi)) * np.sum(np.dot(dl1, dl2.T) / r12)

# ==========================================
# 3. SPATIAL MACRO-TETRAHEDRON GEOMETRY
# ==========================================
def setup_oxygen16_geometry(R_aa=2.8e-15, r_loop=0.85e-15):
    s = R_aa / np.sqrt(2)
    alpha_centers = np.array([
        [ s/2,  0.0, -s/(2*np.sqrt(2))],
        [-s/2,  0.0, -s/(2*np.sqrt(2))],
        [ 0.0,  s/2,  s/(2*np.sqrt(2))],
        [ 0.0, -s/2,  s/(2*np.sqrt(2))]
    ])
    
    d = r_loop / np.sqrt(2)
    loop_offsets = np.array([
        [ d/2,  0.0, -d/(2*np.sqrt(2))], 
        [-d/2,  0.0, -d/(2*np.sqrt(2))], 
        [ 0.0,  d/2,  d/(2*np.sqrt(2))], 
        [ 0.0, -d/2,  d/(2*np.sqrt(2))]  
    ])
    
    loop_centers, loop_types = [], []
    for a_center in alpha_centers:
        for idx, offset in enumerate(loop_offsets):
            loop_centers.append(a_center + offset)
            loop_types.append(0 if idx < 2 else 1)
            
    return np.array(loop_centers), np.array(loop_types)

def calculate_raw_network_energy(angles, centers, identities):
    num_loops = len(centers)
    loops = [generate_loop_points(centers[i], angles[2*i], angles[2*i+1]) for i in range(num_loops)]
    
    total_energy_joules = 0.0
    for i in range(num_loops):
        I_i = I_p if identities[i] == 0 else I_n
        for j in range(i + 1, num_loops):
            I_j = I_p if identities[j] == 0 else I_n
            total_energy_joules += calculate_mutual_inductance(loops[i], loops[j]) * I_i * I_j
            
    return total_energy_joules / 1.602176634e-13

# ==========================================
# 4. EXECUTION ENGINE WITH STEP CALLBACK
# ==========================================
if __name__ == "__main__":
    centers, identities = setup_oxygen16_geometry()
    
    static_angles = np.zeros(32)
    base_raw_energy = calculate_raw_network_energy(static_angles, centers, identities)
    
    bounds = [(-np.pi/12, np.pi/12) for _ in range(32)]
    np.random.seed(42)
    initial_angles = np.random.uniform(-0.01, 0.01, 32)
    
    iteration_counter = 0
    
    def print_step_callback(xk):
        global iteration_counter
        iteration_counter += 1
        current_energy = calculate_raw_network_energy(xk, centers, identities)
        delta_so_far = base_raw_energy - current_energy
        print(f" -> Iteration {iteration_counter:02d} | Current System Energy: {current_energy:.4f} MeV | Rel. Shift: {delta_so_far:.4f} MeV")

    print("Launching Bounded Network Solver...")
    print(f"Static Starting Grid Energy Baseline: {base_raw_energy:.4f} MeV")
    print("----------------------------------------------------------------------")
    
    result = minimize(
        lambda xl: calculate_raw_network_energy(xl, centers, identities), 
        initial_angles, 
        method='L-BFGS-B', 
        bounds=bounds, 
        callback=print_step_callback
    )
    
    relaxed_raw_energy = result.fun
    raw_delta_energy = base_raw_energy - relaxed_raw_energy
    derived_kappa = 127.6200 / raw_delta_energy
    
    print("----------------------------------------------------------------------")
    print("\n=== FINAL MANUSCRIPT RESULTS ===")
    print(f"Convergence Verification Status: {result.success}")
    print(f"Total Iterations Executed: {result.nit}")
    print(f"Static Grid Energy (U_0): {base_raw_energy:.4f} MeV")
    print(f"Relaxed Grid Energy (U_min): {relaxed_raw_energy:.4f} MeV")
    print(f"Symmetry Breaking Energy Yield (Delta E): {raw_delta_energy:.4f} MeV")
    print(f"Mathematically Derived Phase-Lock Multiplier (kappa): {derived_kappa:.4f}")
