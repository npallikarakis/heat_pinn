"""
Physics-Informed Neural Network (PINN) for Heat Equation - PyTorch version
Exact translation of the TensorFlow implementation for fair comparison.
All hyperparameters, seeds, data splits, and evaluation are identical.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set seeds for reproducibility (matches TF version)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import time
start_time = time.time()

# ============================================================================
# SECTION 1: Problem Definition (identical to TF)
# ============================================================================

PI = np.pi
CONST = PI / 2
AC_AMPLITUDE = 0.3

def exact_solution(x, t):
    x = np.asarray(x, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    const_sq = CONST ** 2
    exponential_part = np.sin(PI * x) * np.exp(-const_sq * t)
    oscillatory_part = AC_AMPLITUDE * x * np.sin(2 * PI * t)
    return exponential_part + oscillatory_part

def forcing_term(x, t):
    x = np.asarray(x, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    const_sq = CONST ** 2
    pi_sq = PI ** 2
    f1 = (pi_sq - const_sq) * np.sin(PI * x) * np.exp(-const_sq * t)
    f2 = 2 * PI * AC_AMPLITUDE * x * np.cos(2 * PI * t)
    return f1 + f2

def initial_condition(x):
    return np.sin(PI * x)

def boundary_condition(x_boundary, t):
    if x_boundary == 0:
        return 0.0
    else:  # x_boundary == 1
        return float(AC_AMPLITUDE * np.sin(2 * PI * t))

# ============================================================================
# SECTION 2: Training Data Generation (identical to TF)
# ============================================================================

def generate_training_data(n_interior=3000, n_ic=150, n_bc=10, n_bc_grid=5, seed=42):
    np.random.seed(seed)
    X_interior = np.random.rand(n_interior, 2).astype(np.float32)

    # Initial condition points at t=0
    x_ic = np.random.rand(n_ic, 1).astype(np.float32)
    t_ic = np.zeros((n_ic, 1), dtype=np.float32)
    X_ic = np.hstack([x_ic, t_ic])
    y_ic = initial_condition(x_ic).astype(np.float32)

    # Boundary points: grid + random
    n_bc_random = n_bc - 2 * n_bc_grid

    # BC at x=0
    t_bc_grid_0 = np.linspace(0, 1, n_bc_grid).astype(np.float32).reshape(-1, 1)
    t_bc_random_0 = np.random.rand(n_bc_random, 1).astype(np.float32)
    t_bc_combined_0 = np.vstack([t_bc_grid_0, t_bc_random_0])
    X_bc0 = np.hstack([np.zeros((len(t_bc_combined_0), 1), dtype=np.float32), t_bc_combined_0])
    y_bc0 = np.array([boundary_condition(0, t) for t in t_bc_combined_0[:, 0]]).reshape(-1, 1).astype(np.float32)

    # BC at x=1
    t_bc_grid_1 = np.linspace(0, 1, n_bc_grid).astype(np.float32).reshape(-1, 1)
    t_bc_random_1 = np.random.rand(n_bc_random, 1).astype(np.float32)
    t_bc_combined_1 = np.vstack([t_bc_grid_1, t_bc_random_1])
    X_bc1 = np.hstack([np.ones((len(t_bc_combined_1), 1), dtype=np.float32), t_bc_combined_1])
    y_bc1 = np.array([boundary_condition(1, t) for t in t_bc_combined_1[:, 0]]).reshape(-1, 1).astype(np.float32)

    X_bc = np.vstack([X_bc0, X_bc1])
    y_bc = np.vstack([y_bc0, y_bc1])

    return X_interior, X_ic, y_ic, X_bc, y_bc

# ============================================================================
# SECTION 3: Neural Network Architecture (identical to TF)
# ============================================================================

class PINN(nn.Module):
    def __init__(self, hidden_layers=3, hidden_units=64):
        super(PINN, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(2, hidden_units))
        layers.append(nn.Tanh())
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        # Output layer
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)

        # Xavier (Glorot) uniform initialization like TF's GlorotUniform
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

# ============================================================================
# SECTION 4: Training Loop (mimics TF's training, same output format)
# ============================================================================

def train_pinn(model, X_interior, X_ic, y_ic, X_bc, y_bc,
               epochs=8000, lr_initial=2e-4, lr_final=1e-5,
               weight_ic=10.0, weight_bc=10.0, verbose_freq=500):
    # Convert numpy arrays to torch tensors
    X_int_t = torch.tensor(X_interior, requires_grad=True, dtype=torch.float32)
    X_ic_t = torch.tensor(X_ic, dtype=torch.float32)
    y_ic_t = torch.tensor(y_ic, dtype=torch.float32)
    X_bc_t = torch.tensor(X_bc, dtype=torch.float32)
    y_bc_t = torch.tensor(y_bc, dtype=torch.float32)

    # Extract x and t from interior points (for forcing term)
    x_int = X_int_t[:, 0].reshape(-1, 1)
    t_int = X_int_t[:, 1].reshape(-1, 1)
    f_val = forcing_term(x_int.detach().numpy(), t_int.detach().numpy())
    f_t = torch.tensor(f_val, dtype=torch.float32)

    # Move all data to the same device as the model
    device = next(model.parameters()).device
    X_int_t = X_int_t.to(device)
    x_int = x_int.to(device)
    t_int = t_int.to(device)
    f_t = f_t.to(device)
    X_ic_t = X_ic_t.to(device)
    y_ic_t = y_ic_t.to(device)
    X_bc_t = X_bc_t.to(device)
    y_bc_t = y_bc_t.to(device)

    # Polynomial decay schedule (exactly as in TF)
    def lr_lambda(epoch):
        decay_steps = 5000
        power = 0.5
        if epoch >= decay_steps:
            return lr_final / lr_initial
        return (1 - epoch / decay_steps) ** power

    optimizer = optim.Adam(model.parameters(), lr=lr_initial)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        'epoch': [],
        'loss_total': [],
        'loss_pde': [],
        'loss_ic': [],
        'loss_bc': [],
        'l2_error': []
    }

    print("\nTraining PINN...")
    print(f"Loss = 1.0×PDE + {weight_ic}×IC + {weight_bc}×BC\n")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # PDE residual using automatic differentiation
        u = model(torch.cat([x_int, t_int], dim=1))
        # First derivatives
        u_x = torch.autograd.grad(u, x_int, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t_int, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # Second derivative
        u_xx = torch.autograd.grad(u_x, x_int, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        loss_pde = torch.mean((u_t - u_xx - f_t) ** 2)

        # Initial condition loss
        u_ic = model(X_ic_t)
        loss_ic = torch.mean((u_ic - y_ic_t) ** 2)

        # Boundary condition loss
        u_bc = model(X_bc_t)
        loss_bc = torch.mean((u_bc - y_bc_t) ** 2)

        loss_total = loss_pde + weight_ic * loss_ic + weight_bc * loss_bc

        loss_total.backward()
        # Gradient clipping (same as TF: clip_by_global_norm with 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Record history
        history['epoch'].append(epoch)
        history['loss_total'].append(loss_total.item())
        history['loss_pde'].append(loss_pde.item())
        history['loss_ic'].append(loss_ic.item())
        history['loss_bc'].append(loss_bc.item())

        # Periodic evaluation (identical to TF)
        if epoch % verbose_freq == 0:
            # Generate test grid
            test_grid_x = np.linspace(0, 1, 150)
            test_grid_t = np.linspace(0.01, 1.0, 80)
            X_test, T_test = np.meshgrid(test_grid_x, test_grid_t)
            dx = test_grid_x[1] - test_grid_x[0]
            dt = test_grid_t[1] - test_grid_t[0]
            X_test_flat = np.column_stack([X_test.flatten(), T_test.flatten()]).astype(np.float32)
            X_test_tensor = torch.tensor(X_test_flat, device=device)

            with torch.no_grad():
                u_pred = model(X_test_tensor).cpu().numpy().flatten()
            u_exact = exact_solution(X_test.flatten(), T_test.flatten())
            l2_error = np.sqrt(np.sum((u_exact - u_pred) ** 2) * dx * dt)
            history['l2_error'].append(l2_error)

            print(f"Epoch {epoch:5d} | Total Loss: {loss_total:.4e} | "
                  f"PDE: {loss_pde:.4e} | IC: {loss_ic:.4e} | BC: {loss_bc:.4e} | "
                  f"L2 Error: {l2_error:.4e}")

    print("\nTraining complete!")
    return history

# ============================================================================
# SECTION 5: Visualization Functions (identical filenames to TF)
# ============================================================================

def plot_3d_solutions(model, seed=42):
    test_grid_x = np.linspace(0, 1, 150)
    test_grid_t = np.linspace(0.01, 1.0, 80)
    X_3d, T_3d = np.meshgrid(test_grid_x, test_grid_t)
    X_flat = np.column_stack([X_3d.flatten(), T_3d.flatten()]).astype(np.float32)

    device = next(model.parameters()).device
    X_tensor = torch.tensor(X_flat, device=device)
    with torch.no_grad():
        U_pred = model(X_tensor).cpu().numpy().reshape(X_3d.shape)
    U_exact = exact_solution(X_3d, T_3d)
    E_abs = np.abs(U_exact - U_pred)

    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X_3d, T_3d, U_exact, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('t', fontsize=11)
    ax1.set_zlabel('u(x,t)', fontsize=11)
    ax1.set_title('Exact Solution', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, pad=0.12, shrink=0.8)

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X_3d, T_3d, U_pred, cmap='viridis', edgecolor='none', alpha=0.9)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('t', fontsize=11)
    ax2.set_zlabel('u(x,t)', fontsize=11)
    ax2.set_title('PINN Prediction', fontsize=12, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    fig.colorbar(surf2, ax=ax2, pad=0.12, shrink=0.8)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X_3d, T_3d, E_abs, cmap='hot', edgecolor='none', alpha=0.9)
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('t', fontsize=11)
    ax3.set_zlabel('|Error|', fontsize=11)
    ax3.set_title('Absolute Error', fontsize=12, fontweight='bold')
    ax3.view_init(elev=25, azim=45)
    fig.colorbar(surf3, ax=ax3, pad=0.12, shrink=0.8)

    plt.suptitle(f'PINN Solution Comparison (Seed {seed})', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    filename = f'3d_solutions_seed{seed}_pytorch.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved 3D solutions: {filename}")
    plt.close()

def plot_loss_curves(history, seed=42):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].semilogy(history['epoch'], history['loss_pde'], 'b-', linewidth=2.5)
    axes[0, 0].set_title('PDE Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(history['epoch'], history['loss_ic'], 'g-', linewidth=2.5)
    axes[0, 1].set_title('IC Loss (weight=10)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(history['epoch'], history['loss_bc'], 'm-', linewidth=2.5)
    axes[1, 0].set_title('BC Loss (weight=10)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Loss', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(history['epoch'], history['loss_total'], 'r-', linewidth=2.5)
    axes[1, 1].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Loss', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Loss Curves (Seed {seed})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f'loss_curves_seed{seed}_pytorch.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✓ Saved loss curves: {filename}")
    plt.close()

def plot_solution_snapshots(model, seed=42):
    test_times = [0.2, 0.4, 0.6, 0.8]
    x_plot = np.linspace(0, 1, 150)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    device = next(model.parameters()).device

    for idx, t_test in enumerate(test_times):
        ax = axes[idx // 2, idx % 2]
        X = np.column_stack([x_plot, np.full_like(x_plot, t_test)]).astype(np.float32)
        X_tensor = torch.tensor(X, device=device)
        with torch.no_grad():
            u_pred = model(X_tensor).cpu().numpy().flatten()
        u_exact = exact_solution(x_plot, t_test)

        ax.plot(x_plot, u_exact, 'k-', linewidth=3, label='Exact', zorder=3)
        ax.plot(x_plot, u_pred, 'r--', linewidth=2.5, label='PINN', alpha=0.8)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel(f'u(x, {t_test})', fontsize=11)
        ax.set_title(f'Solution at t = {t_test}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Solution Snapshots (Seed {seed})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f'solution_snapshots_seed{seed}_pytorch.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved snapshots: {filename}")
    plt.close()

# ============================================================================
# SECTION 6: Main Execution (identical output to TF)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Physics-Informed Neural Network for Heat Equation - PyTorch")
    print("=" * 70)

    SEED = 42

    # Generate training data
    print("\nGenerating training data...")
    X_interior, X_ic, y_ic, X_bc, y_bc = generate_training_data(
        n_interior=3000,
        n_ic=150,
        n_bc=20,
        n_bc_grid=5,
        seed=SEED
    )
    print(f"  Interior points: {X_interior.shape[0]}")
    print(f"  Initial condition points: {X_ic.shape[0]}")
    print(f"  Boundary condition points: {X_bc.shape[0]}")

    # Create model
    print("\nCreating neural network...")
    model = PINN(hidden_layers=3, hidden_units=64)
    # Print model summary (PyTorch doesn't have a built-in Keras-like summary, but we can show total parameters)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")

    # Train model
    history = train_pinn(
        model, X_interior, X_ic, y_ic, X_bc, y_bc,
        epochs=8000,
        lr_initial=2e-4,
        lr_final=1e-5,
        weight_ic=10.0,
        weight_bc=10.0,
        verbose_freq=500
    )

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_3d_solutions(model, seed=SEED)
    plot_loss_curves(history, seed=SEED)
    plot_solution_snapshots(model, seed=SEED)

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    test_grid_x = np.linspace(0, 1, 150)
    test_grid_t = np.linspace(0.01, 1.0, 80)
    X_grid, T_grid = np.meshgrid(test_grid_x, test_grid_t)
    dx = test_grid_x[1] - test_grid_x[0]
    dt = test_grid_t[1] - test_grid_t[0]
    X_flat = np.column_stack([X_grid.flatten(), T_grid.flatten()]).astype(np.float32)
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X_flat, device=device)

    with torch.no_grad():
        u_pred = model(X_tensor).cpu().numpy().flatten()
    u_exact = exact_solution(X_flat[:, 0], X_flat[:, 1])
    error = u_exact - u_pred
    mse = np.mean(error ** 2)
    l2_error = np.sqrt(np.sum(error ** 2) * dx * dt)
    linf_error = np.max(np.abs(error))

    print(f"  Mean Squared Error (MSE):  {mse:.6e}")
    print(f"  L2 Error:                  {l2_error:.6e}")
    print(f"  L∞ Error:                  {linf_error:.6e}")
    print("=" * 70)
    print("\n PINN Training Complete!\n")