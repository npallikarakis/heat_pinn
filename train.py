"""
Physics-Informed Neural Network (PINN) for Heat Equation
========================================================

Problem:
    PDE: ∂u/∂t - ∂²u/∂x² = f(x,t)
    Domain: x ∈ [0,1], t ∈ [0,1]

    Exact solution: u(x,t) = sin(πx)·exp(-(π/2)²t) + 0.3·x·sin(2πt)

    Initial condition: u(x,0) = sin(πx)
    Boundary conditions:
        u(0,t) = 0
        u(1,t) = 0.3·sin(2πt)

Author: Nikolaos Pallikarakis
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# SECTION 1: Problem Definition
# ============================================================================

PI = np.pi
CONST = PI / 2
AC_AMPLITUDE = 0.3

def exact_solution(x, t):
    """
    Exact solution to the heat equation.

    u(x,t) = sin(πx)·exp(-(π/2)²t) + 0.3·x·sin(2πt)

    Args:
        x: spatial coordinate
        t: time coordinate

    Returns:
        u(x,t): exact solution value
    """
    x = np.asarray(x, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)

    const_sq = CONST ** 2
    exponential_part = np.sin(PI * x) * np.exp(-const_sq * t)
    oscillatory_part = AC_AMPLITUDE * x * np.sin(2 * PI * t)

    return exponential_part + oscillatory_part


def forcing_term(x, t):
    """
    Forcing term f(x,t) from PDE: ∂u/∂t - ∂²u/∂x² = f

    Args:
        x: spatial coordinate
        t: time coordinate

    Returns:
        f(x,t): forcing term value
    """
    x = np.asarray(x, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)

    const_sq = CONST ** 2
    pi_sq = PI ** 2

    # From exponential part
    f1 = (pi_sq - const_sq) * np.sin(PI * x) * np.exp(-const_sq * t)

    # From oscillatory part
    f2 = 2 * PI * AC_AMPLITUDE * x * np.cos(2 * PI * t)

    return f1 + f2


def initial_condition(x):
    """Initial condition: u(x,0) = sin(πx)"""
    x = np.asarray(x, dtype=np.float32)
    return np.sin(PI * x)


def boundary_condition(x_boundary, t):
    """
    Boundary conditions:
        u(0,t) = 0
        u(1,t) = 0.3·sin(2πt)
    """
    t = np.asarray(t, dtype=np.float32)
    if x_boundary == 0:
        return 0.0
    elif x_boundary == 1:
        return float(AC_AMPLITUDE * np.sin(2 * PI * t))
    else:
        raise ValueError("x_boundary must be 0 or 1")


# ============================================================================
# SECTION 2: Training Data Generation
# ============================================================================

def generate_training_data(n_interior=3000, n_ic=150, n_bc=10, n_bc_grid=5, seed=42):
    """
    Generate collocation points for PINN training.

    Args:
        n_interior: number of interior points for PDE residual
        n_ic: number of initial condition points
        n_bc: total number of boundary points (per boundary)
        n_bc_grid: number of grid points in time for BC
        seed: random seed

    Returns:
        X_interior: interior collocation points [x, t]
        X_ic: initial condition points [x, 0]
        y_ic: initial condition values
        X_bc: boundary condition points [x, t]
        y_bc: boundary condition values
    """
    np.random.seed(seed)

    # Interior points: random sampling in [0,1] x [0,1]
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

    # Combine boundary points
    X_bc = np.vstack([X_bc0, X_bc1])
    y_bc = np.vstack([y_bc0, y_bc1])

    return X_interior, X_ic, y_ic, X_bc, y_bc


# ============================================================================
# SECTION 3: Neural Network Architecture
# ============================================================================

def create_pinn_model(hidden_layers=3, hidden_units=64, seed=42):
    """
    Create a feedforward neural network for PINN.

    Architecture:
        Input: [x, t] (2 features)
        Hidden: 3 layers × 64 units with tanh activation
        Output: u(x,t) (1 value)

    Args:
        hidden_layers: number of hidden layers
        hidden_units: units per hidden layer
        seed: random seed for weight initialization

    Returns:
        model: Keras Sequential model
    """
    import random
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    layers = []

    # Input layer
    layers.append(tf.keras.layers.Dense(
        hidden_units,
        activation='tanh',
        input_shape=(2,),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()
    ))

    # Hidden layers
    for i in range(hidden_layers - 1):
        layers.append(tf.keras.layers.Dense(
            hidden_units,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + i + 1),
            bias_initializer=tf.keras.initializers.Zeros()
        ))

    # Output layer
    layers.append(tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + hidden_layers + 1),
        bias_initializer=tf.keras.initializers.Zeros()
    ))

    return tf.keras.Sequential(layers)


# ============================================================================
# SECTION 4: Training Loop with Physics Loss
# ============================================================================

def train_pinn(model, X_interior, X_ic, y_ic, X_bc, y_bc,
               epochs=8000, lr_initial=2e-4, lr_final=1e-5,
               weight_ic=10.0, weight_bc=10.0, verbose_freq=500):
    """
    Train the PINN model.

    Total Loss = 1.0 × L_PDE + weight_ic × L_IC + weight_bc × L_BC

    The PDE residual is computed using automatic differentiation:
        ∂u/∂t - ∂²u/∂x² - f(x,t) = 0

    Args:
        model: neural network
        X_interior, X_ic, y_ic, X_bc, y_bc: training data
        epochs: number of training iterations
        lr_initial: initial learning rate
        lr_final: final learning rate
        weight_ic: weight for initial condition loss
        weight_bc: weight for boundary condition loss
        verbose_freq: print frequency

    Returns:
        history: dictionary with training history
    """
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=lr_initial,
        decay_steps=5000,
        end_learning_rate=lr_final,
        power=0.5
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    # Convert to TensorFlow constants
    X_int_tf = tf.constant(X_interior, dtype=tf.float32)
    X_ic_tf = tf.constant(X_ic, dtype=tf.float32)
    y_ic_tf = tf.constant(y_ic, dtype=tf.float32)
    X_bc_tf = tf.constant(X_bc, dtype=tf.float32)
    y_bc_tf = tf.constant(y_bc, dtype=tf.float32)

    # Training history
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
        # Use persistent tape for computing second-order derivatives
        with tf.GradientTape(persistent=True) as tape:
            # Extract x and t from interior points
            x = tf.reshape(X_int_tf[:, 0], (-1, 1))
            t = tf.reshape(X_int_tf[:, 1], (-1, 1))
            tape.watch([x, t])

            # Forward pass through network
            u = model(tf.concat([x, t], axis=1), training=True)

            # Compute derivatives using automatic differentiation
            u_x = tape.gradient(u, x)      # ∂u/∂x
            u_xx = tape.gradient(u_x, x)   # ∂²u/∂x²
            u_t = tape.gradient(u, t)      # ∂u/∂t

            # Compute forcing term
            f = forcing_term(X_interior[:, 0], X_interior[:, 1])
            f_tf = tf.constant(f.reshape(-1, 1), dtype=tf.float32)

            # PDE residual: ∂u/∂t - ∂²u/∂x² - f(x,t)
            loss_pde = tf.reduce_mean(tf.square(u_t - u_xx - f_tf))

            # Initial condition loss
            loss_ic = tf.reduce_mean(tf.square(model(X_ic_tf, training=True) - y_ic_tf))

            # Boundary condition loss
            loss_bc = tf.reduce_mean(tf.square(model(X_bc_tf, training=True) - y_bc_tf))

            # Total weighted loss
            loss_total = (
                1.0 * loss_pde +
                weight_ic * loss_ic +
                weight_bc * loss_bc
            )

        # Compute gradients and update weights
        grads = tape.gradient(loss_total, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        del tape

        # Record history
        history['epoch'].append(epoch)
        history['loss_total'].append(float(loss_total.numpy()))
        history['loss_pde'].append(float(loss_pde.numpy()))
        history['loss_ic'].append(float(loss_ic.numpy()))
        history['loss_bc'].append(float(loss_bc.numpy()))

        # Periodic evaluation
        if epoch % verbose_freq == 0:
            # Compute L2 error on test grid
            test_grid_x = np.linspace(0, 1, 150)
            test_grid_t = np.linspace(0.01, 1.0, 80)
            X_test, T_test = np.meshgrid(test_grid_x, test_grid_t)
            dx = test_grid_x[1] - test_grid_x[0]
            dt = test_grid_t[1] - test_grid_t[0]
            X_test_flat = X_test.flatten()
            T_test_flat = T_test.flatten()

            u_exact = exact_solution(X_test_flat, T_test_flat)
            u_pred = model(np.stack([X_test_flat, T_test_flat], axis=1), training=False).numpy().flatten()

            l2_error = np.sqrt(np.sum((u_exact - u_pred) ** 2) * dx * dt)
            history['l2_error'].append(l2_error)

            print(f"Epoch {epoch:5d} | Total Loss: {loss_total:.4e} | "
                  f"PDE: {loss_pde:.4e} | IC: {loss_ic:.4e} | BC: {loss_bc:.4e} | "
                  f"L2 Error: {l2_error:.4e}")

    print("\nTraining complete!")
    return history


# ============================================================================
# SECTION 5: Visualization Functions
# ============================================================================

def plot_3d_solutions(model, seed=42):
    """Generate 3D surface plots comparing exact and PINN solutions."""

    test_grid_x = np.linspace(0, 1, 150)
    test_grid_t = np.linspace(0.01, 1.0, 80)

    X_3d, T_3d = np.meshgrid(test_grid_x, test_grid_t)
    X_flat = X_3d.flatten()
    T_flat = T_3d.flatten()
    X_input = np.column_stack([X_flat, T_flat]).astype(np.float32)

    U_exact = exact_solution(X_3d, T_3d)
    U_pred = model(X_input, training=False).numpy().reshape(X_3d.shape)
    E_abs = np.abs(U_exact - U_pred)

    fig = plt.figure(figsize=(20, 6))

    # Exact solution
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X_3d, T_3d, U_exact, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('t', fontsize=11)
    ax1.set_zlabel('u(x,t)', fontsize=11)
    ax1.set_title('Exact Solution', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, pad=0.12, shrink=0.8)

    # PINN prediction
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X_3d, T_3d, U_pred, cmap='viridis', edgecolor='none', alpha=0.9)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('t', fontsize=11)
    ax2.set_zlabel('u(x,t)', fontsize=11)
    ax2.set_title('PINN Prediction', fontsize=12, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    fig.colorbar(surf2, ax=ax2, pad=0.12, shrink=0.8)

    # Absolute error
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

    filename = f'3d_solutions_seed{seed}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved 3D solutions: {filename}")
    plt.close()


def plot_loss_curves(history, seed=42):
    """Generate loss curve plots."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # PDE Loss
    axes[0, 0].semilogy(history['epoch'], history['loss_pde'], 'b-', linewidth=2.5)
    axes[0, 0].set_title('PDE Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # IC Loss
    axes[0, 1].semilogy(history['epoch'], history['loss_ic'], 'g-', linewidth=2.5)
    axes[0, 1].set_title('IC Loss (weight=10)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # BC Loss
    axes[1, 0].semilogy(history['epoch'], history['loss_bc'], 'm-', linewidth=2.5)
    axes[1, 0].set_title('BC Loss (weight=10)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Loss', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Total Loss
    axes[1, 1].semilogy(history['epoch'], history['loss_total'], 'r-', linewidth=2.5)
    axes[1, 1].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Loss', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Loss Curves (Seed {seed})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'loss_curves_seed{seed}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✓ Saved loss curves: {filename}")
    plt.close()


def plot_solution_snapshots(model, seed=42):
    """Generate solution snapshots at different time points."""

    test_times = [0.2, 0.4, 0.6, 0.8]
    x_plot = np.linspace(0, 1, 150)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for idx, t_test in enumerate(test_times):
        ax = axes[idx // 2, idx % 2]
        X = np.column_stack([x_plot, np.full_like(x_plot, t_test)]).astype(np.float32)

        u_exact = exact_solution(x_plot, t_test)
        u_pred = model(X, training=False).numpy().flatten()

        ax.plot(x_plot, u_exact, 'k-', linewidth=3, label='Exact', zorder=3)
        ax.plot(x_plot, u_pred, 'r--', linewidth=2.5, label='PINN', alpha=0.8)

        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel(f'u(x, {t_test})', fontsize=11)
        ax.set_title(f'Solution at t = {t_test}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Solution Snapshots (Seed {seed})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'solution_snapshots_seed{seed}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved snapshots: {filename}")
    plt.close()


# ============================================================================
# SECTION 6: Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Physics-Informed Neural Network for Heat Equation")
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
    model = create_pinn_model(hidden_layers=3, hidden_units=64, seed=SEED)
    model.summary()

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

    u_exact = exact_solution(X_flat[:, 0], X_flat[:, 1])
    u_pred = model(X_flat, training=False).numpy().flatten()

    error = u_exact - u_pred
    mse = np.mean(error ** 2)
    l2_error = np.sqrt(np.sum(error ** 2) * dx * dt)
    linf_error = np.max(np.abs(error))

    print(f"  Mean Squared Error (MSE):  {mse:.6e}")
    print(f"  L2 Error:                  {l2_error:.6e}")
    print(f"  L∞ Error:                  {linf_error:.6e}")
    print("=" * 70)
    print("\n PINN Training Complete!\n")