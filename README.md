# heat_pinn
Physics-Informed Neural Networks for solving the 1D heat equation

A complete implementation of Physics-Informed Neural Networks (PINNs) to solve the 1D heat equation with time-varying boundary conditions using TensorFlow 2.x. 

Problem Statement:
We solve the following PDE on the domain x ∈ [0,1], t ∈ [0,1]:

∂u/∂t - ∂²u/∂x² = f(x,t)

Force Term: 
f(x,t) = (π² - (π/2)²)·sin(πx)·exp(-(π/2)²t) + 0.6π·x·cos(2πt)

Initial Condition:
u(x,0) = sin(πx)

Boundary Conditions:
u(0,t) = 0
u(1,t) = 0.3·sin(2πt)

Exact Solution:
u(x,t) = sin(πx)·exp(-(π/2)²t) + 0.3·x·sin(2πt)

This problem combines exponential decay (heat diffusion) with oscillatory boundary forcing, creating spatiotemporal dynamics.

🎯 Key Features

✅ Mesh-free solution using neural networks

✅ Automatic differentiation for computing PDE residuals

✅ Physics-informed loss function (no labeled solution data needed)

✅ Achieves L2 error < 0.02 on test grid

✅ Complete visualization suite (3D plots, snapshots, loss curves)

✅ Reproducible results with fixed random seeds

🚀 Quick Start
Prerequisites

python >= 3.7
tensorflow >= 2.4.0
numpy >= 1.19.0
matplotlib >= 3.3.0


Train the PINN for 8000 epochs (a few minutes on CPU)

📊 Results: 
After training, you should see:

Mean Squared Error (MSE):  3.19e-03, 
L2 Error:                  1.80e-02, 
L∞ Error:                  4.67e-02, 

Generated Plots: 

3d_solutions_seed42.png - Comparison of exact vs PINN solution
solution_snapshots_seed42.png - Solution profiles at t=0.2, 0.4, 0.6, 0.8
loss_curves_seed42.png - Training dynamics (PDE, IC, BC losses)
