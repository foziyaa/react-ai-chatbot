# %% [markdown]
# # Convex Optimization in Artificial Intelligence
# ## A Comprehensive Guide with Support Vector Machines

# %% [markdown]
# <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; border-left:5px solid #4CAF50;">
# <h2 style="color:#2E86AB;">📚 Introduction to Convex Optimization</h2>
# <p style="color:black;">Convex optimization forms the mathematical backbone of many machine learning algorithms. Its importance stems from the guarantee of finding global optima, making it crucial for reliable AI systems.</p>
# </div>

# %% [markdown]
# ## 1. Convex Optimization Problem Setup

# %% [markdown]
# ### 1.1 Mathematical Formulation
# 
# A convex optimization problem has the general form:
# 
# $$
# \begin{align*}
# \min_{x} \quad & f_0(x) \\
# \text{subject to} \quad & f_i(x) \leq 0, \quad i = 1, \dots, m \\
# & h_j(x) = 0, \quad j = 1, \dots, p
# \end{align*}
# $$
# 
# where:
# - $f_0(x)$ is the **objective function** (must be convex)
# - $f_i(x)$ are **inequality constraints** (must be convex)
# - $h_j(x)$ are **equality constraints** (must be affine)

# %% [markdown]
# ### 1.2 What Makes a Function Convex?
# 
# **Definition**: A function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if for all $x, y \in \text{dom}(f)$ and $\theta \in [0,1]$:
# 
# $$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$
# 
# **Visual Interpretation**: The line segment between any two points on the function lies above or on the graph.

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# %% [markdown]
# ### 1.3 Visualizing Convex vs Non-Convex Functions

# %%
# Create visualization comparing convex and non-convex functions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Convex function example: f(x) = x²
x = np.linspace(-2, 2, 100)
convex_func = x**2
non_convex_func = x**4 - 2*x**2

ax1.plot(x, convex_func, 'b-', linewidth=3, label='Convex: $f(x) = x^2$')
ax1.fill_between(x, convex_func, convex_func.max(), alpha=0.2, color='blue')
ax1.set_title('Convex Function', fontsize=14, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Non-convex function example
ax2.plot(x, non_convex_func, 'r-', linewidth=3, label='Non-convex: $f(x) = x^4 - 2x^2$')
ax2.fill_between(x, non_convex_func, non_convex_func.max(), alpha=0.2, color='red')
ax2.set_title('Non-Convex Function', fontsize=14, fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.4 Why Convex Problems Guarantee Global Optima
# 
# **Key Properties**:
# 1. **No local minima**: Any local minimum is also a global minimum
# 2. **Convex feasible set**: The intersection of convex sets is convex
# 3. **Efficient algorithms**: Problems can be solved in polynomial time
# 4. **Reliable solutions**: Guaranteed convergence to optimal solution

# %%
# Demonstrate convex optimization landscape
def convex_3d_example():
    fig = plt.figure(figsize=(12, 5))
    
    # Convex bowl
    ax1 = fig.add_subplot(121, projection='3d')
    X = np.linspace(-2, 2, 50)
    Y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(X, Y)
    Z = X**2 + Y**2  # Convex function
    
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
    ax1.set_title('Convex Optimization Landscape\n(Single Global Minimum)', 
                 fontweight='bold', fontsize=12)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X,Y)')
    
    # Non-convex landscape
    ax2 = fig.add_subplot(122, projection='3d')
    Z_nonconvex = np.sin(X) + np.cos(Y) + 0.1*(X**2 + Y**2)
    
    surf2 = ax2.plot_surface(X, Y, Z_nonconvex, cmap=cm.coolwarm, alpha=0.8)
    ax2.set_title('Non-Convex Optimization Landscape\n(Multiple Local Minima)', 
                 fontweight='bold', fontsize=12)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('f(X,Y)')
    
    plt.tight_layout()
    plt.show()

convex_3d_example()

# %% [markdown]
# ## 2. Duality Concept in Convex Optimization

# %% [markdown]
# ### 2.1 Lagrangian Duality
# 
# For a constrained optimization problem:
# 
# $$
# \begin{align*}
# \min_x \quad & f_0(x) \\
# \text{s.t.} \quad & f_i(x) \leq 0, \quad i = 1, \dots, m \\
# & h_j(x) = 0, \quad j = 1, \dots, p
# \end{align*}
# $$
# 
# The **Lagrangian** is defined as:
# 
# $$L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$
# 
# where $\lambda_i \geq 0$ and $\nu_j$ are **Lagrange multipliers**.

# %% [markdown]
# ### 2.2 Primal and Dual Problems
# 
# - **Primal Problem**: $p^* = \min_x \max_{\lambda \geq 0, \nu} L(x, \lambda, \nu)$
# - **Dual Problem**: $d^* = \max_{\lambda \geq 0, \nu} \min_x L(x, \lambda, \nu)$
# 
# **Weak Duality**: $d^* \leq p^*$ (always holds)
# 
# **Strong Duality**: $d^* = p^*$ (holds under certain conditions like Slater's condition)

# %%
# Visualization of Duality Gap
def visualize_duality_gap():
    x = np.linspace(0.1, 3, 100)
    primal = x + 1/x  # Example primal objective
    dual = 2 * np.sqrt(1)  # Example dual objective (constant)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, primal, 'b-', linewidth=3, label='Primal Objective')
    plt.axhline(y=dual, color='r', linestyle='--', linewidth=3, label='Dual Objective')
    plt.fill_between(x, primal, dual, where=(primal >= dual), alpha=0.3, color='gray', label='Duality Gap')
    
    # Optimal point
    opt_x = 1.0
    opt_val = 2.0
    plt.plot(opt_x, opt_val, 'ko', markersize=10, label='Optimal Solution')
    
    plt.xlabel('x')
    plt.ylabel('Objective Value')
    plt.title('Duality Gap in Convex Optimization', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

visualize_duality_gap()

# %% [markdown]
# ### 2.3 Benefits of Duality
# 
# 1. **Simplification**: Dual problem might be easier to solve
# 2. **Optimality certificates**: Dual variables provide sensitivity information
# 3. **Kernel methods**: Enables use of kernel tricks in SVMs
# 4. **Decomposition**: Problems can be broken into smaller subproblems

# %% [markdown]
# ## 3. Implementation: Support Vector Machine (SVM) as Convex Optimization

# %% [markdown]
# ### 3.1 SVM Primal and Dual Formulations

# %% [markdown]
# **Primal Problem (Hard Margin)**:
# $$
# \begin{align*}
# \min_{w,b} \quad & \frac{1}{2}\|w\|^2 \\
# \text{s.t.} \quad & y_i(w^T x_i + b) \geq 1, \quad i = 1, \dots, n
# \end{align*}
# $$
# 
# **Dual Problem**:
# $$
# \begin{align*}
# \max_{\alpha} \quad & \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j \\
# \text{s.t.} \quad & \alpha_i \geq 0, \quad i = 1, \dots, n \\
# & \sum_{i=1}^n \alpha_i y_i = 0
# \end{align*}
# $$

# %%
# Import required libraries for SVM implementation
import cvxopt
from cvxopt import matrix, solvers
from sklearn.datasets import make_blobs, make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configure cvxopt to not show output
solvers.options['show_progress'] = False

# %% [markdown]
# ### 3.2 Generating Synthetic Dataset

# %%
# Create a linearly separable dataset
def generate_svm_data(n_samples=100, random_state=42):
    """Generate a 2D linearly separable dataset for SVM demonstration"""
    X, y = make_blobs(n_samples=n_samples, centers=2, 
                      n_features=2, random_state=random_state,
                      center_box=([-2, -2], [2, 2]))
    
    # Ensure labels are -1 and +1 for SVM
    y = 2 * y - 1
    
    return X, y

X, y = generate_svm_data(n_samples=50)
print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.unique(y, return_counts=True)}")

# Visualize the dataset
plt.figure(figsize=(10, 6))
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', s=50, 
           alpha=0.7, label='Class -1', edgecolors='black')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=50, 
           alpha=0.7, label='Class +1', edgecolors='black')
plt.title('Synthetic Dataset for SVM Classification', fontweight='bold', fontsize=14)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### 3.3 Implementing SVM Dual Problem using CVXOPT

# %%
def svm_dual_implementation(X, y):
    """
    Implement SVM using dual formulation with CVXOPT
    """
    n_samples, n_features = X.shape
    
    # Construct the quadratic programming matrices for SVM dual
    # P = y_i * y_j * x_i^T x_j
    P = matrix(np.outer(y, y) * (X @ X.T).astype(np.double))
    
    # q = -1 vector
    q = matrix(-np.ones(n_samples).astype(np.double))
    
    # G = -I (for alpha >= 0)
    G = matrix(-np.eye(n_samples).astype(np.double))
    
    # h = zero vector
    h = matrix(np.zeros(n_samples).astype(np.double))
    
    # A = y^T
    A = matrix(y.reshape(1, -1).astype(np.double))
    
    # b = 0
    b = matrix(0.0)
    
    # Solve the quadratic programming problem
    solution = solvers.qp(P, q, G, h, A, b)
    
    # Extract Lagrange multipliers
    alpha = np.array(solution['x']).flatten()
    
    # Compute weight vector w
    w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
    
    # Find support vectors (points with alpha > 0)
    sv_indices = alpha > 1e-5
    support_vectors = X[sv_indices]
    support_vector_labels = y[sv_indices]
    support_vector_alphas = alpha[sv_indices]
    
    # Compute bias b using support vectors
    b = 0
    for i in range(len(alpha)):
        if alpha[i] > 1e-5:
            b += y[i] - np.dot(w, X[i])
    b /= len(support_vectors)
    
    return w, b, support_vectors, support_vector_labels, support_vector_alphas, alpha

# %% [markdown]
# ### 3.4 Solving the SVM Optimization Problem

# %%
# Solve SVM using our dual implementation
w, b, support_vectors, sv_labels, sv_alphas, all_alphas = svm_dual_implementation(X, y)

print("=== SVM Dual Optimization Results ===")
print(f"Weight vector (w): {w}")
print(f"Bias term (b): {b:.4f}")
print(f"Number of support vectors: {len(support_vectors)}")
print(f"Support vector coefficients (alpha): {sv_alphas[:5]}")  # Show first 5

# %% [markdown]
# ### 3.5 Visualizing SVM Results

# %%
def plot_svm_result(X, y, w, b, support_vectors, title="SVM Classification Result"):
    """
    Plot SVM decision boundary, margins, and support vectors
    """
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', s=50, 
                alpha=0.7, label='Class -1', edgecolors='black')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=50, 
                alpha=0.7, label='Class +1', edgecolors='black')
    
    # Highlight support vectors
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                s=150, facecolors='none', edgecolors='yellow', 
                linewidths=2, label='Support Vectors')
    
    # Create decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx = np.linspace(x_min, x_max, 100)
    yy = np.linspace(y_min, y_max, 100)
    XX, YY = np.meshgrid(xx, yy)
    
    # Calculate decision function
    Z = w[0] * XX + w[1] * YY + b
    
    # Plot decision boundary and margins
    plt.contour(XX, YY, Z, levels=[-1, 0, 1], colors=['orange', 'black', 'orange'], 
                linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    # Fill decision regions
    plt.contourf(XX, YY, Z, levels=[Z.min(), 0, Z.max()], 
                 alpha=0.2, colors=['red', 'blue'])
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title, fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

# Plot our custom SVM results
plot_svm_result(X, y, w, b, support_vectors, 
                "Custom SVM Implementation - Dual Formulation")

# %% [markdown]
# ### 3.6 Understanding Support Vectors and Margins

# %%
# Detailed analysis of support vectors
print("=== Support Vector Analysis ===")
print(f"Total training samples: {len(X)}")
print(f"Number of support vectors: {len(support_vectors)}")
print(f"Support vector ratio: {len(support_vectors)/len(X):.2%}")

# Calculate margins for each support vector
margins = []
for i, sv in enumerate(support_vectors):
    margin = np.abs(w @ sv + b) / np.linalg.norm(w)
    margins.append(margin)

print(f"Average margin for support vectors: {np.mean(margins):.4f}")
print(f"Margin standard deviation: {np.std(margins):.4f}")

# %% [markdown]
# ## 4. Comparison: Custom SVM vs Scikit-learn SVC

# %% [markdown]
# ### 4.1 Implementing Scikit-learn SVM

# %%
# Train scikit-learn SVM for comparison
sklearn_svm = SVC(kernel='linear', C=1e10)  # Large C for hard margin
sklearn_svm.fit(X, y)

print("=== Scikit-learn SVM Results ===")
print(f"Scikit-learn weight vector: {sklearn_svm.coef_[0]}")
print(f"Scikit-learn bias: {sklearn_svm.intercept_[0]:.4f}")
print(f"Scikit-learn support vectors: {sklearn_svm.support_vectors_.shape[0]}")

# %% [markdown]
# ### 4.2 Comparative Analysis

# %%
def compare_svm_implementations(custom_w, custom_b, custom_sv, sklearn_svm, X, y):
    """
    Compare custom SVM implementation with scikit-learn
    """
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot custom SVM
    # Data points
    ax1.scatter(X[y == -1, 0], X[y == -1, 1], c='red', s=50, alpha=0.7, 
                label='Class -1', edgecolors='black')
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=50, alpha=0.7, 
                label='Class +1', edgecolors='black')
    
    # Support vectors
    ax1.scatter(custom_sv[:, 0], custom_sv[:, 1], s=150, facecolors='none', 
                edgecolors='yellow', linewidths=2, label='Support Vectors')
    
    # Decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    xx = np.linspace(x_min, x_max, 100)
    yy_custom = (-custom_w[0] * xx - custom_b) / custom_w[1]
    ax1.plot(xx, yy_custom, 'k-', linewidth=3, label='Decision Boundary')
    
    # Margins
    yy_margin1 = (-custom_w[0] * xx - custom_b + 1) / custom_w[1]
    yy_margin2 = (-custom_w[0] * xx - custom_b - 1) / custom_w[1]
    ax1.plot(xx, yy_margin1, 'orange', linestyle='--', linewidth=2, label='Margin')
    ax1.plot(xx, yy_margin2, 'orange', linestyle='--', linewidth=2)
    
    ax1.set_title('Custom SVM Implementation\n(Dual Formulation with CVXOPT)', 
                 fontweight='bold', fontsize=12)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot scikit-learn SVM
    # Data points
    ax2.scatter(X[y == -1, 0], X[y == -1, 1], c='red', s=50, alpha=0.7, 
                label='Class -1', edgecolors='black')
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=50, alpha=0.7, 
                label='Class +1', edgecolors='black')
    
    # Support vectors
    ax2.scatter(sklearn_svm.support_vectors_[:, 0], 
                sklearn_svm.support_vectors_[:, 1], 
                s=150, facecolors='none', edgecolors='yellow', 
                linewidths=2, label='Support Vectors')
    
    # Decision boundary and margins
    w_sklearn = sklearn_svm.coef_[0]
    b_sklearn = sklearn_svm.intercept_[0]
    yy_sklearn = (-w_sklearn[0] * xx - b_sklearn) / w_sklearn[1]
    ax2.plot(xx, yy_sklearn, 'k-', linewidth=3, label='Decision Boundary')
    
    yy_margin1_sk = (-w_sklearn[0] * xx - b_sklearn + 1) / w_sklearn[1]
    yy_margin2_sk = (-w_sklearn[0] * xx - b_sklearn - 1) / w_sklearn[1]
    ax2.plot(xx, yy_margin1_sk, 'orange', linestyle='--', linewidth=2, label='Margin')
    ax2.plot(xx, yy_margin2_sk, 'orange', linestyle='--', linewidth=2)
    
    ax2.set_title('Scikit-learn SVM Implementation\n(Optimized Production Code)', 
                 fontweight='bold', fontsize=12)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print quantitative comparison
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISON: CUSTOM SVM vs SCIKIT-LEARN")
    print("="*60)
    
    print(f"\nWeight Vectors:")
    print(f"Custom SVM:    [{custom_w[0]:.6f}, {custom_w[1]:.6f}]")
    print(f"Scikit-learn:  [{w_sklearn[0]:.6f}, {w_sklearn[1]:.6f}]")
    print(f"Difference:    [{abs(custom_w[0]-w_sklearn[0]):.6f}, {abs(custom_w[1]-w_sklearn[1]):.6f}]")
    
    print(f"\nBias Terms:")
    print(f"Custom SVM:    {custom_b:.6f}")
    print(f"Scikit-learn:  {b_sklearn:.6f}")
    print(f"Difference:    {abs(custom_b - b_sklearn):.6f}")
    
    print(f"\nSupport Vectors:")
    print(f"Custom SVM:    {len(custom_sv)} vectors")
    print(f"Scikit-learn:  {len(sklearn_svm.support_vectors_)} vectors")
    
    # Calculate predictions for both models
    custom_pred = np.sign(X @ custom_w + custom_b)
    sklearn_pred = sklearn_svm.predict(X)
    
    print(f"\nTraining Accuracy:")
    print(f"Custom SVM:    {accuracy_score(y, custom_pred):.4f}")
    print(f"Scikit-learn:  {accuracy_score(y, sklearn_pred):.4f}")

# Run comparison
compare_svm_implementations(w, b, support_vectors, sklearn_svm, X, y)

# %% [markdown]
# ## 5. Insights and Real-World Applications

# %% [markdown]
# ### 5.1 Why SVMs are Convex Optimization Problems
# 
# <div style="background-color:#e8f5e8; padding:15px; border-radius:8px; border-left:4px solid #4CAF50;">
# <h4 style="color:#2E7D32;">✅ Convexity Properties of SVM:</h4>
# <ul >
# <li style="color:#2E7D32;"><strong style >Convex Objective</strong>: The norm minimization ‖w‖² is strictly convex</li>
# <li style="color:#2E7D32;"><strong>Convex Constraints</strong>: Linear inequality constraints form a convex feasible set</li>
# <li style="color:#2E7D32;"><strong>Global Optimum Guarantee</strong>: No local minima - always finds the maximum margin hyperplane</li>
# <li style="color:#2E7D32;"><strong>Dual Formulation</strong>: Convex quadratic programming problem with linear constraints</li>
# </ul>
# </div>

# %% [markdown]
# ### 5.2 Real-World Impact of Convex Optimization in AI

# %%
# Create visualization of convex optimization applications
def plot_ai_applications():
    applications = {
        'Support Vector Machines': 'Classification, Regression',
        'Logistic Regression': 'Binary Classification',
        'Neural Networks': 'Convex Loss Functions',
        'Portfolio Optimization': 'Finance & Risk Management',
        'Control Systems': 'Robotics & Automation',
        'Signal Processing': 'Filter Design & Compression'
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(applications))
    
    ax.barh(y_pos, [0.9] * len(applications), align='center', 
            color=plt.cm.Set3(np.linspace(0, 1, len(applications))), alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(applications.keys())
    ax.invert_yaxis()
    ax.set_xlabel('Importance in AI/ML Applications')
    ax.set_title('Real-World Applications of Convex Optimization in AI', 
                fontweight='bold', fontsize=14)
    
    # Add application details
    for i, (app, desc) in enumerate(applications.items()):
        ax.text(0.05, i, desc, va='center', ha='left', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

plot_ai_applications()

# %% [markdown]
# ### 5.3 Key Insights from Our Implementation

# %% [markdown]
# <div style="background-color:#fff3e0; padding:20px; border-radius:10px; border-left:5px solid #FF9800;">
# <h3 style="color:#E65100;">🔍 Critical Observations:</h3>
# 
# <h4>1. <span style="color:#2E86AB;">Mathematical Equivalence</span></h4>
# <ul>
# <li style="color:black;">Both implementations find nearly identical solutions</li>
# <li style="color:black;">Small numerical differences due to solver precision and implementation details</li>
# <li style="color:black;">Same support vectors identified (core concept of SVM)</li>
# </ul>
# 
# <h4>2. <span style="color:#2E86AB;">Computational Efficiency</span></h4>
# <ul>
# <li style="color:black;">Scikit-learn uses optimized algorithms (SMO) for better scalability</li>
# <li style="color:black;">Our CVXOPT implementation demonstrates the mathematical foundation</li>
# <li style="color:black;">Both leverage convexity for guaranteed convergence</li>
# </ul>
# 
# <h4>3. <span style="color:#2E86AB;">Practical Implications</span></h4>
# <ul>
# <li style="color:black;">Convex optimization ensures reproducible and reliable model training</li>
# <li style="color:black;">No hyperparameter tuning needed for convergence guarantees</li>
# <li style="color:black;">Global optimum provides model interpretability and trustworthiness</li>
# </ul>
# </div>

# %% [markdown]
# ### 5.4 Advanced: Soft Margin SVM and Regularization

# %%
# Demonstrate soft-margin SVM with different C values
def demonstrate_soft_margin():
    """Show how regularization parameter C affects SVM"""
    
    # Create a dataset with some overlap
    X_soft, y_soft = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                        n_informative=2, n_clusters_per_class=1,
                                        flip_y=0.1, random_state=42)
    y_soft = 2 * y_soft - 1  # Convert to -1, +1
    
    C_values = [0.1, 1, 10, 1000]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, C_val in enumerate(C_values):
        # Train SVM with different C values
        svm_model = SVC(kernel='linear', C=C_val)
        svm_model.fit(X_soft, y_soft)
        
        # Plot decision boundary
        ax = axes[i]
        ax.scatter(X_soft[y_soft == -1, 0], X_soft[y_soft == -1, 1], 
                  c='red', s=50, alpha=0.7, label='Class -1')
        ax.scatter(X_soft[y_soft == 1, 0], X_soft[y_soft == 1, 1], 
                  c='blue', s=50, alpha=0.7, label='Class +1')
        
        # Create mesh for decision boundary
        x_min, x_max = X_soft[:, 0].min() - 0.5, X_soft[:, 0].max() + 0.5
        y_min, y_max = X_soft[:, 1].min() - 0.5, X_soft[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['orange', 'black', 'orange'],
                  linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
        
        # Highlight support vectors
        ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
                  s=150, facecolors='none', edgecolors='yellow', linewidth=2)
        
        ax.set_title(f'Soft Margin SVM (C={C_val})\nSupport Vectors: {len(svm_model.support_vectors_)}', 
                    fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Impact of Regularization Parameter C on SVM', 
                fontweight='bold', fontsize=16, y=1.02)
    plt.show()

demonstrate_soft_margin()

# %% [markdown]
# ## 6. Conclusion and Key Takeaways

# %% [markdown]
# <div style="background-color:#e3f2fd; padding:25px; border-radius:10px; border:2px solid #2196F3;">
# <h2 style="color:#1976D2; text-align:center;">🎯 Summary: Convex Optimization in AI</h2>
# 
# <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;">
# <div style="background-color:#bbdefb; padding:15px; border-radius:8px;">
# <h4 style="color:#0D47A1;">📊 Mathematical Foundation</h4>
# <ul>
# <li style="color:black;">Convex functions guarantee global optima</li>
# <li style="color:black;">Duality provides computational advantages</li>
# <li style="color:black;">Linear constraints maintain convexity</li>
# <li style="color:black;">Quadratic programming enables efficient solutions</li>
# </ul>
# </div>
# 
# <div style="background-color:#c8e6c9; padding:15px; border-radius:8px;">
# <h4 style="color:#1B5E20;">🤖 AI Applications</h4>
# <ul>
# <li style="color:black;">SVM for classification and regression</li>
# <li style="color:black;">Neural network loss minimization</li>
# <li style="color:black;">Regularization and model selection</li>
# <li style="color:black;">Reinforcement learning value functions</li>
# </ul>
# </div>
# 
# <div style="background-color:#fff9c4; padding:15px; border-radius:8px;">
# <h4 style="color:#F57F17;">⚡ Computational Advantages</h4>
# <ul>
# <li style="color:black;">Polynomial time algorithms</li>
# <li style="color:black;">Guaranteed convergence</li>
# <li style="color:black;">Reliable and reproducible results</li>
# <li style="color:black;">Scalable to large problems</li>
# </ul>
# </div>
# 
# <div style="background-color:#ffccbc; padding:15px; border-radius:8px;">
# <h4 style="color:#BF360C;">🔬 Practical Implementation</h4>
# <ul>
# <li style="color:black;">Dual formulation enables kernel methods</li>
# <li style="color:black;">Support vectors provide model interpretability</li>
# <li style="color:black;">Convexity ensures solution quality</li>
# <li style="color:black;">Wide range of solver options available</li>
# </ul>
# </div>
# </div>
# 
# <div style="text-align: center; margin-top: 20px; padding: 15px; background-color: #ffffff; border-radius: 8px;">
# <h4 style="color:#6A1B9A;">💡 The Convex Optimization Advantage in AI:</h4>
# <p style="color:black;font-size: 16px; font-weight: bold;">
# "Convex optimization provides the mathematical certainty that lets us build AI systems we can trust, 
# with performance guarantees that non-convex methods cannot offer."
# </p>
# </div>
# </div>

# %% [markdown]
# ### 6.1 Further Exploration
# 
# To deepen your understanding of convex optimization in AI:
# 
# 1. **Advanced Topics**:
#    - Kernel methods for non-linear SVMs
#    - Stochastic convex optimization
#    - Distributed convex optimization
# 
# 2. **Practical Extensions**:
#    - Multi-class SVM formulations
#    - SVM for regression (SVR)
#    - Large-scale SVM implementations
# 
# 3. **Mathematical Foundations**:
#    - Karush-Kuhn-Tucker (KKT) conditions
#    - Strong duality and Slater's condition
#    - Convex analysis and optimization theory

# %%
# Final visualization: The convex optimization pipeline
def optimization_pipeline():
    steps = [
        ('Problem Formulation', 'Define objective and constraints'),
        ('Convexity Verification', 'Check if problem is convex'),
        ('Dual Formulation', 'Transform to dual problem (if beneficial)'),
        ('Solver Selection', 'Choose appropriate convex optimizer'),
        ('Solution Extraction', 'Recover primal solution from dual'),
        ('Optimality Verification', 'Check KKT conditions')
    ]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    for i, (step, desc) in enumerate(steps):
        circle = plt.Circle((i*2 + 1, 0.5), 0.4, color=plt.cm.viridis(i/len(steps)), alpha=0.8)
        ax.add_patch(circle)
        ax.text(i*2 + 1, 0.5, str(i+1), ha='center', va='center', 
               fontweight='bold', color='white', fontsize=12)
        ax.text(i*2 + 1, 0, step, ha='center', va='top', 
               fontweight='bold', fontsize=10)
        ax.text(i*2 + 1, -0.2, desc, ha='center', va='top', 
               fontsize=8, style='italic')
        
        if i < len(steps)-1:
            ax.arrow(i*2 + 1.4, 0.5, 1.2, 0, head_width=0.1, 
                    head_length=0.1, fc='gray', ec='gray')
    
    ax.set_xlim(0, len(steps)*2)
    ax.set_ylim(-0.5, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Convex Optimization Pipeline for AI Applications', 
                fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.show()

optimization_pipeline()

print("\n" + "="*70)
print("🎉 NOTEBOOK COMPLETED: Convex Optimization in AI with SVM Example")
print("="*70)
print("\nKey accomplishments:")
print("✅ Demonstrated convex optimization fundamentals")
print("✅ Implemented SVM dual formulation from scratch")
print("✅ Compared custom implementation with scikit-learn")
print("✅ Visualized decision boundaries and support vectors")
print("✅ Analyzed real-world applications and insights")
print("\nThe convexity guarantee ensures our AI models find optimal solutions reliably!")