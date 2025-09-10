# scripts/run_portfolio_optimization.py

import numpy as np
import pyswarms as ps
from qmind_quant.optimization.portfolio_optimizer import PortfolioOptimizer


def objective_function(weights_array, optimizer_instance):
    """
    This is the function the swarm will try to minimize.
    It takes an array of weights from the swarm and returns the score
    from our portfolio evaluator.
    """
    scores = []
    for weights in weights_array:
        # The swarm doesn't know that weights must sum to 1, so we normalize them.
        normalized_weights = weights / np.sum(weights)
        score = optimizer_instance.evaluate(normalized_weights)
        scores.append(score)
    return np.array(scores)


def main():
    """
    Main function to set up and run the Particle Swarm Optimization.
    """
    print("--- Initializing Portfolio Optimizer ---")
    optimizer_instance = PortfolioOptimizer(
        tickers=["AAPL", "GOOG"], initial_capital=100000.0
    )

    # --- PSO Configuration ---
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    n_particles = 10
    dimensions = 2

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=dimensions, options=options
    )

    print("\n--- Starting Particle Swarm Optimization ---")

    # --- THIS IS THE FIX ---
    # We pass our extra arguments directly to the optimize method,
    # and pyswarms will forward them to our objective_function.
    best_cost, best_pos = optimizer.optimize(
        objective_func=objective_function,
        iters=50,
        optimizer_instance=optimizer_instance,  # Pass directly, not in a 'kwargs' dict
    )

    # Normalize the final best position to get the allocation
    final_allocation = best_pos / np.sum(best_pos)

    print("\n--- Optimization Complete ---")
    print(f"Best (Negative) Sharpe Ratio Found: {best_cost:.4f}")
    print("\nOptimal Portfolio Allocation:")
    print(f"  - XGBoost Strategy: {final_allocation[0]:.2%}")
    print(f"  - Buy-and-Hold Strategy: {final_allocation[1]:.2%}")


if __name__ == "__main__":
    main()
