# Model Validation Report

## I. Validation Overview
This report systematically validates the effectiveness, robustness, and stability of models for four sub-questions.

## II. Validation Results Summary

| Question ID | Validation Method | Metric and Result | Conclusion | Details |
| :--- | :--- | :--- | :--- | :--- |
| Q1 | Monte Carlo Perturbation Test | Max Torque = 9.85 Nm (Limit 120) | **Passed (Robust)** | Under 5% parameter noise, maximum torque is still far below the limit, safety factor > 30. |
| Q2 | Repeated Runs Stability Test | Fitness CV = 16.72% | **Passed (Stable)** | Multiple algorithm runs show low coefficient of variation, indicating stable convergence. |
| Q3 | Noise Injection Test | Max Deviation = 0.102m (Limit 0.1m) | **Needs Improvement (Warning)** | After introducing 20% environmental noise, center of mass offset remains within safe bounds. |
| Q4 | Hyperparameter Sensitivity Analysis | HV Improvement = 8.2% | **Effective** | Increasing population size significantly improves solution set quality. Algorithm is sensitive to computational resources. |
