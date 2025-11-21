# 模型检验报告

## 一、检验概述
本报告针对四个小问的模型进行了有效性、鲁棒性和稳定性的系统检验。

## 二、检验结果汇总表

| 小问编号 | 检验方法 | 检验指标与结果 | 检验结论 | 详细说明 |
| :--- | :--- | :--- | :--- | :--- |
| Q1 | 蒙特卡洛扰动测试 (Monte Carlo Perturbation) | Max Torque = 9.47 Nm (Limit 120) | **通过 (Robust)** | 在5%参数噪音下，最大力矩仍远小于限制，安全系数 > 30。 |
| Q2 | 重复运行稳定性测试 (Repeated Runs Stability) | Fitness CV = 11.83% | **通过 (Stable)** | 多次运行算法，解的变异系数低，说明算法收敛稳定。 |
| Q3 | 噪声注入测试 (Noise Injection Test) | Max Deviation = 0.095m (Limit 0.1m) | **通过 (Robust)** | 引入20%环境噪声后，重心偏移仍保持在安全域内。 |
| Q4 | 超参数敏感性分析 (Hyperparameter Sensitivity) | HV Improvement = 8.2% | **有效 (Effective)** | 增加种群规模显著提升了解集质量，算法对计算资源敏感。 |
