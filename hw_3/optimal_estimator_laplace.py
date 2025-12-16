import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

# Параметры распределения Лапласа
mu_true = 8.5
theta_true = 2.0
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
num_samples = 5  # Количество выборок каждого размера


# Генерация выборок из распределения Лапласа
def generate_laplace_sample(n, mu, theta):
    exp1 = np.random.exponential(scale=1.0, size=n)
    exp2 = np.random.exponential(scale=1.0, size=n)
    return mu + (exp1 - exp2) / theta


# Оценка параметров методом максимального правдоподобия
def estimate_params_MLE(sample):
    n = len(sample)

    # μ_MLE = медиана выборки
    mu_MLE = np.median(sample)

    # θ_MLE = n / Σ|X_i - μ_MLE|
    sum_abs_dev = np.sum(np.abs(sample - mu_MLE))
    theta_MLE = n / sum_abs_dev if sum_abs_dev > 0 else np.nan

    return mu_MLE, theta_MLE


# Оптимальная оценка τ(μ,θ) = P(X > μ+1)
def optimal_estimate_tau(sample, method='parametric'):
    n = len(sample)
    mu_MLE, theta_MLE = estimate_params_MLE(sample)

    if method == 'parametric':
        # Параметрическая оценка: τ = (1/2) * e^(-θ_MLE)
        tau_est = 0.5 * np.exp(-theta_MLE)

    elif method == 'empirical':
        # Эмпирическая оценка: доля элементов > μ_MLE + 1
        indicator = (sample > mu_MLE + 1).astype(float)
        tau_est = np.mean(indicator)

    else:
        raise ValueError("method должен быть 'parametric' или 'empirical'")

    return tau_est, mu_MLE, theta_MLE


# Анализ оптимальной оценки
def analyze_optimal_estimate_laplace(sample_sizes, num_samples, mu, theta):
    # Генерация всех выборок
    np.random.seed(42)  # Для воспроизводимости
    all_samples = {}
    for n in sample_sizes:
        all_samples[n] = [generate_laplace_sample(n, mu, theta) for _ in range(num_samples)]

    # Результаты
    results = []
    print("Оптимальная оценка τ_opt = P(X > μ+1) для распределения Лапласа")
    print()
    print(f"{'n':<10} {'τ_opt':<15} {'|τ̂ - τ|':<15} {'Относительная ошибка (%)':<20}")
    print("-" * 90)

    for n in sample_sizes:
        tau_estimates = []
        abs_errors = []
        rel_errors = []

        # Статистики по параметрам (для дополнительного анализа)
        mu_estimates = []
        theta_estimates = []

        for sample in all_samples[n]:
            # Вычисление оптимальной оценки
            tau_est, mu_MLE, theta_MLE = optimal_estimate_tau(sample, method='parametric')

            # Ошибки
            abs_error = np.abs(tau_est - tau_true)
            rel_error = 100 * abs_error / tau_true

            tau_estimates.append(tau_est)
            abs_errors.append(abs_error)
            rel_errors.append(rel_error)

            mu_estimates.append(mu_MLE)
            theta_estimates.append(theta_MLE)

        # Средние значения по выборкам
        tau_mean = np.mean(tau_estimates)
        abs_error_mean = np.mean(abs_errors)
        rel_error_mean = np.mean(rel_errors)

        # Стандартные отклонения
        tau_std = np.std(tau_estimates)

        # Сохраняем результаты
        results.append({
            'n': n,
            'tau_mean': tau_mean,
            'tau_std': tau_std,
            'abs_error_mean': abs_error_mean,
            'rel_error_mean': rel_error_mean,
            'mu_estimates': mu_estimates,
            'theta_estimates': theta_estimates
        })

        print(f"{n:<10} {tau_mean:<15.6f} {abs_error_mean:<15.6f} {rel_error_mean:<20.2f}%")

    return results, all_samples


# Запуск анализа
print("Анализ оптимальной оценки τ(μ,θ) = P(X > μ+1) для распределения Лапласа")
print(f"Параметры: μ = {mu_true}, θ = {theta_true}")
print(f"τ = P(X > {mu_true}+1) = P(X > {mu_true + 1}) = 0.5 * e^(-{theta_true}) = {tau_true:.6f}\n")

results, all_samples = analyze_optimal_estimate_laplace(sample_sizes, num_samples, mu_true, theta_true)