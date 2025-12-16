import numpy as np
import matplotlib.pyplot as plt

# Параметры распределения Лапласа
mu_true = 8.5
theta_true = 2.0
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
num_samples = 5  # Количество выборок каждого размера

# Истинное значение дисперсии D[X] = 2/θ²
D_true = 2 / (theta_true ** 2)


# Генерация выборок из распределения Лапласа
def generate_laplace_sample(n, mu, theta):
    exp1 = np.random.exponential(scale=1.0, size=n)
    exp2 = np.random.exponential(scale=1.0, size=n)
    return mu + (exp1 - exp2) / theta


# Стандартная выборочная дисперсия (несмещенная)
def sample_variance(sample):
    n = len(sample)
    if n < 2:
        return 0.0
    return np.var(sample, ddof=1)  # ddof=1 для несмещенной оценки


# Оптимальная оценка дисперсии для распределения Лапласа
def optimal_variance_estimate(sample):
    n = len(sample)

    # Оценка μ (выборочная медиана)
    mu_hat = np.median(sample)

    # Сумма абсолютных отклонений
    sum_abs_dev = np.sum(np.abs(sample - mu_hat))

    # Оптимальная оценка дисперсии
    if n > 0:
        D_opt = 2 * (sum_abs_dev ** 2) / (n * (n + 1))
    else:
        D_opt = np.nan

    return D_opt, mu_hat


# Анализ оценок дисперсии
def analyze_variance_estimates(sample_sizes, num_samples, mu, theta):
    # Генерация всех выборок
    np.random.seed(42)  # Для воспроизводимости
    all_samples = {}
    for n in sample_sizes:
        all_samples[n] = [generate_laplace_sample(n, mu, theta) for _ in range(num_samples)]

    # Результаты
    results = []
    print("Сравнение оценок дисперсии для распределения Лапласа")
    print()
    print(f"{'n':<10} {'S²':<20} {'D_opt':<20} {'|S² - D[X]|':<20} {'|D_opt - D[X]|':<20}")
    print("-" * 100)

    for n in sample_sizes:
        S2_values = []
        D_opt_values = []
        S2_errors = []
        D_opt_errors = []

        for sample in all_samples[n]:
            # Стандартная выборочная дисперсия
            S2 = sample_variance(sample)
            S2_error = np.abs(S2 - D_true)

            # Оптимальная оценка дисперсии
            D_opt, mu_hat = optimal_variance_estimate(sample)
            D_opt_error = np.abs(D_opt - D_true)

            S2_values.append(S2)
            D_opt_values.append(D_opt)
            S2_errors.append(S2_error)
            D_opt_errors.append(D_opt_error)

        # Средние значения по выборкам
        S2_mean = np.mean(S2_values)
        D_opt_mean = np.mean(D_opt_values)
        S2_error_mean = np.mean(S2_errors)
        D_opt_error_mean = np.mean(D_opt_errors)

        # Стандартные отклонения
        S2_std = np.std(S2_values)
        D_opt_std = np.std(D_opt_values)

        # Сохраняем результаты
        results.append({
            'n': n,
            'S2_mean': S2_mean,
            'S2_std': S2_std,
            'D_opt_mean': D_opt_mean,
            'D_opt_std': D_opt_std,
            'S2_error_mean': S2_error_mean,
            'D_opt_error_mean': D_opt_error_mean,
            'S2_values': S2_values,
            'D_opt_values': D_opt_values
        })

        print(f"{n:<10} {S2_mean:<20.4f} {D_opt_mean:<20.4f} "
              f"{S2_error_mean:<20.4f} {D_opt_error_mean:<20.4f}")

    return results, all_samples


# Запуск анализа
print("Анализ оценок дисперсии для распределения Лапласа")
print(f"Параметры: μ = {mu_true}, θ = {theta_true}")
print(f"Истинная дисперсия: D[X] = 2/θ² = 2/({theta_true}²) = {D_true:.4f}\n")

results, all_samples = analyze_variance_estimates(sample_sizes, num_samples, mu_true, theta_true)