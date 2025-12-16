import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Параметры логарифмического распределения
theta_true = 10 / 11
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
num_samples = 5  # Количество выборок каждого размера


# Теоретическое математическое ожидание логарифмического распределения
def theoretical_mean(theta):
    return -theta / ((1 - theta) * np.log(1 - theta))


# Истинное значение τ(θ) = E[X]
tau_true = theoretical_mean(theta_true)
print(f"Истинное значение E[X] = τ(θ) = {tau_true:.6f}\n")


# Функция вероятности логарифмического распределения
def logarithmic_pmf(k, theta):
    return -1 / np.log(1 - theta) * (theta ** k) / k


# Генерация выборок из логарифмического распределения
def generate_logarithmic_sample(n, theta):
    sample = []
    for _ in range(n):
        u = np.random.uniform(0, 1)
        k = 1
        cum_prob = logarithmic_pmf(k, theta)
        while cum_prob < u and k < 1000:  # Ограничиваем сверху для безопасности
            k += 1
            cum_prob += logarithmic_pmf(k, theta)
        sample.append(k)
    return np.array(sample)


# Оптимальная оценка τ(θ) = E[X] - выборочное среднее
def optimal_estimate_tau(sample):
    return np.mean(sample)


# Генерация всех выборок
np.random.seed(42)  # Фиксируем seed для воспроизводимости
all_samples = {}
for n in sample_sizes:
    all_samples[n] = [generate_logarithmic_sample(n, theta_true) for _ in range(num_samples)]

# Анализ результатов
print("Таблица: Оптимальная оценка τ(θ) = E[X] для логарифмического распределения")
print()
print(f"{'n':<10} {'X̄_opt':<15} {'X̄ - E[X]':<15} {'√n * |X̄ - E[X]|':<20}")
print("-" * 80)

for n in sample_sizes:
    # Вычисляем средние оценки по 5 выборкам
    tau_estimates = []

    for sample in all_samples[n]:
        tau_est = optimal_estimate_tau(sample)
        tau_estimates.append(tau_est)

    # Средняя оценка по 5 выборкам
    tau_mean = np.mean(tau_estimates)

    # Разница между средней оценкой и истинным значением
    diff = tau_mean - tau_true
    scaled_diff = np.sqrt(n) * np.abs(diff)

    print(f"{n:<10} {tau_mean:<15.4f} {diff:<15.4f} {scaled_diff:<20.4f}")
