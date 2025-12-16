import numpy as np
from scipy.optimize import minimize_scalar

# Параметры
theta_true = 10 / 11
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
num_samples = 5


# Функция вероятности логарифмического распределения (такая же как в отчете)
def logarithmic_pmf(k, theta):
    return -1 / np.log(1 - theta) * (theta ** k) / k


# Генерация выборок (точная копия из отчета)
def generate_logarithmic_sample(n, theta):
    sample = []
    for _ in range(n):
        u = np.random.uniform(0, 1)
        k = 1
        cum_prob = logarithmic_pmf(k, theta)
        while cum_prob < u and k < 1000:
            k += 1
            cum_prob += logarithmic_pmf(k, theta)
        sample.append(k)
    return np.array(sample)


# Теоретическое математическое ожидание
def theoretical_mean(theta):
    return -theta / ((1 - theta) * np.log(1 - theta))


# Метод моментов - решение уравнения
def estimate_theta_MM(sample):
    x_bar = np.mean(sample)

    # Уравнение: x̄ + θ/((1-θ)ln(1-θ)) = 0
    def equation(theta):
        if theta <= 1e-10 or theta >= 1 - 1e-10:
            return np.inf
        return (x_bar + theta / ((1 - theta) * np.log(1 - theta))) ** 2

    # Минимизация на интервале (0, 1)
    res = minimize_scalar(equation, bounds=(1e-10, 1 - 1e-10), method='bounded', options={'xatol': 1e-12})
    return res.x


# Метод максимального правдоподобия
def estimate_theta_MLE(sample):
    n = len(sample)
    sum_x = np.sum(sample)

    # Отрицательный логарифм правдоподобия
    def neg_log_likelihood(theta):
        if theta <= 1e-10 or theta >= 1 - 1e-10:
            return np.inf
        return -(-n * np.log(-np.log(1 - theta)) + np.log(theta) * sum_x - np.sum(np.log(sample)))

    # Минимизация
    res = minimize_scalar(neg_log_likelihood, bounds=(1e-10, 1 - 1e-10), method='bounded', options={'xatol': 1e-12})
    return res.x


# Генерация выборок
np.random.seed(42)
log_samples = {}
for n in sample_sizes:
    log_samples[n] = [generate_logarithmic_sample(n, theta_true) for _ in range(num_samples)]

# Вычисление оценок
print(f"{'n':<8} {'θ_MM':<18} {'θ_MLE':<18} "
      f"{'θ_MM - θ':<15} {'θ_MLE - θ':<15} {'θ (истинное)':<15}")
print("-" * 100)

for n in sample_sizes:
    theta_MM_list = []
    theta_MLE_list = []

    for sample in log_samples[n]:
        theta_MM = estimate_theta_MM(sample)
        theta_MLE = estimate_theta_MLE(sample)
        theta_MM_list.append(theta_MM)
        theta_MLE_list.append(theta_MLE)

    theta_MM_mean = np.mean(theta_MM_list)
    theta_MLE_mean = np.mean(theta_MLE_list)

    print(f"{n:<8} {theta_MM_mean:<18.6f} {theta_MLE_mean:<18.6f} "
          f"{theta_MM_mean - theta_true:<15.6f} {theta_MLE_mean - theta_true:<15.6f} {theta_true:<15.6f}")
