import numpy as np
from scipy.optimize import minimize_scalar

# Параметры распределения Лапласа
mu_true = 8.5
theta_true = 2.0
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
num_samples = 5


# Функция плотности распределения Лапласа
def laplace_pdf(x, mu, theta):
    return (theta / 2) * np.exp(-theta * np.abs(x - mu))


# Генерация выборок из распределения Лапласа
def generate_laplace_sample(n, mu, theta):
    # Метод 1: через разность экспоненциальных
    exp1 = np.random.exponential(scale=1.0, size=n)
    exp2 = np.random.exponential(scale=1.0, size=n)
    return mu + (exp1 - exp2) / theta

    # Альтернативный метод: через обратную функцию распределения
    # u = np.random.uniform(0, 1, n)
    # return mu + np.sign(u - 0.5) * (1/theta) * np.log(1 - 2 * np.abs(u - 0.5))


# Метод моментов для распределения Лапласа
def estimate_params_MM(sample):
    n = len(sample)

    # Оценка μ (выборочное среднее)
    mu_MM = np.mean(sample)

    # Оценка θ (используем выборочную дисперсию с поправкой Бесселя)
    # В отчете используется формула: θ_MM = √(2/S²), где S² - выборочная дисперсия с поправкой Бесселя
    S2 = np.var(sample, ddof=1)  # ddof=1 для несмещенной оценки
    if S2 > 0:
        # Формула из отчета: θ_MM = √(2/S²) = √(2/выборочная дисперсия)
        theta_MM = np.sqrt(2 / S2)

        # Альтернативная формула из отчета: θ_MM = √(2(n-1)/(n * S²))
        # theta_MM = np.sqrt(2 * (n-1) / (n * S2))
    else:
        theta_MM = np.nan

    return mu_MM, theta_MM


# Метод максимального правдоподобия для распределения Лапласа
def estimate_params_MLE(sample):
    n = len(sample)

    # Оценка μ (медиана выборки)
    mu_MLE = np.median(sample)

    # Оценка θ
    sum_abs_dev = np.sum(np.abs(sample - mu_MLE))
    if sum_abs_dev > 0:
        theta_MLE = n / sum_abs_dev
    else:
        theta_MLE = np.nan

    return mu_MLE, theta_MLE


# Альтернативная версия МПП через максимизацию правдоподобия
def estimate_params_MLE_optim(sample):
    n = len(sample)

    # Функция правдоподобия (логарифм)
    def neg_log_likelihood(params):
        mu, theta = params
        if theta <= 0:
            return np.inf
        return -(-n * np.log(2) + n * np.log(theta) - theta * np.sum(np.abs(sample - mu)))

    # Начальные приближения
    mu0 = np.median(sample)
    theta0 = n / np.sum(np.abs(sample - mu0)) if np.sum(np.abs(sample - mu0)) > 0 else 1.0

    # Минимизация (метод Нелдера-Мида)
    from scipy.optimize import minimize
    res = minimize(neg_log_likelihood, [mu0, theta0], method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})

    return res.x[0], res.x[1]


# Генерация выборок
np.random.seed(42)  # Для воспроизводимости
laplace_samples = {}
for n in sample_sizes:
    laplace_samples[n] = [generate_laplace_sample(n, mu_true, theta_true) for _ in range(num_samples)]

# Вычисление оценок
print(f"{'n':<8} {'μ_MM':<12} {'θ_MM':<12} {'μ_MLE':<12} {'θ_MLE':<12} "
      f"{'E[X]':<10} {'D[X]':<10}")
print("-" * 90)

for n in sample_sizes:
    mu_MM_list, theta_MM_list = [], []
    mu_MLE_list, theta_MLE_list = [], []

    for sample in laplace_samples[n]:
        # Метод моментов
        mu_MM, theta_MM = estimate_params_MM(sample)
        mu_MM_list.append(mu_MM)
        theta_MM_list.append(theta_MM)

        # Метод максимального правдоподобия
        mu_MLE, theta_MLE = estimate_params_MLE(sample)
        mu_MLE_list.append(mu_MLE)
        theta_MLE_list.append(theta_MLE)

    # Средние значения
    mu_MM_mean = np.mean(mu_MM_list)
    theta_MM_mean = np.mean(theta_MM_list)
    mu_MLE_mean = np.mean(mu_MLE_list)
    theta_MLE_mean = np.mean(theta_MLE_list)

    print(f"{n:<8} {mu_MM_mean:<12.4f} {theta_MM_mean:<12.4f} "
          f"{mu_MLE_mean:<12.4f} {theta_MLE_mean:<12.4f} "
          f"{mu_true:<10.4f} {2 / theta_true ** 2:<10.4f}")