import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ks_2samp

# Параметры распределений
theta_log = 10 / 11  # Для логарифмического распределения
mu_laplace = 8.5  # Для распределения Лапласа
theta_laplace = 2.0  # Для распределения Лапласа

# Объемы выборок для анализа
sample_sizes = {
    'одинаковый': [5, 10, 100],
    'разный': [(100, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
}

# Критическое значение для α = 0.05
t_critical = 1.358


# Функции генерации выборок
def generate_logarithmic_sample(n, theta):
    sample = []
    for _ in range(n):
        u = np.random.uniform(0, 1)
        k = 1
        pmf = -1 / np.log(1 - theta) * (theta ** k) / k
        while pmf < u and k < 1000:
            k += 1
            pmf += -1 / np.log(1 - theta) * (theta ** k) / k
        sample.append(k)
    return np.array(sample)


def generate_laplace_sample(n, mu, theta):
    exp1 = np.random.exponential(scale=1.0, size=n)
    exp2 = np.random.exponential(scale=1.0, size=n)
    return mu + (exp1 - exp2) / theta


# Функция критерия Смирнова
def smirnov_test(sample1, sample2):
    # Вычисляем эмпирические функции распределения
    n1, n2 = len(sample1), len(sample2)

    # Объединяем выборки и сортируем
    combined = np.concatenate([sample1, sample2])
    combined_sorted = np.sort(combined)

    # Вычисляем эмпирические функции распределения
    ecdf1 = np.searchsorted(np.sort(sample1), combined_sorted, side='right') / n1
    ecdf2 = np.searchsorted(np.sort(sample2), combined_sorted, side='right') / n2

    # Статистика Колмогорова-Смирнова
    D_stat = np.max(np.abs(ecdf1 - ecdf2))

    # Нормированная статистика
    T_stat = np.sqrt((n1 * n2) / (n1 + n2)) * D_stat

    # Решение
    decision = T_stat <= t_critical

    return D_stat, T_stat, decision


# Анализ для логарифмического распределения
def analyze_smirnov_logarithmic():
    print("Логарифмическое распределение (θ = 10/11)")
    print("Критерий однородности Смирнова")

    # Для воспроизводимости результатов
    np.random.seed(42)

    # 1. Выборки одинакового объема
    print("\nПроверка однородности для выборок одинакового объема:")
    print("-" * 90)
    print(f"{'n':<8} {'Сравнение':<20} {'D':<12} {'√(n²/(2n)) * D':<20} {'Решение':<15}")
    print("-" * 90)

    results_same = {}

    for n in sample_sizes['одинаковый']:
        # Генерируем 3 выборки для сравнения
        samples = [generate_logarithmic_sample(n, theta_log) for _ in range(3)]

        # Сравниваем выборку 1 с 2 и 2 с 3
        comparisons = [("Выб.1 vs Выб.2", samples[0], samples[1]),
                       ("Выб.2 vs Выб.3", samples[1], samples[2])]

        results_same[n] = []

        for label, sample1, sample2 in comparisons:
            D_stat, T_stat, decision = smirnov_test(sample1, sample2)

            # Для одинаковых выборок: √(n²/(2n)) * D = √(n/2) * D
            T_same = np.sqrt(n / 2) * D_stat

            results_same[n].append({
                'label': label,
                'D': D_stat,
                'T_same': T_same,
                'decision': decision
            })

            decision_text = "Принята" if decision else "Отвергнута"
            print(f"{n:<8} {label:<20} {D_stat:<12.4f} {T_same:<20.4f} {decision_text:<15}")

    # Выборки разного объема
    print("\nПроверка однородности для выборок разного объема:")
    print("-" * 90)
    print(f"{'n,m':<12} {'D':<15} {'(nm/(n+m))':<20} {'(nm/(n+m)) * D':<20} {'Решение':<15}")
    print("-" * 90)

    results_diff = {}

    for n, m in sample_sizes['разный']:
        # Генерируем по одной выборке каждого объема
        sample1 = generate_logarithmic_sample(n, theta_log)
        sample2 = generate_logarithmic_sample(m, theta_log)

        D_stat, T_stat, decision = smirnov_test(sample1, sample2)

        results_diff[(n, m)] = {
            'D': D_stat,
            'T_stat': T_stat,
            'decision': decision
        }

        decision_text = "Принята" if decision else "Отвергнута"
        factor = np.sqrt((n * m) / (n + m))
        print(f"{n},{m:<10} {D_stat:<15.4f} {factor:<20.4f} {T_stat:<20.4f} {decision_text:<15}")

    return results_same, results_diff


# Анализ для распределения Лапласа
def analyze_smirnov_laplace():
    print("Распределение Лапласа (μ = 8.5, θ = 2)")
    print("Критерий однородности Смирнова")

    # Для воспроизводимости результатов (используем другой seed)
    np.random.seed(123)

    # Выборки одинакового объема
    print("\nПроверка однородности для выборок одинакового объема:")
    print("-" * 90)
    print(f"{'n':<8} {'Сравнение':<20} {'D':<12} {'√(n²/(2n)) * D':<20} {'Решение':<15}")
    print("-" * 90)

    results_same = {}

    for n in sample_sizes['одинаковый']:
        # Генерируем 3 выборки для сравнения
        samples = [generate_laplace_sample(n, mu_laplace, theta_laplace) for _ in range(3)]

        # Сравниваем выборку 1 с 2 и 2 с 3
        comparisons = [("Выб.1 vs Выб.2", samples[0], samples[1]),
                       ("Выб.2 vs Выб.3", samples[1], samples[2])]

        results_same[n] = []

        for label, sample1, sample2 in comparisons:
            D_stat, T_stat, decision = smirnov_test(sample1, sample2)

            # Для одинаковых выборок: √(n²/(2n)) * D = √(n/2) * D
            T_same = np.sqrt(n / 2) * D_stat

            results_same[n].append({
                'label': label,
                'D': D_stat,
                'T_same': T_same,
                'decision': decision
            })

            decision_text = "Принята" if decision else "Отвергнута"
            print(f"{n:<8} {label:<20} {D_stat:<12.4f} {T_same:<20.4f} {decision_text:<15}")

    # Выборки разного объема
    print("\nПроверка однородности для выборок разного объема:")
    print("-" * 90)
    print(f"{'n,m':<12} {'D':<15} {'√(nm/(n+m))':<20} {'√(nm/(n+m)) * D':<20} {'Решение':<15}")
    print("-" * 90)

    results_diff = {}

    for n, m in sample_sizes['разный']:
        # Генерируем по одной выборке каждого объема
        sample1 = generate_laplace_sample(n, mu_laplace, theta_laplace)
        sample2 = generate_laplace_sample(m, mu_laplace, theta_laplace)

        D_stat, T_stat, decision = smirnov_test(sample1, sample2)

        results_diff[(n, m)] = {
            'D': D_stat,
            'T_stat': T_stat,
            'decision': decision
        }

        decision_text = "Принята" if decision else "Отвергнута"
        factor = np.sqrt((n * m) / (n + m))
        print(f"{n},{m:<10} {D_stat:<15.4f} {factor:<20.4f} {T_stat:<20.4f} {decision_text:<15}")

    return results_same, results_diff


# Основной анализ
print("Критерий однородности Смирнова - Анализ результатов")
print(f"Критическое значение для α = 0.05: t_critical = {t_critical}")

# Анализ для логарифмического распределения
results_log_same, results_log_diff = analyze_smirnov_logarithmic()

# Анализ для распределения Лапласа
results_laplace_same, results_laplace_diff = analyze_smirnov_laplace()