import numpy as np
import math
from scipy.stats import chi2

# Параметры логарифмического распределения
theta = 10 / 11


# Функция вероятности логарифмического распределения
def logarithmic_pmf(k, theta):
    if k < 1:
        return 0
    return -1 / math.log(1 - theta) * (theta ** k) / k


# Функция распределения логарифмического распределения
def logarithmic_cdf(k, theta):
    if k < 1:
        return 0
    total = 0.0
    for j in range(1, int(k) + 1):
        total += logarithmic_pmf(j, theta)
    return total


# Генерация выборки из логарифмического распределения
def generate_logarithmic_sample(n, theta):
    sample = []
    for _ in range(n):
        # Метод обратного преобразования
        u = np.random.uniform(0, 1)
        k = 1
        F = logarithmic_pmf(k, theta)
        while u > F:
            k += 1
            F += logarithmic_pmf(k, theta)
        sample.append(k)
    return np.array(sample)


# Оценка параметра theta методом моментов
def estimate_theta_mm(sample):
    n = len(sample)
    sample_mean = np.mean(sample)

    # Решаем уравнение: sample_mean = -θ/((1-θ)*ln(1-θ))
    # Используем метод деления пополам
    eps = 1e-10
    a, b = eps, 1 - eps

    def f(theta_val):
        if theta_val >= 1 or theta_val <= 0:
            return float('inf')
        return sample_mean + theta_val / ((1 - theta_val) * math.log(1 - theta_val))

    # Метод деления пополам
    for _ in range(100):
        mid = (a + b) / 2
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid

    return (a + b) / 2


# Критерий хи-квадрат для логарифмического распределения (простая гипотеза)
def chi_square_test_logarithmic_simple(sample, theta, alpha=0.05):
    n = len(sample)

    # Определяем интервалы
    # Сначала вычисляем теоретические вероятности для k = 1, 2, 3, ...
    max_k = 20  # Начальное значение
    probs = []
    k_values = []
    cum_prob = 0

    # Находим достаточно большое k, чтобы покрыть 99.9% вероятности
    for k in range(1, 100):
        prob = logarithmic_pmf(k, theta)
        if prob > 1e-6:  # Игнорируем очень маленькие вероятности
            probs.append(prob)
            k_values.append(k)
            cum_prob += prob
            if cum_prob > 0.999:
                break

    # Добавляем последний интервал для всех оставшихся значений
    if cum_prob < 1:
        probs.append(1 - cum_prob)
        k_values.append(k_values[-1] + 1)  # Обозначаем как "k_max+"

    # Объединяем интервалы, чтобы ожидаемые частоты были >= 5
    expected = [n * p for p in probs]
    merged_probs = []
    merged_intervals = []
    current_prob = 0
    current_interval = []

    for i, (prob, k) in enumerate(zip(probs, k_values)):
        current_prob += prob
        current_interval.append(k)

        if n * current_prob >= 5 or i == len(probs) - 1:
            merged_probs.append(current_prob)
            if len(current_interval) == 1:
                merged_intervals.append(f"{current_interval[0]}")
            else:
                merged_intervals.append(f"{current_interval[0]}-{current_interval[-1]}")
            current_prob = 0
            current_interval = []

    # Если последний интервал слишком маленький, объединяем с предыдущим
    if len(merged_probs) > 1 and n * merged_probs[-1] < 5:
        merged_probs[-2] += merged_probs[-1]
        merged_intervals[-2] = f"{merged_intervals[-2].split('-')[0]}-{merged_intervals[-1].split('-')[-1]}"
        merged_probs.pop()
        merged_intervals.pop()

    # Вычисляем наблюдаемые частоты
    observed = [0] * len(merged_probs)

    for value in sample:
        placed = False
        for i, interval_str in enumerate(merged_intervals):
            if '-' in interval_str:
                start, end = map(int, interval_str.split('-'))
                if start <= value <= end:
                    observed[i] += 1
                    placed = True
                    break
            else:
                if value == int(interval_str):
                    observed[i] += 1
                    placed = True
                    break

        # Если значение не попало ни в один интервал (должно быть в последнем)
        if not placed and len(merged_intervals) > 0:
            observed[-1] += 1

    # Вычисляем статистику хи-квадрат
    chi2_stat = 0
    for i in range(len(merged_probs)):
        exp_freq = n * merged_probs[i]
        if exp_freq > 0:
            chi2_stat += (observed[i] - exp_freq) ** 2 / exp_freq

    # Число степеней свободы
    df = len(merged_probs) - 1

    # Критическое значение
    critical_value = chi2.ppf(1 - alpha, df)

    # p-значение
    p_value = 1 - chi2.cdf(chi2_stat, df)

    # Решение
    reject = chi2_stat > critical_value

    return chi2_stat, p_value, reject, df, critical_value, observed, merged_probs, merged_intervals


# Функция для проведения эксперимента
def run_chi_square_experiment(sample_sizes, num_samples_per_size, theta, alpha=0.05, hypothesis_type='simple'):
    results = {
        'sample_size': [],
        'sample_num': [],
        'chi2': [],
        'p_value': [],
        'rejected': [],
        'df': [],
        'critical_value': [],
        'theta_est': []
    }

    summary = {
        'sample_size': [],
        'num_samples': [],
        'avg_chi2': [],
        'rejected_count': [],
        'rejection_rate': [],
        'critical_value': []
    }

    for n in sample_sizes:
        total_rejected = 0
        sum_chi2 = 0
        current_critical_value = None

        for sample_num in range(1, num_samples_per_size + 1):
            # Генерация выборки
            sample = generate_logarithmic_sample(n, theta)

            if hypothesis_type == 'simple':
                # Простая гипотеза
                chi2_stat, p_value, reject, df, critical, observed, probs, intervals = chi_square_test_logarithmic_simple(
                    sample, theta, alpha
                )
                theta_est = theta  # Известный параметр
            else:
                # Сложная гипотеза
                chi2_stat, p_value, reject, df, critical, theta_est, observed, probs, intervals = chi_square_test_logarithmic_composite(
                    sample, alpha
                )

            # Сохранение результатов
            results['sample_size'].append(n)
            results['sample_num'].append(sample_num)
            results['chi2'].append(chi2_stat)
            results['p_value'].append(p_value)
            results['rejected'].append(reject)
            results['df'].append(df)
            results['critical_value'].append(critical)
            results['theta_est'].append(theta_est)

            # Статистика для сводки
            sum_chi2 += chi2_stat
            current_critical_value = critical

            if reject:
                total_rejected += 1

        # Добавление сводной информации
        summary['sample_size'].append(n)
        summary['num_samples'].append(num_samples_per_size)
        summary['avg_chi2'].append(sum_chi2 / num_samples_per_size)
        summary['rejected_count'].append(total_rejected)
        summary['rejection_rate'].append(total_rejected / num_samples_per_size)
        summary['critical_value'].append(current_critical_value)

    return results, summary


# Основная программа
if __name__ == "__main__":
    # Параметры эксперимента
    np.random.seed(42)  # Для воспроизводимости
    sample_sizes = [100, 200, 400, 600, 800, 1000]
    num_samples_per_size = 5
    alpha = 0.05
    print("КРИТЕРИЙ ХИ-КВАДРАТ ДЛЯ ЛОГАРИФМИЧЕСКОГО РАСПРЕДЕЛЕНИЯ")
    print("\nПРОСТАЯ ГИПОТЕЗА (θ известен = 10/11 ≈ 0.909091)")
    simple_results, simple_summary = run_chi_square_experiment(
        sample_sizes, num_samples_per_size, theta, alpha, hypothesis_type='simple'
    )

    # Вывод сводной таблицы для простой гипотезы
    print("\nСводная таблица (простая гипотеза):")
    print("-" * 90)
    print(
        f"{'Объем выборки':<15} {'Среднее χ²':<15} {'Критическое значение':<20} {'Отвергнуто H0':<15} {'Доля отвержений':<15}")
    print("-" * 90)

    for i in range(len(simple_summary['sample_size'])):
        n = simple_summary['sample_size'][i]
        avg_chi2 = simple_summary['avg_chi2'][i]
        crit_val = simple_summary['critical_value'][i]
        rejected = simple_summary['rejected_count'][i]
        rate = simple_summary['rejection_rate'][i]

        print(f"{n:<15} {avg_chi2:<15.2f} {crit_val:<20.3f} {rejected:<15} {rate:<15.2%}")

    print("-" * 90)

    print("\nПОДРОБНЫЙ АНАЛИЗ ДЛЯ ВЫБОРКИ ОБЪЕМОМ 100 (простая гипотеза)")
    # Найдем первую выборку объемом 100
    for i in range(len(simple_results['sample_size'])):
        sample = generate_logarithmic_sample(100, theta)
        chi2_stat, p_value, reject, df, critical, observed, probs, intervals = chi_square_test_logarithmic_simple(
            sample, theta, alpha
        )

        print(f"\nВыборка 1, n=100:")
        print(f"  Статистика χ²: {chi2_stat:.4f}")
        print(f"  Число степеней свободы: {df}")
        print(f"  Критическое значение (α=0.05): {critical:.4f}")
        print(f"  p-значение: {p_value:.4f}")
        print(f"  Решение: {'ОТВЕРГНУТА' if reject else 'ПРИНЯТА'}")

        print("\n  Распределение по интервалам:")
        print("  " + "-" * 50)
        print(f"  {'Интервал':<15} {'Наблюдаемая':<15} {'Ожидаемая':<15} {'(O-E)²/E':<15}")
        print("  " + "-" * 50)

        total_obs = 0
        total_exp = 0
        for j in range(len(intervals)):
            obs = observed[j]
            exp = 100 * probs[j]
            chi2_contrib = (obs - exp) ** 2 / exp if exp > 0 else 0
            print(f"  {intervals[j]:<15} {obs:<15.0f} {exp:<15.2f} {chi2_contrib:<15.4f}")
            total_obs += obs
            total_exp += exp

        print("  " + "-" * 50)
        print(f"  {'Всего':<15} {total_obs:<15.0f} {total_exp:<15.2f} {chi2_stat:<15.4f}")
        break