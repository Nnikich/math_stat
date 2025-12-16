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


# Функция для группировки значений в фиксированное количество интервалов
def create_fixed_intervals_for_logarithmic(theta_est, n, num_intervals):
    # Сначала вычисляем теоретические вероятности для k = 1, 2, 3, ...
    probs_all = []
    k_values = []
    cum_prob = 0

    # Собираем вероятности для k от 1 до 50
    for k in range(1, 51):
        prob = logarithmic_pmf(k, theta_est)
        if prob > 1e-10:  # Игнорируем очень маленькие вероятности
            probs_all.append(prob)
            k_values.append(k)
            cum_prob += prob
            if cum_prob > 0.9999:
                break

    # Если сумма вероятностей меньше 1, добавляем последний интервал
    if cum_prob < 1:
        probs_all.append(1 - cum_prob)
        k_values.append(k_values[-1] + 1)

    # Определяем количество значений в каждом интервале
    num_values = len(probs_all)
    values_per_interval = max(1, num_values // num_intervals)

    # Формируем интервалы
    intervals = []
    probs = []
    boundaries = []
    current_prob = 0
    start_k = k_values[0]

    for i in range(num_intervals):
        if i == num_intervals - 1:
            # Последний интервал - все оставшиеся значения
            end_k = k_values[-1]
            interval_prob = sum(probs_all[i * values_per_interval:])
            if start_k == end_k:
                intervals.append(f"{start_k}")
            else:
                intervals.append(f"{start_k}-{end_k}")
            probs.append(interval_prob)
            boundaries.append((start_k, end_k))
        else:
            # Определяем конец текущего интервала
            end_idx = min((i + 1) * values_per_interval - 1, len(k_values) - 1)
            end_k = k_values[end_idx]
            interval_prob = sum(probs_all[i * values_per_interval:(i + 1) * values_per_interval])

            if start_k == end_k:
                intervals.append(f"{start_k}")
            else:
                intervals.append(f"{start_k}-{end_k}")

            probs.append(interval_prob)
            boundaries.append((start_k, end_k))
            start_k = end_k + 1

    return intervals, probs, boundaries


# Вспомогательная функция для безопасного извлечения границ интервала
def get_interval_bounds(interval_str):
    parts = interval_str.split('-')
    if len(parts) == 1:
        val = int(parts[0])
        return val, val
    else:
        # Берем первый и последний элемент
        return int(parts[0]), int(parts[-1])


# Критерий хи-квадрат для логарифмического распределения (сложная гипотеза)
def chi_square_test_logarithmic_composite(sample, alpha=0.05):
    n = len(sample)

    # Оцениваем параметр theta
    theta_est = estimate_theta_mm(sample)

    # Определяем количество интервалов в зависимости от объема выборки
    if n == 100:
        num_intervals = 5
    elif n == 200:
        num_intervals = 6
    elif n == 400:
        num_intervals = 7
    elif n == 600:
        num_intervals = 8
    elif n == 800:
        num_intervals = 8
    elif n == 1000:
        num_intervals = 9
    else:
        # Для других объемов используем эмпирическое правило
        num_intervals = max(4, min(10, int(5 * math.log10(n))))

    # Создаем фиксированные интервалы
    intervals, probs, boundaries = create_fixed_intervals_for_logarithmic(theta_est, n, num_intervals)

    # Вычисляем наблюдаемые частоты
    observed = [0] * len(intervals)

    for value in sample:
        placed = False
        for i, (start, end) in enumerate(boundaries):
            if start <= value <= end:
                observed[i] += 1
                placed = True
                break

        # Если значение не попало ни в один интервал
        if not placed and len(intervals) > 0:
            observed[-1] += 1

    # Вычисляем ожидаемые частоты
    expected = [n * p for p in probs]

    # Проверяем условие np_i >= 5 и объединяем интервалы при необходимости
    i = 0
    while i < len(intervals):
        if expected[i] < 5:
            if i + 1 < len(intervals):
                # Объединяем с правым соседом
                # Получаем текущие границы
                current_start, current_end = get_interval_bounds(intervals[i])
                next_start, next_end = get_interval_bounds(intervals[i + 1])

                # Создаем новый интервал
                new_start = min(current_start, next_start)
                new_end = max(current_end, next_end)
                if new_start == new_end:
                    new_interval = f"{new_start}"
                else:
                    new_interval = f"{new_start}-{new_end}"

                # Обновляем данные
                intervals[i] = new_interval
                probs[i] = probs[i] + probs[i + 1]
                observed[i] = observed[i] + observed[i + 1]
                expected[i] = expected[i] + expected[i + 1]

                # Удаляем правого соседа
                intervals.pop(i + 1)
                probs.pop(i + 1)
                observed.pop(i + 1)
                expected.pop(i + 1)
                boundaries.pop(i + 1)

                # Обновляем границы
                boundaries[i] = (new_start, new_end)

            elif i > 0:
                # Объединяем с левым соседом
                prev_start, prev_end = get_interval_bounds(intervals[i - 1])
                current_start, current_end = get_interval_bounds(intervals[i])

                # Создаем новый интервал
                new_start = min(prev_start, current_start)
                new_end = max(prev_end, current_end)
                if new_start == new_end:
                    new_interval = f"{new_start}"
                else:
                    new_interval = f"{new_start}-{new_end}"

                # Обновляем левого соседа
                intervals[i - 1] = new_interval
                probs[i - 1] = probs[i - 1] + probs[i]
                observed[i - 1] = observed[i - 1] + observed[i]
                expected[i - 1] = expected[i - 1] + expected[i]

                # Удаляем текущий интервал
                intervals.pop(i)
                probs.pop(i)
                observed.pop(i)
                expected.pop(i)
                boundaries.pop(i)

                # Обновляем границы
                boundaries[i - 1] = (new_start, new_end)
                i -= 1
        i += 1

    # Пересчитываем границы интервалов (на всякий случай)
    boundaries = []
    for interval in intervals:
        start, end = get_interval_bounds(interval)
        boundaries.append((start, end))

    # Вычисляем статистику хи-квадрат
    chi2_stat = 0
    for i in range(len(probs)):
        exp_freq = n * probs[i]
        if exp_freq > 0:
            chi2_stat += (observed[i] - exp_freq) ** 2 / exp_freq
        else:
            # Если ожидаемая частота равна 0, пропускаем этот интервал
            pass

    # Число степеней свободы
    # Для сложной гипотезы: df = (число интервалов - 1 - число оцененных параметров)
    df = len(intervals) - 1 - 1  # Оценили 1 параметр

    if df < 1:
        df = 1  # Минимум 1 степень свободы

    # Критическое значение
    critical_value = chi2.ppf(1 - alpha, df)

    # p-значение
    p_value = 1 - chi2.cdf(chi2_stat, df) if df > 0 else 1.0

    # Решение
    reject = chi2_stat > critical_value

    return chi2_stat, p_value, reject, df, critical_value, theta_est, observed, probs, intervals


# Функция для проведения эксперимента со сложной гипотезой
def run_chi_square_experiment_composite(sample_sizes, num_samples_per_size, alpha=0.05):
    results = []
    summary = {
        'sample_size': [],
        'avg_chi2': [],
        'critical_value': [],
        'rejected_count': [],
        'rejection_rate': [],
        'num_intervals': [],
        'avg_theta_est': []
    }

    for n in sample_sizes:
        chi2_stats = []
        rejected = []
        theta_ests = []
        num_intervals_list = []
        critical_values = []

        for sample_num in range(1, num_samples_per_size + 1):
            # Генерация выборки
            sample = generate_logarithmic_sample(n, theta)

            # Проверка гипотезы
            chi2_stat, p_value, reject, df, critical, theta_est, observed, probs, intervals = chi_square_test_logarithmic_composite(
                sample, alpha
            )

            # Сохранение результатов
            results.append({
                'sample_size': n,
                'sample_num': sample_num,
                'chi2': chi2_stat,
                'p_value': p_value,
                'rejected': reject,
                'df': df,
                'critical_value': critical,
                'theta_est': theta_est,
                'observed': observed,
                'intervals': intervals,
                'probs': probs
            })

            # Статистика для сводки
            chi2_stats.append(chi2_stat)
            rejected.append(reject)
            theta_ests.append(theta_est)
            num_intervals_list.append(len(intervals))
            critical_values.append(critical)

        # Добавление сводной информации
        summary['sample_size'].append(n)
        summary['avg_chi2'].append(np.mean(chi2_stats))

        # Используем среднее критическое значение
        summary['critical_value'].append(np.mean(critical_values))

        summary['rejected_count'].append(np.sum(rejected))
        summary['rejection_rate'].append(np.mean(rejected))
        summary['num_intervals'].append(np.mean(num_intervals_list))
        summary['avg_theta_est'].append(np.mean(theta_ests))

    return results, summary


# Основная программа
if __name__ == "__main__":
    # Параметры эксперимента
    np.random.seed(42)  # Для воспроизводимости
    sample_sizes = [100, 200, 400, 600, 800, 1000]
    num_samples_per_size = 5
    alpha = 0.05
    print("КРИТЕРИЙ ХИ-КВАДРАТ ДЛЯ ЛОГАРИФМИЧЕСКОГО РАСПРЕДЕЛЕНИЯ")

    # Сложная гипотеза (неизвестный параметр)
    print("\nСЛОЖНАЯ ГИПОТЕЗА (θ неизвестен, оценивается по выборке)")

    composite_results, composite_summary = run_chi_square_experiment_composite(
        sample_sizes, num_samples_per_size, alpha
    )

    # Вывод сводной таблицы для сложной гипотезы
    print("\nСводная таблица (сложная гипотеза):")
    print("-" * 90)
    print(
        f"{'Объем выборки':<15} {'Среднее χ²':<15} {'Критическое значение':<20} {'Отвергнуто H0':<15} {'Доля отвержений':<15}")
    print("-" * 90)

    total_rejected = 0
    total_tests = 0

    for i in range(len(composite_summary['sample_size'])):
        n = composite_summary['sample_size'][i]
        avg_chi2 = composite_summary['avg_chi2'][i]
        crit_val = composite_summary['critical_value'][i]
        rejected = composite_summary['rejected_count'][i]
        rate = composite_summary['rejection_rate'][i]

        total_rejected += rejected
        total_tests += num_samples_per_size

        print(f"{n:<15} {avg_chi2:<15.2f} {crit_val:<20.3f} {rejected:<15} {rate:<15.2%}")

    print("-" * 90)
