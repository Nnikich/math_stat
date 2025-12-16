import numpy as np
import pandas as pd
from scipy import stats
import math

# Параметры распределения Лапласа
MU_TRUE = 8.5
THETA_TRUE = 2

# Уровень значимости
ALPHA = 0.05


def generate_laplace_samples(n, num_samples=5):
    samples = []
    for _ in range(num_samples):
        # Генерация равномерного распределения
        u = np.random.uniform(-0.5, 0.5, n)
        # Преобразование в распределение Лапласа
        x = MU_TRUE - np.sign(u) * np.log(1 - 2 * np.abs(u)) / THETA_TRUE
        samples.append(x)
    return samples


def laplace_cdf(x, mu, theta):
    return 0.5 * (1 + np.sign(x - mu) * (1 - np.exp(-theta * np.abs(x - mu))))


def mle_laplace(sample):
    mu_est = np.median(sample)
    theta_est = 1 / np.mean(np.abs(sample - mu_est))
    return mu_est, theta_est


def kolmogorov_smirnov_simple(sample):
    n = len(sample)
    sorted_sample = np.sort(sample)

    # Эмпирическая функция распределения
    ecdf = np.arange(1, n + 1) / n

    # Теоретическая функция распределения
    tcdf = laplace_cdf(sorted_sample, MU_TRUE, THETA_TRUE)

    # Статистика Колмогорова
    D_plus = np.max(ecdf - tcdf)
    D_minus = np.max(tcdf - (np.arange(0, n) / n))
    D = max(D_plus, D_minus)

    # Умножение на sqrt(n)
    S = math.sqrt(n) * D

    # Критическое значение
    t_critical = 1.3581  # Для α=0.05

    # Решение
    decision = "Принята" if S <= t_critical else "Отверг."

    return S, t_critical, decision


def chi_square_simple_laplace(sample):
    n = len(sample)

    # Разбиение на 10 интервалов равной вероятности
    k = 10
    p_expected = 1 / k

    # Квантили распределения Лапласа
    quantiles = []
    for i in range(1, k):
        p = i / k
        if p <= 0.5:
            q = MU_TRUE + np.log(2 * p) / THETA_TRUE
        else:
            q = MU_TRUE - np.log(2 * (1 - p)) / THETA_TRUE
        quantiles.append(q)

    # Границы интервалов
    boundaries = [-np.inf] + quantiles + [np.inf]

    # Наблюдаемые частоты
    observed = np.histogram(sample, bins=boundaries)[0]

    # Ожидаемые частоты
    expected = np.full(k, n * p_expected)

    # Статистика хи-квадрат
    chi2 = np.sum((observed - expected) ** 2 / expected)

    # Степени свободы
    df = k - 1

    # Критическое значение
    chi2_critical = stats.chi2.ppf(1 - ALPHA, df)

    # Решение
    decision = "Принята" if chi2 <= chi2_critical else "Отверг."

    return chi2, df, chi2_critical, decision


def kolmogorov_smirnov_composite(sample):
    n = len(sample)

    # Оценка параметров
    mu_est, theta_est = mle_laplace(sample)

    sorted_sample = np.sort(sample)

    # Эмпирическая функция распределения
    ecdf = np.arange(1, n + 1) / n

    # Теоретическая функция распределения с оцененными параметрами
    tcdf = laplace_cdf(sorted_sample, mu_est, theta_est)

    # Статистика Колмогорова
    D_plus = np.max(ecdf - tcdf)
    D_minus = np.max(tcdf - (np.arange(0, n) / n))
    D = max(D_plus, D_minus)

    # Умножение на sqrt(n)
    S = math.sqrt(n) * D

    # Критические значения из отчета
    critical_values = {
        5: 0.895, 10: 0.905, 100: 1.025, 200: 1.045,
        400: 1.060, 600: 1.065, 800: 1.070, 1000: 1.075
    }

    t_critical = critical_values.get(n, 1.3581)

    # Решение
    decision = "Принята" if S <= t_critical else "Отверг."

    return S, t_critical, decision


def chi_square_composite_laplace(sample):
    n = len(sample)

    # Оценка параметров
    mu_est, theta_est = mle_laplace(sample)

    # Количество интервалов - используем фиксированные значения из отчета для n>=100
    # Для n<100 не применяем критерий
    k_dict = {
        100: 7, 200: 7, 400: 8, 600: 8, 800: 9, 1000: 10
    }

    if n in k_dict:
        k = k_dict[n]
    else:
        # Для n < 100 не применяем критерий
        return None, None, None, None, "Не применим"

    # Разбиение на k интервалов равной вероятности с оцененными параметрами
    p_expected = 1 / k

    # Квантили распределения Лапласа с оцененными параметрами
    quantiles = []
    for i in range(1, k):
        p = i / k
        if p <= 0.5:
            q = mu_est + np.log(2 * p) / theta_est
        else:
            q = mu_est - np.log(2 * (1 - p)) / theta_est
        quantiles.append(q)

    # Границы интервалов
    boundaries = [-np.inf] + quantiles + [np.inf]

    # Наблюдаемые частоты
    observed = np.histogram(sample, bins=boundaries)[0]

    # Ожидаемые частоты
    expected = np.full(k, n * p_expected)

    # Статистика хи-квадрат
    chi2 = np.sum((observed - expected) ** 2 / expected)

    # Степени свободы (оценены 2 параметра)
    df = k - 1 - 2

    # Критическое значение
    chi2_critical = stats.chi2.ppf(1 - ALPHA, df) if df > 0 else None

    # Решение
    if df <= 0:
        decision = "Не применим"
    else:
        decision = "Принята" if chi2 <= chi2_critical else "Отверг."

    return chi2, k, df, chi2_critical, decision


def main():
    volumes = [5, 10, 100, 200, 400, 600, 800, 1000]
    num_samples = 5

    results_ks_simple = []
    results_chi2_simple = []
    results_ks_composite = []
    results_chi2_composite = []

    for n in volumes:
        print(f"Объем выборки: n = {n}")

        # Генерация выборок
        samples = generate_laplace_samples(n, num_samples)

        # Проверка простой гипотезы Колмогорова (всегда применяется)
        ks_simple_stats = []
        ks_simple_decisions = []

        # Проверка сложной гипотезы Колмогорова (всегда применяется)
        ks_composite_stats = []
        ks_composite_decisions = []

        # Проверка простой гипотезы хи-квадрат (только для n>=100)
        chi2_simple_stats = []
        chi2_simple_decisions = []

        # Проверка сложной гипотезы хи-квадрат (только для n>=100)
        chi2_composite_stats = []
        chi2_composite_ks = []
        chi2_composite_dfs = []
        chi2_composite_decisions = []

        for i, sample in enumerate(samples):
            print(f"\n--- Выборка {i + 1} ---")

            # Критерий Колмогорова (простая гипотеза) - всегда применяется
            S, t_crit, decision = kolmogorov_smirnov_simple(sample)
            ks_simple_stats.append(S)
            ks_simple_decisions.append(decision)
            print(f"Колмогоров (простая): S = {S:.4f}, решение: {decision}")

            # Критерий Колмогорова (сложная гипотеза) - всегда применяется
            S_comp, t_crit_comp, decision_comp = kolmogorov_smirnov_composite(sample)
            ks_composite_stats.append(S_comp)
            ks_composite_decisions.append(decision_comp)
            print(f"Колмогоров (сложная): S = {S_comp:.4f}, решение: {decision_comp}")

            # Критерий хи-квадрат (простая гипотеза) - только для n>=100
            if n >= 100:
                chi2, df, chi2_crit, decision = chi_square_simple_laplace(sample)
                chi2_simple_stats.append(chi2)
                chi2_simple_decisions.append(decision)
                print(f"Хи-квадрат (простая): χ² = {chi2:.4f}, df = {df}, решение: {decision}")
            else:
                print(f"Хи-квадрат (простая): Не применим (n < 100)")

            # Критерий хи-квадрат (сложная гипотеза) - только для n>=100
            if n >= 100:
                result = chi_square_composite_laplace(sample)
                if result[0] is not None:
                    chi2_comp, k, df_comp, chi2_crit_comp, decision_comp = result
                    chi2_composite_stats.append(chi2_comp)
                    chi2_composite_ks.append(k)
                    chi2_composite_dfs.append(df_comp)
                    chi2_composite_decisions.append(decision_comp)
                    print(
                        f"Хи-квадрат (сложная): χ² = {chi2_comp:.4f}, k = {k}, df = {df_comp}, решение: {decision_comp}")
                else:
                    print(f"Хи-квадрат (сложная): Не применим")
            else:
                print(f"Хи-квадрат (сложная): Не применим (n < 100)")

        # Сохранение результатов
        results_ks_simple.append({
            'n': n,
            'stats': ks_simple_stats,
            'decisions': ks_simple_decisions
        })

        results_ks_composite.append({
            'n': n,
            'stats': ks_composite_stats,
            'decisions': ks_composite_decisions
        })

        # Сохраняем результаты хи-квадрат только для n>=100
        if n >= 100:
            results_chi2_simple.append({
                'n': n,
                'stats': chi2_simple_stats,
                'decisions': chi2_simple_decisions
            })

            results_chi2_composite.append({
                'n': n,
                'stats': chi2_composite_stats,
                'k': chi2_composite_ks,
                'df': chi2_composite_dfs,
                'decisions': chi2_composite_decisions
            })

    print("ТАБЛИЦА 1: Критерий Колмогорова (простая гипотеза)")
    print("Статистика S = √n·D_n для каждой выборки")
    print()

    # Создаем DataFrame для красивой таблицы
    data = {}
    for res in results_ks_simple:
        n = res['n']
        data[f'n={n}'] = [f"{stat:.4f}" for stat in res['stats']]

    df_table = pd.DataFrame(data)
    df_table.index = [f'Выборка {i + 1}' for i in range(5)]
    print(df_table.to_string())

    print("ТАБЛИЦА 2: Критерий Колмогорова (сложная гипотеза)")
    print("Статистика S = √n·D_n для каждой выборки")
    print()

    data = {}
    for res in results_ks_composite:
        n = res['n']
        data[f'n={n}'] = [f"{stat:.4f}" for stat in res['stats']]

    df_table = pd.DataFrame(data)
    df_table.index = [f'Выборка {i + 1}' for i in range(5)]
    print(df_table.to_string())

    print("ТАБЛИЦА 3: Критерий хи-квадрат (простая гипотеза)")
    print("Только для n ≥ 100")
    print()

    print(
        f"{'Объем n':<10} {'Среднее χ²':<15} {'Крит.знач. χ²₀.₉₅,₉':<20} {'Отвергнуто H₀':<15} {'Доля отвержений':<15}")
    print("-" * 80)

    for res in results_chi2_simple:
        n = res['n']
        avg_chi2 = np.mean(res['stats'])
        rejected = sum(1 for d in res['decisions'] if d == "Отверг.")
        rejected_rate = rejected / len(res['decisions']) * 100
        df = 9  # 10 интервалов - 1
        chi2_crit = stats.chi2.ppf(1 - ALPHA, df)

        print(
            f"{n:<10} {avg_chi2:<15.2f} {chi2_crit:<20.3f} {rejected}/{len(res['decisions']):<15} {rejected_rate:<15.1f}%")

    print("ТАБЛИЦА 4: Критерий хи-квадрат (сложная гипотеза)")
    print("Только для n ≥ 100")
    print()

    print(f"{'n':<5} {'k':<5} {'f':<5} {'Крит.знач.':<12} {'χ² по выборкам':<40} {'Принято H₀':<15}")
    print("-" * 85)

    for res in results_chi2_composite:
        n = res['n']
        k = res['k'][0] if res['k'] else 0
        df = res['df'][0] if res['df'] and res['df'][0] > 0 else 0

        if df > 0:
            chi2_crit = stats.chi2.ppf(1 - ALPHA, df)
        else:
            chi2_crit = None

        chi2_values = ", ".join(f"{val:.2f}" for val in res['stats'])
        accepted = sum(1 for d in res['decisions'] if d == "Принята")
        total = len(res['decisions'])

        if chi2_crit:
            print(f"{n:<5} {k:<5} {df:<5} {chi2_crit:<12.3f} {chi2_values:<40} {accepted}/{total}")
        else:
            print(f"{n:<5} {k:<5} {df:<5} {'--':<12} {chi2_values:<40} {accepted}/{total}")

    print("ИТОГОВАЯ СТАТИСТИКА")

    # Подсчет общей статистики
    total_ks_simple = 0
    accepted_ks_simple = 0

    total_ks_composite = 0
    accepted_ks_composite = 0

    total_chi2_simple = 0
    accepted_chi2_simple = 0

    total_chi2_composite = 0
    accepted_chi2_composite = 0

    # Критерий Колмогорова (простая гипотеза)
    for res in results_ks_simple:
        total_ks_simple += len(res['decisions'])
        accepted_ks_simple += sum(1 for d in res['decisions'] if d == "Принята")

    # Критерий Колмогорова (сложная гипотеза)
    for res in results_ks_composite:
        total_ks_composite += len(res['decisions'])
        accepted_ks_composite += sum(1 for d in res['decisions'] if d == "Принята")

    # Критерий хи-квадрат (простая гипотеза)
    for res in results_chi2_simple:
        total_chi2_simple += len(res['decisions'])
        accepted_chi2_simple += sum(1 for d in res['decisions'] if d == "Принята")

    # Критерий хи-квадрат (сложная гипотеза)
    for res in results_chi2_composite:
        total_chi2_composite += len(res['decisions'])
        accepted_chi2_composite += sum(1 for d in res['decisions'] if d == "Принята")

    print(f"{'Критерий':<35} {'Принято H₀':<15} {'Отвергнуто H₀':<15} {'Всего':<10} {'Доля принятых':<15}")
    print("-" * 90)

    print(
        f"{'Колмогоров (простая гипотеза)':<35} {accepted_ks_simple:<15} {total_ks_simple - accepted_ks_simple:<15} {total_ks_simple:<10} {accepted_ks_simple / total_ks_simple * 100:<15.1f}%")
    print(
        f"{'Колмогоров (сложная гипотеза)':<35} {accepted_ks_composite:<15} {total_ks_composite - accepted_ks_composite:<15} {total_ks_composite:<10} {accepted_ks_composite / total_ks_composite * 100:<15.1f}%")
    print(
        f"{'Хи-квадрат (простая гипотеза)':<35} {accepted_chi2_simple:<15} {total_chi2_simple - accepted_chi2_simple:<15} {total_chi2_simple:<10} {accepted_chi2_simple / total_chi2_simple * 100:<15.1f}%")
    print(
        f"{'Хи-квадрат (сложная гипотеза)':<35} {accepted_chi2_composite:<15} {total_chi2_composite - accepted_chi2_composite:<15} {total_chi2_composite:<10} {accepted_chi2_composite / total_chi2_composite * 100:<15.1f}%")

    print("ОБЪЯСНЕНИЕ КОЛИЧЕСТВА ПРОВЕРОК:")
    print("1. Колмогоров (простая и сложная гипотеза): 8 объемов выборок × 5 выборок = 40 проверок")
    print(
        "2. Хи-квадрат (простая гипотеза): 6 объемов выборок (100, 200, 400, 600, 800, 1000) × 5 выборок = 30 проверок")
    print(
        "3. Хи-квадрат (сложная гипотеза): 6 объемов выборок (100, 200, 400, 600, 800, 1000) × 5 выборок = 30 проверок")
    print("\nПричина: критерий хи-квадрат требует выполнения условия np_i ≥ 5, что невозможно для n < 100")


if __name__ == "__main__":
    # Для воспроизводимости результатов
    np.random.seed(42)
    main()