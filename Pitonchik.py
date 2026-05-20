"""
РГР №3. Регрессионные модели и их интерпретация.
Вариант A-4:
x — Максимальный объём каналов памяти (MMAX)
y — Относительная производительность (PRP)

Используемые библиотеки:
  numpy      — вычисления
  pandas     — чтение CSV
  scipy      — статистика
  matplotlib — графики
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =============================================================================

# CSV-файл должен лежать в одной папке со скриптом
# и содержать столбцы: x,y

df = pd.read_csv('RGR3_A_4.csv')

# Проверка столбцов
if 'x' not in df.columns or 'y' not in df.columns:
    raise ValueError("CSV-файл должен содержать столбцы 'x' и 'y'")

# Преобразование в numpy
x = df['x'].to_numpy(dtype=float)
y = df['y'].to_numpy(dtype=float)

n = len(x)
x_star = 21.0240

print("=" * 70)
print("ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ")
print("=" * 70)
print(df.head())

# =============================================================================
# 2. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# =============================================================================

x_mean = np.mean(x)
y_mean = np.mean(y)

Sxx = np.sum((x - x_mean) ** 2)
Sxy = np.sum((x - x_mean) * (y - y_mean))
Syy = np.sum((y - y_mean) ** 2)

print("\n" + "=" * 70)
print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
print("=" * 70)
print(f"n = {n}")
print(f"x̄ = {x_mean:.4f}")
print(f"ȳ = {y_mean:.4f}")
print(f"Sxx = {Sxx:.4f}")
print(f"Sxy = {Sxy:.4f}")
print(f"Syy = {Syy:.4f}")

# =============================================================================
# 3. ЛИНЕЙНАЯ МОДЕЛЬ
# =============================================================================

b1 = Sxy / Sxx
b0 = y_mean - b1 * x_mean

y_hat_lin = b0 + b1 * x
e_lin = y - y_hat_lin

RSS_lin = np.sum(e_lin ** 2)
R2_lin = 1 - RSS_lin / Syy
RMSE_lin = np.sqrt(RSS_lin / n)
A_lin = np.mean(np.abs(e_lin / y)) * 100

# Оценка дисперсии
s2_lin = RSS_lin / (n - 2)
s_lin = np.sqrt(s2_lin)

# Стандартные ошибки
SE_b1 = s_lin / np.sqrt(Sxx)
SE_b0 = s_lin * np.sqrt(1 / n + x_mean ** 2 / Sxx)

# t-критерий
t_crit = stats.t.ppf(0.975, df=n - 2)
t_obs = b1 / SE_b1

# Доверительные интервалы
CI_b0 = (
    b0 - t_crit * SE_b0,
    b0 + t_crit * SE_b0
)

CI_b1 = (
    b1 - t_crit * SE_b1,
    b1 + t_crit * SE_b1
)

y_pred_lin = b0 + b1 * x_star

print("\n" + "=" * 70)
print("ЛИНЕЙНАЯ МОДЕЛЬ")
print("=" * 70)

print(f"ŷ = {b0:.4f} + {b1:.4f}·x")
print(f"R² = {R2_lin:.4f}")
print(f"RMSE = {RMSE_lin:.4f}")
print(f"A = {A_lin:.2f}%")
print(f"RSS = {RSS_lin:.4f}")

print(f"\ns² = {s2_lin:.4f}")
print(f"s = {s_lin:.4f}")

print(f"\nSE(b0) = {SE_b0:.4f}")
print(f"SE(b1) = {SE_b1:.4f}")

print(f"\nt_набл = {t_obs:.4f}")
print(f"t_крит = {t_crit:.4f}")

if abs(t_obs) > t_crit:
    print("Гипотеза H0: b1 = 0 ОТВЕРГАЕТСЯ")
else:
    print("Нет оснований отвергнуть H0")

print(f"\nДИ для b0: ({CI_b0[0]:.4f}; {CI_b0[1]:.4f})")
print(f"ДИ для b1: ({CI_b1[0]:.4f}; {CI_b1[1]:.4f})")

print(f"\nПрогноз при x*={x_star}: ŷ = {y_pred_lin:.4f}")

# =============================================================================
# 4. КВАДРАТИЧНАЯ МОДЕЛЬ
# =============================================================================

coeffs_q = np.polyfit(x, y, 2)

c2_q, c1_q, c0_q = coeffs_q

y_hat_quad = np.polyval(coeffs_q, x)
e_quad = y - y_hat_quad

RSS_quad = np.sum(e_quad ** 2)
R2_quad = 1 - RSS_quad / Syy
RMSE_quad = np.sqrt(RSS_quad / n)
A_quad = np.mean(np.abs(e_quad / y)) * 100

y_pred_quad = np.polyval(coeffs_q, x_star)

print("\n" + "=" * 70)
print("КВАДРАТИЧНАЯ МОДЕЛЬ")
print("=" * 70)

print(f"ŷ = {c0_q:.4f} + {c1_q:.4f}·x + ({c2_q:.6f})·x²")
print(f"R² = {R2_quad:.4f}")
print(f"RMSE = {RMSE_quad:.4f}")
print(f"A = {A_quad:.2f}%")

print(f"\nПрогноз при x*={x_star}: ŷ = {y_pred_quad:.4f}")

# =============================================================================
# 5. СТЕПЕННАЯ МОДЕЛЬ
# =============================================================================

ln_x = np.log(x)
ln_y = np.log(y)

ln_x_mean = np.mean(ln_x)
ln_y_mean = np.mean(ln_y)

Sxx_pw = np.sum((ln_x - ln_x_mean) ** 2)
Sxy_pw = np.sum((ln_x - ln_x_mean) * (ln_y - ln_y_mean))
Syy_pw = np.sum((ln_y - ln_y_mean) ** 2)

b_pw = Sxy_pw / Sxx_pw
ln_a_pw = ln_y_mean - b_pw * ln_x_mean

a_pw = np.exp(ln_a_pw)

# R² в логарифмическом пространстве
ln_yhat_pw = ln_a_pw + b_pw * ln_x

RSS_pw_log = np.sum((ln_y - ln_yhat_pw) ** 2)
R2_pw_log = 1 - RSS_pw_log / Syy_pw

# В исходном пространстве
y_hat_pow = a_pw * x ** b_pw
e_pow = y - y_hat_pow

RSS_pow = np.sum(e_pow ** 2)
R2_pow = 1 - RSS_pow / Syy
RMSE_pow = np.sqrt(RSS_pow / n)
A_pow = np.mean(np.abs(e_pow / y)) * 100

y_pred_pow = a_pw * x_star ** b_pw

print("\n" + "=" * 70)
print("СТЕПЕННАЯ МОДЕЛЬ")
print("=" * 70)

print(f"ŷ = {a_pw:.4f} · x^{b_pw:.4f}")

print(f"\nR² (лог. пространство) = {R2_pw_log:.4f}")
print(f"R² (исходное пространство) = {R2_pow:.4f}")

print(f"RMSE = {RMSE_pow:.4f}")
print(f"A = {A_pow:.2f}%")

print(f"\nПрогноз при x*={x_star}: ŷ = {y_pred_pow:.4f}")

# =============================================================================
# 6. СРАВНЕНИЕ МОДЕЛЕЙ
# =============================================================================

print("\n" + "=" * 70)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

print(f"{'Модель':<16} {'R²':>8} {'RMSE':>10} {'A,%':>10} {'ŷ(x*)':>12}")
print("-" * 60)

print(f"{'Линейная':<16} "
      f"{R2_lin:>8.4f} "
      f"{RMSE_lin:>10.2f} "
      f"{A_lin:>10.2f} "
      f"{y_pred_lin:>12.2f}")

print(f"{'Квадратичная':<16} "
      f"{R2_quad:>8.4f} "
      f"{RMSE_quad:>10.2f} "
      f"{A_quad:>10.2f} "
      f"{y_pred_quad:>12.2f}")

print(f"{'Степенная':<16} "
      f"{R2_pow:>8.4f} "
      f"{RMSE_pow:>10.2f} "
      f"{A_pow:>10.2f} "
      f"{y_pred_pow:>12.2f}")

# =============================================================================
# 7. ТАБЛИЦА ОСТАТКОВ
# =============================================================================

print("\n" + "=" * 70)
print("ТАБЛИЦА ОСТАТКОВ (первые 10 наблюдений)")
print("=" * 70)

print(f"{'i':>3} {'x':>6} {'y':>8} "
      f"{'ŷ_лин':>10} {'e_лин':>10} "
      f"{'ŷ_кв':>10} {'e_кв':>10} "
      f"{'ŷ_ст':>10} {'e_ст':>10}")

for i in range(min(10, n)):
    print(
        f"{i+1:>3} "
        f"{x[i]:>6.1f} "
        f"{y[i]:>8.1f} "
        f"{y_hat_lin[i]:>10.2f} "
        f"{e_lin[i]:>10.2f} "
        f"{y_hat_quad[i]:>10.2f} "
        f"{e_quad[i]:>10.2f} "
        f"{y_hat_pow[i]:>10.2f} "
        f"{e_pow[i]:>10.2f}"
    )

# =============================================================================
# 8. ГРАФИКИ
# =============================================================================

x_plot = np.linspace(x.min(), x.max(), 600)

y_plot_lin = b0 + b1 * x_plot
y_plot_quad = np.polyval(coeffs_q, x_plot)
y_plot_pow = a_pw * x_plot ** b_pw

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle(
    'РГР №3 — Регрессионный анализ\n'
    'x — максимальный объём каналов памяти\n'
    'y — относительная производительность',
    fontsize=14,
    fontweight='bold'
)

# -----------------------------------------------------------------------------
# (1) Диаграмма рассеяния
# -----------------------------------------------------------------------------

ax = axes[0, 0]

ax.scatter(
    x,
    y,
    color='steelblue',
    alpha=0.7,
    s=50
)

ax.set_title('1. Диаграмма рассеяния', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# (2) Линейная модель
# -----------------------------------------------------------------------------

ax = axes[0, 1]

ax.scatter(x, y, color='steelblue', alpha=0.6, s=45)
ax.plot(x_plot, y_plot_lin, 'r-', linewidth=2.2)

ax.set_title(
    f'2. Линейная модель\nR² = {R2_lin:.4f}',
    fontweight='bold'
)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# (3) Квадратичная модель
# -----------------------------------------------------------------------------

ax = axes[0, 2]

ax.scatter(x, y, color='steelblue', alpha=0.6, s=45)
ax.plot(x_plot, y_plot_quad, 'g-', linewidth=2.2)

ax.set_title(
    f'3. Квадратичная модель\nR² = {R2_quad:.4f}',
    fontweight='bold'
)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# (4) Степенная модель
# -----------------------------------------------------------------------------

ax = axes[1, 0]

ax.scatter(x, y, color='steelblue', alpha=0.6, s=45)
ax.plot(x_plot, y_plot_pow, 'm-', linewidth=2.2)

ax.set_title(
    f'4. Степенная модель\nR² = {R2_pow:.4f}',
    fontweight='bold'
)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# (5) Линеаризация степенной модели
# -----------------------------------------------------------------------------

ax = axes[1, 1]

ax.scatter(
    ln_x,
    ln_y,
    color='purple',
    alpha=0.7,
    s=45
)

ax.plot(
    ln_x,
    ln_a_pw + b_pw * ln_x,
    'r-',
    linewidth=2.2
)

ax.set_title(
    f'5. Линеаризация\nR²(log) = {R2_pw_log:.4f}',
    fontweight='bold'
)

ax.set_xlabel('ln(x)')
ax.set_ylabel('ln(y)')
ax.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# (6) Остатки линейной модели
# -----------------------------------------------------------------------------

ax = axes[1, 2]

ax.scatter(
    x,
    e_lin,
    color='tomato',
    alpha=0.7,
    s=45
)

ax.axhline(0, color='black', linestyle='--')

ax.set_title(
    '6. Остатки линейной модели',
    fontweight='bold'
)

ax.set_xlabel('x')
ax.set_ylabel('e = y − ŷ')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# СОХРАНЕНИЕ
plt.savefig(
    'rgr3_a4_plots.png',
    dpi=300,
    bbox_inches='tight'
)

print("\nГрафики сохранены: rgr3_a4_plots.png")

# =============================================================================
# 9. ФИНАЛЬНЫЙ ВЫВОД
# =============================================================================

print("\n" + "=" * 70)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 70)

print(f"""
Исследовалась зависимость относительной производительности вычислительной
системы от максимального объёма каналов памяти.

Количество наблюдений: n = {n}

Построены три модели:

1. Линейная:
   ŷ = {b0:.4f} + {b1:.4f}·x

2. Квадратичная:
   ŷ = {c0_q:.4f} + {c1_q:.4f}·x + ({c2_q:.6f})·x²

3. Степенная:
   ŷ = {a_pw:.4f}·x^{b_pw:.4f}

Сравнение моделей:

Линейная:
  R² = {R2_lin:.4f}
  RMSE = {RMSE_lin:.2f}
  A = {A_lin:.2f}%

Квадратичная:
  R² = {R2_quad:.4f}
  RMSE = {RMSE_quad:.2f}
  A = {A_quad:.2f}%

Степенная:
  R² = {R2_pow:.4f}
  RMSE = {RMSE_pow:.2f}
  A = {A_pow:.2f}%

Коэффициент наклона линейной модели статистически значим:
|t| = {t_obs:.4f} > t_крит = {t_crit:.4f}

Прогноз при x* = {x_star}:

Линейная модель:
  ŷ = {y_pred_lin:.2f}

Квадратичная модель:
  ŷ = {y_pred_quad:.2f}

Степенная модель:
  ŷ = {y_pred_pow:.2f}
""")
