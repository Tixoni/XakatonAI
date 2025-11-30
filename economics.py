import math

investment = 8_000_000           # инвестиции
annual_savings_before = 8_000_000  # текущие затраты
annual_savings_after = 2_000_000   # затраты после внедрения
opex = 1_000_000                   # операционные расходы
years = 10                         # горизонт расчета
discount_rate = 0.10               # ставка дисконтирования 10%

# Экономия: до – после
annual_economy = annual_savings_before - annual_savings_after  # 6 млн/год

# Чистый денежный поток
annual_cashflow = annual_economy - opex  # 5 млн/год

def npv(investment, cf, r, years):
    """Расчет NPV"""
    total_pv = 0
    for t in range(1, years + 1):
        total_pv += cf / ((1 + r) ** t)
    return total_pv - investment


def irr(investment, cf, years):
    """Поиск IRR итерационно методом дихотомии"""
    low, high = 0.0, 1.0  # 0%–100%

    for _ in range(1000):
        mid = (low + high) / 2
        npv_mid = npv(investment, cf, mid, years)
        if npv_mid > 0:
            low = mid
        else:
            high = mid

    return mid


def dpp(investment, cf, r, years):
    """Дисконтированный срок окупаемости"""
    cumulative = 0.0

    for t in range(1, years + 1):
        pv = cf / ((1 + r) ** t)
        if cumulative + pv >= investment:
            remaining = investment - cumulative
            return t - 1 + (remaining / pv)
        cumulative += pv

    return None  # не окупается в пределах срока


def dpi(investment, cf, r, years):
    """Discounted Profitability Index"""
    total_pv = 0
    for t in range(1, years + 1):
        total_pv += cf / ((1 + r) ** t)
    return total_pv / investment


NPV = npv(investment, annual_cashflow, discount_rate, years)
IRR = irr(investment, annual_cashflow, years)
DPP = dpp(investment, annual_cashflow, discount_rate, years)
DPI = dpi(investment, annual_cashflow, discount_rate, years)


print("------ РЕЗУЛЬТАТЫ ИНВЕСТИЦИОННОГО АНАЛИЗА ------")
print(f"Горизонт расчета: {years} лет")
print(f"Инвестиции: {investment:,.0f} ₽")
print(f"Ежегодный чистый денежный поток: {annual_cashflow:,.0f} ₽")
print(f"Ставка дисконтирования: {discount_rate*100:.1f}%")
print()
print(f"NPV: {NPV:,.2f} ₽")
print(f"IRR: {IRR*100:.2f}%")
print(f"DPP (дисконтированный срок окупаемости): {DPP:.2f} лет")
print(f"DPI: {DPI:.2f}")