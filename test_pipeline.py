#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import sys
import pandas as pd
import numpy as np

try:
    from src.recipe import RecipeNutritionist

    nutritionist_available = True
except ImportError:
    print(
        "Предупреждение: src/recipe.py не найден в контексте, тесты данных будут пропущены."
    )
    nutritionist_available = False


def run_test_header(title):
    print("\n" + "=" * 70)
    print(f" ТЕСТ: {title}")
    print("=" * 70)


def main():
    print("=== ИНИЦИАЛИЗАЦИЯ СКВОЗНОГО АВТОТЕСТА ПРОЕКТА ===\n")

    # Флаг контроля сквозного прохождения тестов
    all_tests_passed = True

    # -------------------------------------------------------------------------
    # ТЕСТ 1: Проверка Whitelist ingredients
    # -------------------------------------------------------------------------
    run_test_header(
        "Валидация Whitelist матрицы признаков (perfect_ingredient_columns.json)"
    )
    json_path = "datasets/perfect_ingredient_columns.json"

    if not os.path.exists(json_path):
        print(f"ОШИБКА: Файл {json_path} не найден!")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        perfect_features = json.load(f)

    df_features = pd.DataFrame(columns=perfect_features)
    garbage_list = [
        "title",
        "leftovers",
        "california",
        "dominican republic",
        "protein",
        "low cholesterol",
        "pan-fry",
        "fruit",
        "stew",
        "braise",
    ]
    has_garbage = df_features.columns.isin(garbage_list).any()

    has_milk = any("milk" in col.lower() for col in df_features.columns)
    has_jam = any("jam" in col.lower() for col in df_features.columns)
    required_check = has_milk and has_jam

    if not has_garbage:
        print(
            "Успех: Лингвистический мусор и географические теги полностью отфильтрованы."
        )
    else:
        print("СБОЙ: В признаках обнаружен запрещенный мусор!")
        all_tests_passed = False

    if required_check:
        print(
            "Успех: Обязательные маркеры чеклиста (milk, jam) присутствуют в матрице."
        )
    else:
        print("СБОЙ: Обязательные маркеры (milk, jam) отсутствуют!")
        all_tests_passed = False

    # -------------------------------------------------------------------------
    # ТЕСТ 2: Верификация категорий суточного меню
    # -------------------------------------------------------------------------
    run_test_header("Верификация детерминированной категоризации блюд в пуле рецептов")
    pool_path = "datasets/recipes_full_pool.csv"

    if os.path.exists(pool_path):
        df_pool = pd.read_csv(pool_path)
        test_titles = {
            "Swedish Cardamom Rolls": "breakfast",
            "Grilled Ham and Gouda Sandwiches with Frisée and Caramelized Onions": "lunch",
            "Melon and Berry Compote": "dinner",
        }

        for title, col_type in test_titles.items():
            matched_rows = df_pool[
                df_pool["title"].str.lower().str.strip() == title.lower().strip()
            ]
            if not matched_rows.empty:
                row = matched_rows.iloc[0]
                b_val = row.get("is_breakfast", row.get("breakfast", 0))
                l_val = row.get("is_lunch", row.get("lunch", 0))
                d_val = row.get("is_dinner", row.get("dinner", 0))

                print(f"Блюдо: {title}")
                print(f"  -> Завтрак: {b_val} | Обед: {l_val} | Ужин: {d_val}")

                if (
                    (col_type == "breakfast" and b_val == 1)
                    or (col_type == "lunch" and l_val == 1)
                    or (col_type == "dinner" and d_val == 1)
                ):
                    print(f"  Статус {col_type.upper()} подтвержден.")
                else:
                    print(f"  СБОЙ: Ошибка распределения флагов для '{title}'")
                    all_tests_passed = False
            else:
                print(f"СБОЙ: Рецепт '{title}' не найден в пуле.")
                all_tests_passed = False
    else:
        print(f"СБОЙ: База по пути {pool_path} не найдена.")
        all_tests_passed = False

    # -------------------------------------------------------------------------
    # ТЕСТ 3: Интеграционный тест бизнес-логики класса RecipeNutritionist
    # -------------------------------------------------------------------------
    if nutritionist_available:
        run_test_header("Интеграционный тест бизнес-логики класса RecipeNutritionist")
        try:
            client = RecipeNutritionist()

            # Тест А: Guardrails
            print("Проверка слоя кулинарной безопасности Guardrails:")
            test_dangerous = ["saffron", "gin", "chile pepper"]
            res_dangerous = client.predict_rating_class(test_dangerous)
            print(f"  Ввод: {', '.join(test_dangerous)}")
            print(f"  Вердикт модели: {res_dangerous}")
            if res_dangerous == "bad":
                print(
                    "  Успех: Деструктивная комбинация заблокирована слоем бизнес-правил."
                )
            else:
                print("  СБОЙ: Модель не заблокировала деструктивную пару!")
                all_tests_passed = False

            # Тест Б: Монте-Карло
            print("\nПроверка стохастической оптимизации рационов (Monte Carlo):")
            menu = client.generate_daily_menu()

            if menu:
                print("  Успех: Суточный рацион успешно сгенерирован алгоритмом.")

                b_meal = menu.get("BREAKFAST", {})
                l_meal = menu.get("LUNCH", {})
                d_meal = menu.get("DINNER", {})

                print(
                    f"  [Завтрак]: {b_meal.get('title')} (Рейтинг: {b_meal.get('rating'):.2f})"
                )
                print(
                    f"  [Обед]: {l_meal.get('title')} (Рейтинг: {l_meal.get('rating'):.2f})"
                )
                print(
                    f"  [Ужин]: {d_meal.get('title')} (Рейтинг: {d_meal.get('rating'):.2f})"
                )

                # Считаем сумму макронутриентов и калорий %DV напрямую из вложенных словарей бэкенда
                b_nutr = b_meal.get("nutrients", {})
                l_nutr = l_meal.get("nutrients", {})
                d_nutr = d_meal.get("nutrients", {})

                total_calories = (
                    b_nutr.get("calories", 0)
                    + l_nutr.get("calories", 0)
                    + d_nutr.get("calories", 0)
                )
                total_fat = (
                    b_nutr.get("fat", 0) + l_nutr.get("fat", 0) + d_nutr.get("fat", 0)
                )
                total_protein = (
                    b_nutr.get("protein", 0)
                    + l_nutr.get("protein", 0)
                    + d_nutr.get("protein", 0)
                )

                # Выводим на экран полную, честную информацию по КБЖУ согласно ТЗ
                print(
                    f"  Суммарные макронутриенты %DV: Calories={total_calories}%, Fat={total_fat}%, Protein={total_protein}%"
                )

                # Жесткая сквозная проверка соблюдения лимитов FDA по всем трем параметрам одновременно
                if total_calories <= 100 and total_fat <= 100 and total_protein <= 100:
                    print("  Успех: Жесткие лимиты FDA КБЖУ <= 100% строго соблюдены.")
                else:
                    print(
                        "  СБОЙ: Обнаружено превышение суточного лимита нутриентов > 100%!"
                    )
                    all_tests_passed = False

            else:
                print("  СБОЙ: Алгоритм Монте-Карло вернул None.")
                all_tests_passed = False

        except Exception as e:
            print(f"СБОЙ: Системное исключение в процессе инференса: {e}")
            all_tests_passed = False

    # =========================================================================
    # ФИНАЛЬНЫЙ СУММАРНЫЙ ВЕРДИКТ СКРИПТА
    # =========================================================================
    print("\n" + "=" * 70)
    if all_tests_passed:
        print(" СТАТУС: ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО (SUCCESS)")
        print("=" * 70)
        sys.exit(0)
    else:
        print(" СТАТУС: ОБНАРУЖЕНЫ ОШИБКИ ПРИ ВАЛИДАЦИИ (FAILED)")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
