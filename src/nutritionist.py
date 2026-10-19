#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from recipe import RecipeNutritionist
except ImportError:
    print("ошибка: не найден файл recipe.py в текущей директории!")
    sys.exit(1)


def main():
    try:
        if len(sys.argv) < 2:
            print(
                "\nНекорректный запуск программы. Варианты запуска:"
                "\n1. Необходимо передать как минимум один ингредиент после названия программы:\nПример: python nutritionist.py ingredient1, ingredient2, ..."
                "\n2. Передать название программы с ключом --menu для получения меню на день \nПример: python nutritionist.py --menu"
            )
            sys.exit(0)

        elif len(sys.argv) == 2 and sys.argv[1].strip() == "--menu":
            # Вызываем класс без аргументов — он сам найдет правильные пути через base_dir
            nutritionist = RecipeNutritionist()

            # вызываем оптимизатор Монте-Карло и парсим итоговую структуру данных
            menu = nutritionist.generate_daily_menu()
            for meal_type, data in menu.items():
                print(f"\n{meal_type}")
                print("-" * 21)
                print(f"{data['title']} (rating: {data['rating']:.3f})")
                print("Ingredients:")
                for ing in data["ingredients"]:
                    print(f"- {ing}")

                print("Nutrients:")
                # извлекаем значения из вложенного JSON-объекта 'nutrients', нормированного по стандартам FDA %DV
                print(f"calories - {data['nutrients']['calories']}% of Daily Value")
                print(f"protein - {data['nutrients']['protein']}% of Daily Value")
                print(f"total fat - {data['nutrients']['fat']}% of Daily Value")
                print(f"URL: {data['url']}")

            sys.exit(0)

        raw_input = " ".join(sys.argv[1:])
        ingredients = [i.strip() for i in raw_input.split(",") if i.strip()]

        # ингредиент короче 3 символов точно мусор
        fake_ingredients = [ing for ing in ingredients if len(ing.lower().strip()) < 3]

        if fake_ingredients:
            print("\nI. OUR FORECAST")
            missing_str = ", ".join(fake_ingredients)
            print(
                f"the following ingredients are missing in our database: {missing_str}"
            )
            print("III. TOP-3 SIMILAR RECIPES")
            print("Похожие рецепты не найдены.")
            print()
            sys.exit(0)

        # Вызываем класс без аргументов для стандартного инференса сочетаемости
        nutritionist = RecipeNutritionist()

        # ========================================================================
        # OUR FORECAST
        # ========================================================================

        print("\nI. OUR FORECAST")
        res = nutritionist.predict_rating_class(ingredients)

        # метод класса спроектирован так, что если ингредиент прошёл базовый фильтр длины (len >= 3),
        # но полностью отсутствует в нашем кулинарном 334-колонном справочнике, он возвращает строковый маркер ошибки (например, "missing: saffron"). Поэтому осуществляем перехват отсутствия, изолируя ошибку, выводим список неизвестных системе фич и выходим через sys.exit(0).
        if str(res).startswith("missing:"):
            missing_list = res.replace("missing:", "").strip()
            print(
                f"the following ingredients are missing in our database: {missing_list}"
            )
            print("III. TOP-3 SIMILAR RECIPES")
            print("Похожие рецепты не найдены.")
            print()
            sys.exit(0)

        elif res == "bad":
            print(
                "You might find it tasty, but in our opinion, it is a bad idea to have a"
            )
            print("dish with that list of ingredients.")
            print()
            # if len(ingredients) <= 5:
            #     sys.exit(0)
        elif res == "so-so":
            print("The combination is so-so. It could be edible, but don't expect a")
            print("culinary masterpiece.")
        else:
            print(
                "Excellent choice! Our machine learning model predicts this dish will"
            )
            print("taste great!")

        # =====================================================================
        # NUTRITION FACTS (%DV ДЛЯ КАЖДОГО ПРОДУКТА)
        # =====================================================================
        print("\nII. NUTRITION FACTS")

        for ing in ingredients:
            df_nut = nutritionist.get_nutrition_facts(ing)

            if df_nut is None or df_nut.empty:
                print(f"{ing.capitalize()}")
                print("Данные о пищевой ценности отсутствуют в базе.")
                continue

            row = df_nut.iloc[0]
            # вывожу на печать исходное слово пользователя (например, milk или jam)
            print(f"{str(ing).strip().capitalize()}")

            for col in df_nut.columns:
                if "%DV" in col:
                    name = (
                        col.replace("_%DV", "")
                        .replace("Total_", "Total ")
                        .replace("Saturated_", "Saturated ")
                        .replace("Dietary_", "Dietary ")
                    )

                    val = row[col]
                    val_int = int(round(val)) if pd.notnull(val) else 0
                    print(f"{name} - {val_int}% of Daily Value")
            print()

        # =========================================================================
        # TOP-3 SIMILAR RECIPES
        # ===========================================================================

        print("\nIII. TOP-3 SIMILAR RECIPES:")
        top_recipes = nutritionist.get_top_recipes(ingredients)

        if not top_recipes:
            print("Похожие рецепты не найдены.")
        else:
            for r in top_recipes:
                print(
                    f"- {r['title'].strip()}, rating: {r['rating']:.1f}, URL:\n  {r['url']}"
                )

    except Exception as global_error:
        print(f"\nСистемный сбой во время выполнения программы: {global_error}")
        sys.exit(0)


if __name__ == "__main__":
    main()
