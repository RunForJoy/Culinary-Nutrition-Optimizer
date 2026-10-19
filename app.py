import json
import os
import pandas as pd
import streamlit as st
from src.recipe import RecipeNutritionist

# Настройка конфигурации страницы Streamlit
st.set_page_config(
    page_title="Culinary Nutritionist & Optimizer", page_icon="📊", layout="wide"
)


# Оптимизированная функция загрузки текстовых инструкций рецептов из сырого JSON
@st.cache_data
def load_raw_recipes_text():
    # Строим путь относительно расположения app.py наружу и вглубь материалов
    json_path = os.path.join(
        "materials", "Epicurious_dataset", "full_format_recipes.json"
    )
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        # Собираем быстрый хэш-словарь {Название: [Список шагов]} для мгновенного поиска за O(1)
        text_db = {}
        for item in raw_data:
            if "title" in item and "directions" in item:
                text_db[item["title"].lower().strip()] = item["directions"]
        return text_db
    return {}


# Инициализация ядра бэкенда с кэшированием памяти
@st.cache_resource
def load_nutritionist():
    return RecipeNutritionist()


try:
    nutritionist = load_nutritionist()
    recipes_text_db = load_raw_recipes_text()
except Exception as e:
    st.error(f"Ошибка загрузки конфигурационных артефактов бэкенда: {e}")
    st.stop()

st.title("📊 Culinary Nutritionist & Recipe Matcher")
st.markdown(
    "Промышленная система оценки сочетаемости ингредиентов, расчета нутрициологического профиля FDA и стохастической оптимизации рационов."
)

tab_forecast, tab_menu = st.tabs(
    ["🔮 Прогноз сочетаемости и КБЖУ", "📅 Генератор суточного меню"]
)

with tab_forecast:
    st.header("Анализ пользовательского состава ингредиентов")
    available_ingredients = list(nutritionist.search_matrix.columns)

    user_inputs = st.multiselect(
        "Выберите или введите ингредиенты блюда:",
        options=available_ingredients,
        help="Поиск работает по подстрокам и точному совпадению с Whitelist-матрицей из 334 признаков.",
    )

    if st.button("Запустить инференс анализа", type="primary"):
        if not user_inputs:
            st.warning("Пожалуйста, выберите хотя бы один ингредиент для анализа.")
        else:
            st.subheader("I. OUR FORECAST")
            with st.spinner("Модели рассчитывают гастрономический вектор..."):
                prediction = nutritionist.predict_rating_class(user_inputs)

            if prediction == "great":
                st.success(
                    "Excellent choice! Our machine learning model predicts this dish will taste great!"
                )
            elif prediction == "so-so":
                st.info(
                    "The combination is so-so. It could be edible, but don't expect a culinary masterpiece."
                )
            elif prediction == "bad":
                st.error(
                    "You might find it tasty, but in our opinion, it is a bad idea to have a dish with that list of ingredients."
                )
            else:
                st.warning(f"Статус инференса: {prediction}")

                # =====================================================================
            # II. NUTRITION FACTS (ПОКАЗ ПО КАЖДОМУ ИНГРЕДИЕНТУ ОТДЕЛЬНО)
            # =====================================================================
            st.subheader("II. NUTRITION FACTS (% Daily Value)")

            with st.spinner("Агрегирование данных из матриц..."):
                has_any_nutrition = False

                for ing in user_inputs:
                    df_nut = nutritionist.get_nutrition_facts(ing)

                    if df_nut is not None and not df_nut.empty:
                        has_any_nutrition = True
                        row = df_nut.iloc[0]

                        # Собираем нутриенты строго для ТЕКУЩЕГО ингредиента
                        current_ingredient_profile = {}
                        for col in df_nut.columns:
                            if "%DV" in col:
                                name = (
                                    col.replace("_%DV", "")
                                    .replace("Total_", "Total ")
                                    .replace("Saturated_", "Saturated ")
                                    .replace("Dietary_", "Dietary ")
                                )
                                current_ingredient_profile[name] = round(row[col], 2)

                        if current_ingredient_profile:
                            # Создаем DataFrame для текущего продукта
                            nutr_df = pd.DataFrame(
                                list(current_ingredient_profile.items()),
                                columns=["Нутриент", "Значение (%DV)"],
                            )
                            # Сортируем по убыванию внутри этого продукта
                            nutr_df = nutr_df.sort_values(
                                by="Значение (%DV)", ascending=False
                            )
                            nutr_df = nutr_df.set_index("Нутриент")

                            # Выводим в отдельный раскрывающийся блок
                            with st.expander(f"📖 {ing.capitalize()}"):
                                st.dataframe(nutr_df, use_container_width=True)

                if not has_any_nutrition:
                    st.info(
                        "Данные о пищевой ценности для выбранных позиций отсутствуют."
                    )

            st.subheader("III. TOP-3 SIMILAR RECIPES")
            with st.spinner("Расчет косинусного сходства..."):
                similar_recipes = nutritionist.get_top_recipes(user_inputs)

            if not similar_recipes:
                st.info("Похожие рецепты на основе пересечения признаков не найдены.")
            else:
                for idx, recipe in enumerate(similar_recipes):
                    recipe_title = recipe["title"]
                    st.markdown(
                        f"**{idx + 1}. {recipe_title}** — Оригинальный рейтинг: `{recipe['rating']:.1f}`"
                    )

                    # Локальный вывод шагов приготовления из JSON для похожих рецептов
                    clean_title = recipe_title.lower().strip()
                    if clean_title in recipes_text_db:
                        with st.expander("📖 Посмотреть шаги приготовления:"):
                            for step_idx, step in enumerate(
                                recipes_text_db[clean_title]
                            ):
                                st.write(f"**Шаг {step_idx + 1}:** {step}")
                    st.markdown("---")

with tab_menu:
    st.header("Стохастическая оптимизация суточного меню (Monte Carlo)")
    st.markdown(
        "Алгоритм совершает случайный вероятностный поиск по изолированным пулам трапез, максимизируя суммарный рейтинг блюд и жестко контролируя лимит КБЖУ до 100% суточной нормы FDA одновременно."
    )

    if st.button("Сгенерировать сбалансированное меню", type="primary"):
        with st.spinner("Генерация оптимального рациона методом Монте-Карло..."):
            daily_menu = nutritionist.generate_daily_menu()

        if not daily_menu:
            st.error(
                "Не удалось подобрать рацион, удовлетворяющий жестким ограничениям КБЖУ <= 100% DV."
            )
        else:
            cols = st.columns(3)
            meals_config = [
                ("BREAKFAST", "🍳 Breakfast"),
                ("LUNCH", "🥪 Lunch"),
                ("DINNER", "🍽️ Dinner"),
            ]

            for idx, (back_key, display_name) in enumerate(meals_config):
                with cols[idx]:
                    st.subheader(display_name)
                    meal_data = daily_menu.get(back_key, {})
                    meal_title = meal_data.get("title", "Unknown")

                    st.metric(
                        label="Рейтинг блюда",
                        value=f"{meal_data.get('rating', 0.0):.2f}",
                    )
                    st.markdown(f"**{meal_title}**")

                    # Выпадающий блок №1: Оригинальные ингредиенты из матрицы признаков
                    with st.expander("Посмотреть состав ингредиентов:"):
                        for ing in meal_data.get("ingredients", []):
                            st.write(f"- {ing}")

                    clean_meal_title = meal_title.lower().strip()
                    if clean_meal_title in recipes_text_db:
                        with st.expander("📖 Пошаговое руководство (Directions):"):
                            for step_idx, step in enumerate(
                                recipes_text_db[clean_meal_title]
                            ):
                                st.write(f"**Шаг {step_idx + 1}:** {step}")
                    else:
                        st.caption(
                            "Подробный текст инструкции для данного архивного блюда отсутствует в локальном JSON."
                        )
