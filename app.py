import streamlit as st
import pandas as pd
import re
from random import choice
from streamlit_echarts import st_echarts
import matplotlib.pyplot as plt

# Load the dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Lookup table for ingredient calories (per gram)
INGREDIENT_CALORIES = {
    "rice": 1.3,  # calories per gram
    "wheat": 3.4,
    "sugar": 4,
    "salt": 0,
    "oil": 8.8,
    "butter": 7.2,
    "milk": 0.6,
    "flour": 3.6,
    "chicken": 2.3,
    "fish": 2.0,
    "egg": 1.5,
    "vegetable": 0.3,
    "potato": 0.8,
    "tomato": 0.2,
    "onion": 0.4,
    "cheese": 4,
    "bread": 2.7,
    "pasta": 3.7,
    "lentils": 1.2,
    "beans": 1.5,
    "chickpeas": 3.6,
    "yogurt": 0.6,
    "curd": 0.6,
    "ghee": 9,
    "honey": 3,
    "nuts": 6,
    "seeds": 5,
    "spices": 0.1,
    "herbs": 0.1,
}

# Substitution database for healthier alternatives
SUBSTITUTIONS = {
    "fried chicken": "grilled chicken",
    "sour cream": "Greek yogurt",
    "white rice": "brown rice",
    "white bread": "whole grain bread",
    "mayonnaise": "avocado spread",
    "sugar": "honey",
    "butter": "olive oil",
    "potato chips": "kale chips",
    "ice cream": "frozen yogurt",
    "cream cheese": "low-fat cottage cheese",
    "bacon": "turkey bacon",
    "soda": "Masala Soda Shikanji Recipe",
    "pasta": "zucchini noodles",
    "fried fish": "baked fish",
    "cream": "coconut milk",
    "cheese": "low-fat cheese",
    "milk chocolate": "dark chocolate",
    "ground beef": "ground turkey",
    "flour": "almond flour",
    "burger": "Veggie Chop Burger Recipe",
    "pizza": "Multigrain Pizza with Roasted Vegetables" ,
    "fries": "baked sweet potato fries",
    "fried shrimp": "grilled shrimp",
    "nachos": "Papad Nachos With Salsa Recipe",
    "donuts": "whole grain muffins with fruit",
    "pancakes": "Savoury Carrot And Coriander Pancakes Recipe",
    "waffles": "Healthy Oats Waffles Recipe",
    "milkshake": "smoothie with yogurt and fruit",
    "fried rice": "cauliflower fried rice",
    "mashed potatoes": "mashed cauliflower",
    "mac and cheese": "cauliflower mac and cheese",
    "fried noodles": "stir-fried zucchini noodles",
    "fried dumplings": "steamed dumplings",
    "fried spring rolls": "baked spring rolls",
    "fried calamari": "grilled calamari",
    "fried tofu": "baked tofu",
    "fried samosa": "baked samosa",
    "fried pakora": "baked pakora",
    "fried tempura": "steamed vegetables with light dipping sauce",
    "fried onion rings": "baked onion rings",
    "fried mozzarella sticks": "baked mozzarella sticks",
    "fried chicken wings": "baked chicken wings",
    "fried fish and chips": "baked fish with roasted potatoes",
    "fried calamari rings": "grilled calamari rings",
    "fried plantains": "baked plantains",
    "fried empanadas": "baked empanadas",
    "fried pierogi": "baked pierogi",
    "fried wontons": "steamed wontons",
    "fried crab cakes": "baked crab cakes",
    "fried falafel": "baked falafel",
    "fried arancini": "baked arancini",
    "fried churros": "baked churros",
    "fried beignets": "baked beignets",
    "fried doughnuts": "baked doughnuts",
    "fried croquettes": "baked croquettes",
    "fried tater tots": "baked tater tots",
    "fried hash browns": "baked hash browns",
    "fried corn dogs": "baked corn dogs",
    "fried hush puppies": "baked hush puppies",
    "fried jalapeno poppers": "baked jalapeno poppers",
    "fried mochi": "baked mochi",
    "fried banana fritters": "baked banana fritters",
    "fried apple fritters": "baked apple fritters",
    "fried zucchini sticks": "baked zucchini sticks",
    "fried eggplant": "baked eggplant",
    "fried okra": "baked okra",
    "fried green tomatoes": "baked green tomatoes",
    "fried mushrooms": "baked mushrooms",
    "fried artichoke hearts": "baked artichoke hearts",
    "fried pickles": "baked pickles",
    "fried cheese curds": "baked cheese curds",
    "fried ravioli": "baked ravioli",
    "fried gnocchi": "baked gnocchi",
}

# Function to suggest healthier alternatives
def suggest_healthier_alternative(dish):
    return SUBSTITUTIONS.get(dish.lower(), "No alternative found. Consider reducing portion size or cooking method.")

# Function to estimate calories from ingredients
def estimate_calories(ingredients):
    total_calories = 0
    for ingredient in ingredients.split(','):
        ingredient = ingredient.strip().lower()
        # Extract quantity (e.g., "100g rice" -> 100)
        quantity_match = re.search(r"(\d+)\s*g", ingredient)
        if quantity_match:
            quantity = float(quantity_match.group(1))
        else:
            quantity = 50  # default to 50g if no quantity is specified
        # Find matching ingredient in lookup table
        for key, value in INGREDIENT_CALORIES.items():
            if key in ingredient:
                total_calories += quantity * value
                break
    return round(total_calories)

# Function to find recipes based on dietary preferences
def find_recipes(df, diet_preference):
    if diet_preference == "Any":
        return df.sample(min(3, len(df))).to_dict('records')
    else:
        filtered_recipes = df[df['Diet'].str.contains(diet_preference, case=False, na=False)]
        return filtered_recipes.sample(min(3, len(filtered_recipes))).to_dict('records')

# Mock function for ImageFinder
def find_image(recipe_name):
    return "https://via.placeholder.com/150"

# Streamlit App
st.set_page_config(page_title="Health & Nutrition App", page_icon="💪", layout="wide")

# Load datasets
df = load_dataset('archanas_kitchenmain.csv')
food_df = pd.read_csv('food.csv')

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Automatic Diet Plan", "Nutrition Analysis", "Healthier Alternatives"])

if page == "Automatic Diet Plan":
    class Person:
        def __init__(self, age, height, weight, gender, activity, meals_calories_perc, weight_loss, diet_preference):
            self.age = age
            self.height = height
            self.weight = weight
            self.gender = gender
            self.activity = activity
            self.meals_calories_perc = meals_calories_perc
            self.weight_loss = weight_loss
            self.diet_preference = diet_preference

        def calculate_bmi(self):
            bmi = round(self.weight / ((self.height / 100) ** 2), 2)
            return bmi

        def display_result(self):
            bmi = self.calculate_bmi()
            bmi_string = f'{bmi} kg/m²'
            if bmi < 18.5:
                category = 'Underweight'
                color = 'Red'
            elif 18.5 <= bmi < 25:
                category = 'Normal'
                color = 'Green'
            elif 25 <= bmi < 30:
                category = 'Overweight'
                color = 'Yellow'
            else:
                category = 'Obesity'
                color = 'Red'
            return bmi_string, category, color

        def calculate_bmr(self):
            if self.gender == 'Male':
                bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
            else:
                bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
            return bmr

        def calories_calculator(self):
            activites = ['Little/no exercise', 'Light exercise', 'Moderate exercise (3-5 days/wk)', 'Very active (6-7 days/wk)', 'Extra active (very active & physical job)']
            weights = [1.2, 1.375, 1.55, 1.725, 1.9]
            weight = weights[activites.index(self.activity)]
            maintain_calories = self.calculate_bmr() * weight
            return maintain_calories

        def generate_recommendations(self, df):
            recommendations = []
            for meal in self.meals_calories_perc:
                recommended_recipes = find_recipes(df, self.diet_preference)
                for recipe in recommended_recipes:
                    recipe['Calories'] = estimate_calories(recipe['Ingredients'])
                recommendations.append(recommended_recipes)
            for recommendation in recommendations:
                for recipe in recommendation:
                    recipe['image_link'] = find_image(recipe['RecipeName'])
            return recommendations

    class Display:
        def __init__(self):
            self.plans = ["Maintain weight", "Mild weight loss", "Weight loss", "Extreme weight loss"]
            self.weights = [1, 0.9, 0.8, 0.6]
            self.losses = ['-0 kg/week', '-0.25 kg/week', '-0.5 kg/week', '-1 kg/week']
            pass

        def display_bmi(self, person):
            st.header('BMI CALCULATOR')
            bmi_string, category, color = person.display_result()
            st.metric(label="Body Mass Index (BMI)", value=bmi_string)
            new_title = f'<p style="font-family:sans-serif; color:{color}; font-size: 25px;">{category}</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            st.markdown(
                """
                Healthy BMI range: 18.5 kg/m² - 25 kg/m².
                """)

        def display_calories(self, person):
            st.header('CALORIES CALCULATOR')
            maintain_calories = person.calories_calculator()
            st.write('The results show a number of daily calorie estimates that can be used as a guideline for how many calories to consume each day to maintain, lose, or gain weight at a chosen rate.')
            for plan, weight, loss, col in zip(self.plans, self.weights, self.losses, st.columns(4)):
                with col:
                    st.metric(label=plan, value=f'{round(maintain_calories * weight)} Calories/day', delta=loss, delta_color="inverse")

        def display_recommendation(self, person, recommendations):
            st.header('DIET RECOMMENDATOR')
            with st.spinner('Generating recommendations...'):
                meals = person.meals_calories_perc
                st.subheader('Recommended recipes:')
                for meal_name, column, recommendation in zip(meals, st.columns(len(meals)), recommendations):
                    with column:
                        st.markdown(f'##### {meal_name.upper()}')
                        for recipe in recommendation:
                            recipe_name = recipe['RecipeName']
                            expander = st.expander(recipe_name)
                            recipe_link = recipe['image_link']
                            recipe_img = f'<div><center><img src={recipe_link} alt={recipe_name}></center></div>'

                            expander.markdown(recipe_img, unsafe_allow_html=True)
                            expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Estimated Calories: {recipe["Calories"]}</h5>', unsafe_allow_html=True)
                            expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Ingredients:</h5>', unsafe_allow_html=True)
                            for ingredient in recipe['Ingredients'].split(','):
                                expander.markdown(f"""
                                            - {ingredient.strip()}
                                """)
                            expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Recipe Instructions:</h5>', unsafe_allow_html=True)
                            for instruction in recipe['Instructions'].split('.'):
                                if instruction.strip():
                                    expander.markdown(f"""
                                                - {instruction.strip()}
                                    """)
                            expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Cooking and Preparation Time:</h5>', unsafe_allow_html=True)
                            expander.markdown(f"""
                                    - Preparation Time: {recipe['PrepTimeInMins']}min
                                    - Servings        : {recipe['Servings']}
                                """)

        def display_meal_choices(self, person, recommendations):
            st.subheader('Choose your meal composition:')
            if len(recommendations) == 3:
                breakfast_column, launch_column, dinner_column = st.columns(3)
                with breakfast_column:
                    breakfast_choice = st.selectbox(f'Choose your breakfast:', [recipe['RecipeName'] for recipe in recommendations[0]])
                with launch_column:
                    launch_choice = st.selectbox(f'Choose your launch:', [recipe['RecipeName'] for recipe in recommendations[1]])
                with dinner_column:
                    dinner_choice = st.selectbox(f'Choose your dinner:', [recipe['RecipeName'] for recipe in recommendations[2]])
                choices = [breakfast_choice, launch_choice, dinner_choice]
            elif len(recommendations) == 4:
                breakfast_column, morning_snack, launch_column, dinner_column = st.columns(4)
                with breakfast_column:
                    breakfast_choice = st.selectbox(f'Choose your breakfast:', [recipe['RecipeName'] for recipe in recommendations[0]])
                with morning_snack:
                    morning_snack = st.selectbox(f'Choose your morning_snack:', [recipe['RecipeName'] for recipe in recommendations[1]])
                with launch_column:
                    launch_choice = st.selectbox(f'Choose your launch:', [recipe['RecipeName'] for recipe in recommendations[2]])
                with dinner_column:
                    dinner_choice = st.selectbox(f'Choose your dinner:', [recipe['RecipeName'] for recipe in recommendations[3]])
                choices = [breakfast_choice, morning_snack, launch_choice, dinner_choice]
            else:
                breakfast_column, morning_snack, launch_column, afternoon_snack, dinner_column = st.columns(5)
                with breakfast_column:
                    breakfast_choice = st.selectbox(f'Choose your breakfast:', [recipe['RecipeName'] for recipe in recommendations[0]])
                with morning_snack:
                    morning_snack = st.selectbox(f'Choose your morning_snack:', [recipe['RecipeName'] for recipe in recommendations[1]])
                with launch_column:
                    launch_choice = st.selectbox(f'Choose your launch:', [recipe['RecipeName'] for recipe in recommendations[2]])
                with afternoon_snack:
                    afternoon_snack = st.selectbox(f'Choose your afternoon:', [recipe['RecipeName'] for recipe in recommendations[3]])
                with dinner_column:
                    dinner_choice = st.selectbox(f'Choose your dinner:', [recipe['RecipeName'] for recipe in recommendations[4]])
                choices = [breakfast_choice, morning_snack, launch_choice, afternoon_snack, dinner_choice]

            # Calculate total calories for chosen meals
            total_calories = 0
            for choice, meals_ in zip(choices, recommendations):
                for meal in meals_:
                    if meal['RecipeName'] == choice:
                        total_calories += meal['Calories']

            # Display total calories in a bar chart
            st.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Total Calories in Selected Meals:</h5>', unsafe_allow_html=True)
            total_calories_graph_options = {
                "xAxis": {
                    "type": "category",
                    "data": ['Total Calories'],
                },
                "yAxis": {"type": "value"},
                "series": [
                    {
                        "data": [{"value": total_calories, "itemStyle": {"color": "#33FF8D"}}],
                        "type": "bar",
                    }
                ],
            }
            st_echarts(options=total_calories_graph_options, height="400px")

    # Initialize session state
    if 'generated' not in st.session_state:
        st.session_state.generated = False
        st.session_state.recommendations = None
        st.session_state.person = None
        st.session_state.weight_loss_option = None

    # Main page content
    st.title("Automatic Diet Plan Generator")
    st.write("Get personalized diet recommendations based on your profile")

    display = Display()
    
    with st.form("diet_plan_form"):
        st.write("Please fill in your details to generate a personalized diet plan")
        age = st.number_input('Age', min_value=2, max_value=120, step=1)
        height = st.number_input('Height(cm)', min_value=50, max_value=300, step=1)
        weight = st.number_input('Weight(kg)', min_value=10, max_value=300, step=1)
        gender = st.radio('Gender', ('Male', 'Female'))
        activity = st.select_slider('Activity Level', options=['Little/no exercise', 'Light exercise', 'Moderate exercise (3-5 days/wk)', 'Very active (6-7 days/wk)', 'Extra active (very active & physical job)'])
        option = st.selectbox('Choose your weight loss plan:', display.plans)
        st.session_state.weight_loss_option = option
        weight_loss = display.weights[display.plans.index(option)]
        diet_preference = st.selectbox('Choose your dietary preference:', ['Any', 'Vegetarian', 'Vegan', 'Gluten-Free', 'Non-Vegetarian'])
        number_of_meals = st.slider('Meals per day', min_value=3, max_value=5, step=1, value=3)
        if number_of_meals == 3:
            meals_calories_perc = {'breakfast': 0.35, 'lunch': 0.40, 'dinner': 0.25}
        elif number_of_meals == 4:
            meals_calories_perc = {'breakfast': 0.30, 'morning snack': 0.05, 'lunch': 0.40, 'dinner': 0.25}
        else:
            meals_calories_perc = {'breakfast': 0.30, 'morning snack': 0.05, 'lunch': 0.40, 'afternoon snack': 0.05, 'dinner': 0.20}
        
        submitted = st.form_submit_button("Generate Diet Plan")
        
        if submitted:
            st.session_state.generated = True
            person = Person(age, height, weight, gender, activity, meals_calories_perc, weight_loss, diet_preference)
            with st.spinner('Generating recommendations...'):
                recommendations = person.generate_recommendations(df)
                st.session_state.recommendations = recommendations
                st.session_state.person = person

    if st.session_state.generated:
        st.success('Diet plan generated successfully!')
        display.display_bmi(st.session_state.person)
        display.display_calories(st.session_state.person)
        display.display_recommendation(st.session_state.person, st.session_state.recommendations)
        display.display_meal_choices(st.session_state.person, st.session_state.recommendations)

elif page == "Nutrition Analysis":
    st.title("Nutrition Analysis")
    st.write("Analyze the nutritional content of various foods")
    
    if food_df is not None:
        # Food selection
        st.subheader("Select foods to analyze:")
        selected_foods = st.multiselect(
            "Choose foods from the list below",
            food_df['Food_items'].unique(),
            default=food_df['Food_items'].iloc[:2].tolist()
        )
        
        if selected_foods:
            # Filter the dataframe for selected foods
            selected_data = food_df[food_df['Food_items'].isin(selected_foods)]
            
            # Display raw data
            st.subheader("Nutritional Data")
            st.dataframe(selected_data)
            
            # Calculate totals
            totals = selected_data[['Calories', 'Proteins', 'Carbohydrates', 'Fats']].sum()
            
            # Display summary metrics
            st.subheader("Total Nutrition")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Calories", f"{totals['Calories']} kcal")
            with col2:
                st.metric("Total Protein", f"{totals['Proteins']}g")
            with col3:
                st.metric("Total Carbs", f"{totals['Carbohydrates']}g")
            with col4:
                st.metric("Total Fat", f"{totals['Fats']}g")
            
            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["Macronutrient Distribution", "Food Comparison"])
            
            with tab1:
                # Macronutrient distribution pie chart
                st.subheader("Macronutrient Distribution")
                
                pie_options = {
                    "tooltip": {"trigger": "item"},
                    "legend": {"top": "5%", "left": "center"},
                    "series": [
                        {
                            "name": "Macronutrients",
                            "type": "pie",
                            "radius": ["40%", "70%"],
                            "avoidLabelOverlap": False,
                            "itemStyle": {
                                "borderRadius": 10,
                                "borderColor": "#fff",
                                "borderWidth": 2
                            },
                            "label": {"show": False, "position": "center"},
                            "emphasis": {
                                "label": {"show": True, "fontSize": "18", "fontWeight": "bold"}
                            },
                            "labelLine": {"show": False},
                            "data": [
                                {"value": totals['Proteins'], "name": "Protein"},
                                {"value": totals['Carbohydrates'], "name": "Carbs"},
                                {"value": totals['Fats'], "name": "Fats"}
                            ]
                        }
                    ]
                }
                st_echarts(options=pie_options, height="500px")
                
                # Calories from macronutrients
                st.subheader("Calories from Macronutrients")
                st.write("Protein and carbs provide 4 calories per gram, fat provides 9 calories per gram")
                
                calories_from = {
                    'Protein': totals['Proteins'] * 4,
                    'Carbs': totals['Carbohydrates'] * 4,
                    'Fat': totals['Fats'] * 9
                }
                
                bar_options = {
                    "xAxis": {
                        "type": "category",
                        "data": list(calories_from.keys()),
                    },
                    "yAxis": {"type": "value"},
                    "series": [{
                        "data": [
                            {"value": calories_from['Protein'], "itemStyle": {"color": "#4ECDC4"}},
                            {"value": calories_from['Carbs'], "itemStyle": {"color": "#45B7D1"}},
                            {"value": calories_from['Fat'], "itemStyle": {"color": "#FFA07A"}}
                        ],
                        "type": "bar"
                    }]
                }
                st_echarts(options=bar_options, height="400px")
            
            with tab2:
                st.subheader("Food Comparison")
                
                # Comparison bar chart
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                selected_data.set_index('Food_items')[['Proteins', 'Carbohydrates', 'Fats']].plot(
                    kind='bar', ax=ax2, color=['#4ECDC4', '#45B7D1', '#FFA07A']
                )
                ax2.set_ylabel("Grams")
                ax2.set_title("Nutritional Content per Food Item")
                st.pyplot(fig2)
                
                
elif page == "Healthier Alternatives":
    st.title("Healthier Alternatives")
    st.write("Find healthier alternatives for your favorite dishes")
    
    if df is not None:
        # User input
        st.subheader("Find Alternatives")
        user_dish = st.text_input("Enter a dish you'd like to find healthier alternatives for:")
        
        if user_dish:
            # Find exact matches in our dataset
            similar_dishes = df[df['RecipeName'].str.contains(user_dish, case=False, na=False)]
            
            if not similar_dishes.empty:
                st.subheader("Original Dish Options")
                st.write(f"Found {len(similar_dishes)} similar dishes in our database:")
                
                for _, row in similar_dishes.head(5).iterrows():  # Show up to 3 similar dishes
                    with st.expander(row['RecipeName']):
                        st.markdown(f"**Estimated Calories:** {estimate_calories(row['Ingredients'])}")
                        st.markdown("**Ingredients:**")
                        for ingredient in row['Ingredients'].split(','):
                            st.markdown(f"- {ingredient.strip()}")
            
            # Get healthier alternatives
            alternative = suggest_healthier_alternative(user_dish)
            
            if alternative != "No alternative found. Consider reducing portion size or cooking method.":
                st.subheader("Suggested Healthier Alternative")
                st.success(f"Try this instead: {alternative}")
                
                # Find recipes for the alternative in our dataset
                alt_recipes = df[df['RecipeName'].str.contains(alternative, case=False, na=False)]
                
                if not alt_recipes.empty:
                    st.write(f"Here are some recipes for '{alternative}':")
                    
                    for _, row in alt_recipes.head(3).iterrows():  # Show up to 2 alternative recipes
                        with st.expander(row['RecipeName']):
                            st.markdown(f"**Estimated Calories:** {estimate_calories(row['Ingredients'])}")
                            st.markdown("**Ingredients:**")
                            for ingredient in row['Ingredients'].split(','):
                                st.markdown(f"- {ingredient.strip()}")
                            st.markdown("**Instructions:**")
                            for instruction in row['Instructions'].split('.'):
                                if instruction.strip():
                                    st.markdown(f"- {instruction.strip()}")
                else:
                    st.info("We don't have specific recipes for this alternative in our database, but here are some general tips:")
                    st.markdown("- Reduce oil and butter amounts")
                    st.markdown("- Use whole grain alternatives")
                    st.markdown("- Increase vegetable content")
                    st.markdown("- Bake or grill instead of frying")
            else:
                st.warning(alternative)
                st.info("Here are some general tips for healthier eating:")
                st.markdown("- Choose cooking methods like baking, grilling, or steaming instead of frying")
                st.markdown("- Reduce portion sizes of high-calorie foods")
                st.markdown("- Increase intake of vegetables and fruits")
                st.markdown("- Choose whole grain options when available")
                st.markdown("- Limit added sugars and processed foods")
                
                # Try to find similar but healthier recipes
                main_ingredient = user_dish.split()[0].lower()
                if main_ingredient in INGREDIENT_CALORIES:
                    healthier_recipes = df[
                        (df['Ingredients'].str.contains(main_ingredient, case=False)) &
                        (~df['RecipeName'].str.contains('fried|deep fry|cream|butter', case=False))
                    ]
                    
                    if not healthier_recipes.empty:
                        st.subheader(f"Healthier {main_ingredient.capitalize()} Recipes")
                        st.write(f"Here are some healthier recipes using {main_ingredient}:")
                        
                        for _, row in healthier_recipes.head(2).iterrows():
                            with st.expander(row['RecipeName']):
                                st.markdown(f"**Estimated Calories:** {estimate_calories(row['Ingredients'])}")
                                st.markdown("**Ingredients:**")
                                for ingredient in row['Ingredients'].split(','):
                                    st.markdown(f"- {ingredient.strip()}")