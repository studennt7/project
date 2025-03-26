import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Функция для загрузки и анализа данных
def load_and_analyze_data(file):
    try:
        df = pd.read_excel(file)
        
        # Проверка на обязательные колонки
        required_columns = ['Дата', 'Объем продаж', 'Вид продукта', 'Местоположение', 'Цена', 'Тип покупателя']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return f"В файле отсутствуют колонки: {', '.join(missing_columns)}", None
        
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Выручка'] = df['Объем продаж'] * df['Цена']
        
        total_sales = df['Объем продаж'].sum()
        total_revenue = df['Выручка'].sum()
        avg_price = df['Цена'].mean()
        
        info = f"Общий объем продаж: {total_sales:,.0f}\n" \
               f"Общая выручка: {total_revenue:,.2f} руб.\n" \
               f"Средняя цена: {avg_price:.2f} руб."
        
        return info, df
        
    except Exception as e:
        return f"Ошибка обработки файла: {str(e)}", None


# Интерфейс приложения
st.title("Простой анализ продаж")

st.markdown("""
Загрузите Excel файл с данными о продажах. 
Необходимые колонки:
- Дата
- Объем продаж 
- Вид продукта
- Местоположение
- Цена
- Тип покупателя
""")

# Форма загрузки файла
uploaded_file = st.file_uploader("Загрузите Excel-файл", type="xlsx")

if uploaded_file is not None:
    info, df = load_and_analyze_data(uploaded_file)
    
    # Выводим информацию о продажах
    st.text(info)
    
    if df is not None:
        # Построение графика динамики продаж
        plt.figure(figsize=(10, 6))
        plt.plot(df['Дата'], df['Объем продаж'], label="Объем продаж")
        plt.xlabel('Дата')
        plt.ylabel('Объем продаж')
        plt.title('Динамика продаж')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

