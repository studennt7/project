import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta
from fpdf import FPDF
import tempfile
import os

# Функция для загрузки и анализа данных
@st.cache_data
def load_and_analyze_data(file):
    try:
        df = pd.read_excel(file)

        required_columns = ['Дата', 'Объем продаж', 'Вид продукта', 'Местоположение', 'Сумма', 'Тип покупателя']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return f"В файле отсутствуют колонки: {', '.join(missing_columns)}", None

        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Выручка'] = df['Объем продаж'] * df['Сумма']

        return None, df

    except Exception as e:
        return f"Ошибка обработки файла: {str(e)}", None

# Функция для прогнозирования

def make_forecast(df, periods=30):
    try:
        daily_data = df.groupby('Дата').agg({'Объем продаж': 'sum', 'Выручка': 'sum'}).reset_index()
        daily_data = daily_data.set_index('Дата').asfreq('D').fillna(0)

        X = np.arange(len(daily_data)).reshape(-1, 1)
        y = daily_data['Объем продаж'].values
        model = LinearRegression()
        model.fit(X, y)

        future_dates = pd.date_range(start=daily_data.index[-1] + timedelta(days=1), periods=periods)
        X_future = np.arange(len(daily_data), len(daily_data) + periods).reshape(-1, 1)
        y_pred = model.predict(X_future)

        forecast_df = pd.DataFrame({
            'Дата': future_dates,
            'Объем продаж': y_pred,
            'Тип': 'Прогноз'
        })

        actual_df = pd.DataFrame({
            'Дата': daily_data.index,
            'Объем продаж': daily_data['Объем продаж'],
            'Тип': 'Факт'
        })

        combined_df = pd.concat([actual_df, forecast_df])

        return combined_df, None

    except Exception as e:
        return None, f"Ошибка прогнозирования: {str(e)}"

# Функция для генерации рекомендаций

def generate_recommendations(df):
    recommendations = []

    try:
        daily_sales = df.groupby('Дата')['Объем продаж'].sum()
        decomposition = seasonal_decompose(daily_sales, period=7)

        if decomposition.seasonal.std() > (daily_sales.mean() * 0.1):
            recommendations.append(
                "🔍 Обнаружена заметная недельная сезонность в продажах. Рекомендуется планировать запасы и персонал с учетом этих колебаний."
            )
    except:
        pass

    top_products = df.groupby('Вид продукта')['Объем продаж'].sum().nlargest(3)
    if len(top_products) > 0:
        recommendations.append(
            f"🏆 Топ-3 продукта по объему продаж: {', '.join(top_products.index)}. Рекомендуется увеличить их наличие и продвижение."
        )

    customer_stats = df.groupby('Тип покупателя')['Выручка'].agg(['sum', 'count'])
    if len(customer_stats) > 1:
        best_customer = customer_stats['sum'].idxmax()
        recommendations.append(
            f"👥 Наибольшую выручку приносят покупатели типа '{best_customer}'. Рекомендуется разработать для них специальную программу лояльности."
        )

    location_stats = df.groupby('Местоположение')['Выручка'].sum()
    if len(location_stats) > 1:
        best_location = location_stats.idxmax()
        worst_location = location_stats.idxmin()
        recommendations.append(
            f"📍 Локация '{best_location}' показывает наилучшие результаты, а '{worst_location}' - наихудшие. Рекомендуется изучить причины различий."
        )

    return recommendations if recommendations else ["🔎 Недостаточно данных для формирования рекомендаций"]

# Функция для генерации PDF отчета

def generate_pdf_report(df, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", '', "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.cell(200, 10, txt="Отчет по продажам", ln=True, align='C')

    pdf.ln(10)
    pdf.set_font("DejaVu", size=10)
    for rec in recommendations:
        pdf.multi_cell(0, 8, txt=rec)

    pdf.ln(5)
    pdf.set_font("DejaVu", size=9)
    for index, row in df.head(30).iterrows():  # ограничим до 30 строк
        text = f"{row['Дата'].strftime('%Y-%m-%d')} | {row['Вид продукта']} | {row['Объем продаж']} | {row['Сумма']} руб. | {row['Тип покупателя']}"
        pdf.multi_cell(0, 7, txt=text)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp.seek(0)
        return tmp.read(), tmp.name

# Интерфейс приложения
st.set_page_config(page_title="Sales Smart Analytics", layout="wide")
st.title("📊 Sales Smart Analytics")

# Загрузка данных
with st.expander("📁 Загрузите данные", expanded=True):
    st.markdown("""
    Загрузите Excel файл с данными о продажах. 
    Необходимые колонки:
    - Дата
    - Объем продаж 
    - Вид продукта
    - Местоположение
    - Сумма
    - Тип покупателя
    """)

    uploaded_file = st.file_uploader("Выберите Excel-файл", type="xlsx", label_visibility="collapsed")

if uploaded_file is not None:
    error_msg, df = load_and_analyze_data(uploaded_file)

    if error_msg:
        st.error(error_msg)
    elif df is not None:
        ...  # оставшаяся часть кода анализа

        st.download_button(label="Скачать обработанные данные", data=filtered_df.to_csv(index=False).encode('utf-8'), file_name='sales_analysis.csv', mime='text/csv')

        pdf_data, pdf_path = generate_pdf_report(filtered_df, recommendations)
        st.download_button("📄 Скачать PDF отчет", data=pdf_data, file_name="sales_report.pdf", mime="application/pdf")

