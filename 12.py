import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from fpdf import FPDF
import base64
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка страницы
st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide", page_icon="📊")

# Автоматическая установка зависимостей
def install_packages():
    packages = ['statsmodels', 'fpdf2', 'plotly', 'scikit-learn', 'Pillow']
    import subprocess
    import sys
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:
    st.error("Не удалось загрузить необходимые библиотеки. Пожалуйста, установите statsmodels.")
    st.stop()

# Функции анализа данных
@st.cache_data
def load_and_analyze_data(file):
    try:
        df = pd.read_excel(file)
        
        required_columns = ['Дата', 'Объем продаж', 'Вид продукта', 'Местоположение', 'Сумма', 'Тип покупателя']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return f"Отсутствуют колонки: {', '.join(missing_columns)}", None
        
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Выручка'] = df['Объем продаж'] * df['Сумма']
        
        return None, df
        
    except Exception as e:
        return f"Ошибка загрузки: {str(e)}", None

def make_forecast(df, periods=7):
    try:
        daily_data = df.groupby('Дата').agg({'Объем продаж':'sum', 'Выручка':'sum'}).reset_index()
        daily_data = daily_data.set_index('Дата').asfreq('D').fillna(0)
        
        X = np.arange(len(daily_data)).reshape(-1, 1)
        y = daily_data['Объем продаж'].values
        model = LinearRegression()
        model.fit(X, y)
        
        future_dates = pd.date_range(
            start=daily_data.index[-1] + timedelta(days=1),
            periods=periods
        )
        X_future = np.arange(len(daily_data), len(daily_data)+periods).reshape(-1, 1)
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
        
        return pd.concat([actual_df, forecast_df]), None
        
    except Exception as e:
        return None, f"Ошибка прогноза: {str(e)}"

def generate_recommendations(df):
    recommendations = []
    
    try:
        daily_sales = df.groupby('Дата')['Объем продаж'].sum()
        decomposition = seasonal_decompose(daily_sales, period=7)
        
        if decomposition.seasonal.std() > (daily_sales.mean() * 0.1):
            recommendations.append(
                "🔍 Обнаружена недельная сезонность. Планируйте запасы и персонал соответственно."
            )
    except:
        pass
    
    top_products = df.groupby('Вид продукта')['Объем продаж'].sum().nlargest(3)
    if len(top_products) > 0:
        recommendations.append(
            f"🏆 Топ-3 продукта: {', '.join(top_products.index)}. Увеличьте их наличие."
        )
    
    customer_stats = df.groupby('Тип покупателя')['Выручка'].agg(['sum', 'count'])
    if len(customer_stats) > 1:
        best_customer = customer_stats['sum'].idxmax()
        recommendations.append(
            f"👥 Основная выручка от '{best_customer}'. Разработайте программу лояльности."
        )
    
    location_stats = df.groupby('Местоположение')['Выручка'].sum()
    if len(location_stats) > 1:
        best_loc = location_stats.idxmax()
        worst_loc = location_stats.idxmin()
        recommendations.append(
            f"📍 Лучшая локация: {best_loc}, худшая: {worst_loc}. Изучите причины."
        )
    
    return recommendations if recommendations else ["🔎 Недостаточно данных для рекомендаций"]

# Функция создания PDF
def create_pdf_report(df, kpis, figs, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Заголовок
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Отчет по анализу продаж", ln=1, align='C')
    pdf.ln(10)
    
    # Основные метрики
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Ключевые показатели", ln=1)
    pdf.set_font("Arial", size=12)
    for kpi in kpis.split('\n'):
        pdf.cell(200, 10, txt=kpi, ln=1)
    pdf.ln(10)
    
    # Графики
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Визуализация данных", ln=1)
    
    for fig in figs:
        img_path = tempfile.mktemp(suffix='.png')
        fig.write_image(img_path)
        pdf.image(img_path, w=190)
        pdf.ln(5)
        os.unlink(img_path)
    
    # Рекомендации
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Рекомендации", ln=1)
    pdf.set_font("Arial", size=12)
    for rec in recommendations:
        pdf.multi_cell(190, 10, txt=rec)
        pdf.ln(2)
    
    # Сохраняем PDF
    pdf_path = tempfile.mktemp(suffix='.pdf')
    pdf.output(pdf_path)
    return pdf_path

# Интерфейс приложения
st.title("📈 Sales Analytics Dashboard")

with st.expander("📁 Загрузите данные", expanded=True):
    uploaded_file = st.file_uploader(
        "Выберите Excel-файл с данными продаж", 
        type="xlsx",
        help="Файл должен содержать колонки: Дата, Объем продаж, Вид продукта, Местоположение, Сумма, Тип покупателя"
    )

if uploaded_file is not None:
    error_msg, df = load_and_analyze_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        # Расчет KPI
        total_sales = df['Объем продаж'].sum()
        total_revenue = df['Выручка'].sum()
        avg_price = df['Сумма'].mean()
        unique_products = df['Вид продукта'].nunique()
        
        kpi_text = f"Общий объем продаж: {total_sales:,.0f}\n" \
                  f"Общая выручка: {total_revenue:,.2f} руб.\n" \
                  f"Средняя цена: {avg_price:.2f} руб.\n" \
                  f"Количество продуктов: {unique_products}"
        
        # Фильтры
        st.sidebar.header("Фильтры")
        min_date = df['Дата'].min().date()
        max_date = df['Дата'].max().date()
        date_range = st.sidebar.date_input(
            "Диапазон дат",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        products = st.sidebar.multiselect(
            "Выберите продукты",
            options=df['Вид продукта'].unique(),
            default=df['Вид продукта'].unique()
        )
        
        # Применение фильтров
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[
                (df['Дата'].dt.date >= start_date) & 
                (df['Дата'].dt.date <= end_date) &
                (df['Вид продукта'].isin(products))
            ]
        else:
            filtered_df = df[df['Вид продукта'].isin(products)]
        
        # Вкладки
        tab1, tab2, tab3, tab4 = st.tabs(["Обзор", "Продукты", "Локации", "Отчет"])
        
        with tab1:
            st.subheader("Ключевые показатели")
            st.text(kpi_text)
            
            fig1 = px.line(
                filtered_df.groupby('Дата')['Объем продаж'].sum().reset_index(),
                x='Дата',
                y='Объем продаж',
                title='Динамика продаж'
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            forecast_df, _ = make_forecast(filtered_df)
            if forecast_df is not None:
                fig2 = px.line(
                    forecast_df,
                    x='Дата',
                    y='Объем продаж',
                    color='Тип',
                    title='Прогноз продаж на 7 дней'
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            fig3 = px.bar(
                filtered_df.groupby('Вид продукта')['Объем продаж'].sum().reset_index(),
                x='Вид продукта',
                y='Объем продаж',
                title='Продажи по продуктам'
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            fig4 = px.scatter(
                filtered_df.groupby('Вид продукта').agg(
                    avg_price=('Сумма', 'mean'),
                    total_sales=('Объем продаж', 'sum')
                ).reset_index(),
                x='avg_price',
                y='total_sales',
                text='Вид продукта',
                title='Зависимость объема от цены'
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        with tab3:
            fig5 = px.pie(
                filtered_df.groupby('Местоположение')['Выручка'].sum().reset_index(),
                names='Местоположение',
                values='Выручка',
                title='Распределение выручки'
            )
            st.plotly_chart(fig5, use_container_width=True)
            
            fig6 = px.line(
                filtered_df.groupby(['Местоположение', 'Дата'])['Объем продаж'].sum().reset_index(),
                x='Дата',
                y='Объем продаж',
                color='Местоположение',
                title='Динамика по локациям'
            )
            st.plotly_chart(fig6, use_container_width=True)
        
        with tab4:
            st.subheader("Генерация отчета")
            
            recommendations = generate_recommendations(filtered_df)
            st.info("#### Рекомендации")
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Подготовка данных для PDF
            figs_for_pdf = [fig1, fig3, fig5]
            if forecast_df is not None:
                figs_for_pdf.append(fig2)
            
            # Создание и скачивание PDF
            pdf_path = create_pdf_report(
                filtered_df, 
                kpi_text, 
                figs_for_pdf, 
                recommendations
            )
            
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            st.download_button(
                label="Скачать PDF отчет",
                data=pdf_bytes,
                file_name="sales_analytics_report.pdf",
                mime="application/pdf"
            )
            
            os.unlink(pdf_path)
