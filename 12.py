import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta

# Функция для загрузки и анализа данных
@st.cache_data
def load_and_analyze_data(file):
    try:
        df = pd.read_excel(file)
        
        # Проверка на обязательные колонки
        required_columns = ['Дата', 'Объем продаж', 'Вид продукта', 'Местоположение', 'Сумма', 'Тип покупателя']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return f"В файле отсутствуют колонки: {', '.join(missing_columns)}", None
        
        # Обработка данных
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Выручка'] = df['Объем продаж'] * df['Сумма']
        
        return None, df
        
    except Exception as e:
        return f"Ошибка обработки файла: {str(e)}", None

# Функция для прогнозирования
def make_forecast(df, periods=7):
    try:
        # Подготовка данных для временного ряда
        daily_data = df.groupby('Дата').agg({'Объем продаж':'sum', 'Выручка':'sum'}).reset_index()
        daily_data = daily_data.set_index('Дата').asfreq('D').fillna(0)
        
        # Линейная регрессия для прогноза
        X = np.arange(len(daily_data)).reshape(-1, 1)
        y = daily_data['Объем продаж'].values
        model = LinearRegression()
        model.fit(X, y)
        
        # Прогноз на будущие периоды
        future_dates = pd.date_range(
            start=daily_data.index[-1] + timedelta(days=1),
            periods=periods
        )
        X_future = np.arange(len(daily_data), len(daily_data)+periods).reshape(-1, 1)
        y_pred = model.predict(X_future)
        
        # Создаем DataFrame с прогнозом
        forecast_df = pd.DataFrame({
            'Дата': future_dates,
            'Объем продаж': y_pred,
            'Тип': 'Прогноз'
        })
        
        # Фактические данные
        actual_df = pd.DataFrame({
            'Дата': daily_data.index,
            'Объем продаж': daily_data['Объем продаж'],
            'Тип': 'Факт'
        })
        
        # Объединяем факт и прогноз
        combined_df = pd.concat([actual_df, forecast_df])
        
        return combined_df, None
        
    except Exception as e:
        return None, f"Ошибка прогнозирования: {str(e)}"

# Функция для генерации рекомендаций
def generate_recommendations(df):
    recommendations = []
    
    # Анализ сезонности
    try:
        daily_sales = df.groupby('Дата')['Объем продаж'].sum()
        decomposition = seasonal_decompose(daily_sales, period=7)
        
        if decomposition.seasonal.std() > (daily_sales.mean() * 0.1):
            recommendations.append(
                "🔍 Обнаружена заметная недельная сезонность в продажах. "
                "Рекомендуется планировать запасы и персонал с учетом этих колебаний."
            )
    except:
        pass
    
    # Анализ топовых продуктов
    top_products = df.groupby('Вид продукта')['Объем продаж'].sum().nlargest(3)
    if len(top_products) > 0:
        recommendations.append(
            f"🏆 Топ-3 продукта по объему продаж: {', '.join(top_products.index)}. "
            "Рекомендуется увеличить их наличие и продвижение."
        )
    
    # Анализ покупателей
    customer_stats = df.groupby('Тип покупателя')['Выручка'].agg(['sum', 'count'])
    if len(customer_stats) > 1:
        best_customer = customer_stats['sum'].idxmax()
        recommendations.append(
            f"👥 Наибольшую выручку приносят покупатели типа '{best_customer}'. "
            "Рекомендуется разработать для них специальную программу лояльности."
        )
    
    # Анализ локаций
    location_stats = df.groupby('Местоположение')['Выручка'].sum()
    if len(location_stats) > 1:
        best_location = location_stats.idxmax()
        worst_location = location_stats.idxmin()
        recommendations.append(
            f"📍 Локация '{best_location}' показывает наилучшие результаты, "
            f"а '{worst_location}' - наихудшие. Рекомендуется изучить причины различий."
        )
    
    return recommendations if recommendations else ["🔎 Недостаточно данных для формирования рекомендаций"]

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

# Основной анализ
if uploaded_file is not None:
    error_msg, df = load_and_analyze_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    elif df is not None:
        # Расчет KPI
        total_sales = df['Объем продаж'].sum()
        total_revenue = df['Выручка'].sum()
        avg_price = df['Сумма'].mean()
        unique_products = df['Вид продукта'].nunique()
        
        # Отображение KPI
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Общий объем продаж", f"{total_sales:,.0f}")
        col2.metric("Общая выручка", f"{total_revenue:,.2f} руб.")
        col3.metric("Средняя сумма", f"{avg_price:,.2f} руб.")
        col4.metric("Видов продуктов", unique_products)
        
        # Фильтры
        st.subheader("Фильтры для анализа")
        min_date = df['Дата'].min().date()
        max_date = df['Дата'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            selected_dates = st.date_input(
                "Диапазон дат",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        with col2:
            selected_products = st.multiselect(
                "Выберите продукты",
                options=df['Вид продукта'].unique(),
                default=df['Вид продукта'].unique()
            )
        
        # Применение фильтров
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
            filtered_df = df[
                (df['Дата'].dt.date >= start_date) & 
                (df['Дата'].dt.date <= end_date) &
                (df['Вид продукта'].isin(selected_products))
            ]
        else:
            filtered_df = df[df['Вид продукта'].isin(selected_products)]
        
        # Визуализации
        tab1, tab2, tab3, tab4 = st.tabs(["Динамика продаж", "Анализ продуктов", "Анализ локаций", "Прогноз и рекомендации"])
        
        with tab1:
            fig = px.line(
                filtered_df.groupby('Дата')['Объем продаж'].sum().reset_index(),
                x='Дата',
                y='Объем продаж',
                title='Динамика продаж',
                labels={'Объем продаж': 'Объем продаж', 'Дата': 'Дата'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                product_sales = filtered_df.groupby('Вид продукта')['Объем продаж'].sum().reset_index()
                fig = px.bar(
                    product_sales,
                    x='Вид продукта',
                    y='Объем продаж',
                    title='Продажи по продуктам'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                price_analysis = filtered_df.groupby('Вид продукта').agg(
                    avg_price=('Сумма', 'mean'),
                    total_sales=('Объем продаж', 'sum')
                ).reset_index()
                
                fig = px.scatter(
                    price_analysis,
                    x='avg_price',
                    y='total_sales',
                    text='Вид продукта',
                    title='Зависимость объема от цены',
                    labels={'avg_price': 'Средняя цена', 'total_sales': 'Объем продаж'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                location_sales = filtered_df.groupby('Местоположение')['Выручка'].sum().reset_index()
                fig = px.pie(
                    location_sales,
                    names='Местоположение',
                    values='Выручка',
                    title='Распределение выручки по локациям'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                location_trend = filtered_df.groupby(['Местоположение', 'Дата'])['Объем продаж'].sum().reset_index()
                fig = px.line(
                    location_trend,
                    x='Дата',
                    y='Объем продаж',
                    color='Местоположение',
                    title='Динамика по локациям'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Прогноз продаж на 7 дней")
            forecast_df, forecast_error = make_forecast(filtered_df)
            
            if forecast_error:
                st.error(forecast_error)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(
                        forecast_df,
                        x='Дата',
                        y='Объем продаж',
                        color='Тип',
                        title='Факт и прогноз продаж',
                        line_dash='Тип',
                        color_discrete_map={'Факт':'blue', 'Прогноз':'red'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(
                        forecast_df[forecast_df['Тип'] == 'Прогноз'][['Дата', 'Объем продаж']]
                        .rename(columns={'Объем продаж': 'Прогноз объема'})
                        .style.format({'Прогноз объема': '{:.1f}'}),
                        hide_index=True
                    )
            
            st.subheader("Рекомендации")
            recommendations = generate_recommendations(filtered_df)
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        # Экспорт данных
        st.download_button(
            label="Скачать обработанные данные",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='sales_analysis.csv',
            mime='text/csv'
        )


