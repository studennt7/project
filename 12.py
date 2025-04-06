import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

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
        # Агрегируем данные по дням
        daily_data = df.groupby('Дата').agg({'Объем продаж': 'sum'}).reset_index()
        daily_data = daily_data.set_index('Дата').asfreq('D').fillna(0)
        
        # Проверяем достаточность данных для анализа сезонности
        if len(daily_data) < 30:
            return None, "Недостаточно данных для прогноза (требуется минимум 30 дней)"
        
        # Анализ сезонности
        decomposition = seasonal_decompose(daily_data['Объем продаж'], period=7)
        
        # Используем тройное экспоненциальное сглаживание (Holt-Winters)
        model = ExponentialSmoothing(
            daily_data['Объем продаж'],
            seasonal_periods=7,
            trend='add',
            seasonal='add',
            damped_trend=True
        ).fit()
        
        # Прогнозируем
        forecast = model.forecast(periods)
        future_dates = pd.date_range(
            start=daily_data.index[-1] + timedelta(days=1),
            periods=periods
        )
        
        # Создаем DataFrame с прогнозом
        forecast_df = pd.DataFrame({
            'Дата': future_dates,
            'Объем продаж': forecast,
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
        total_sales = df['Объем продаж'].sum()
        total_revenue = df['Выручка'].sum()
        avg_price = df['Сумма'].mean()
        unique_products = df['Вид продукта'].nunique()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Общий объем продаж", f"{total_sales:,.0f}")
        col2.metric("Общая выручка", f"{total_revenue:,.2f} руб.")
        col3.metric("Средняя сумма", f"{avg_price:,.2f} руб.")
        col4.metric("Видов продуктов", unique_products)

        st.subheader("Фильтры для анализа")
        min_date = df['Дата'].min().date()
        max_date = df['Дата'].max().date()

        col1, col2 = st.columns(2)
        with col1:
            selected_dates = st.date_input("Диапазон дат", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        with col2:
            selected_products = st.multiselect("Выберите продукты", options=df['Вид продукта'].unique(), default=df['Вид продукта'].unique())

        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
            filtered_df = df[(df['Дата'].dt.date >= start_date) & (df['Дата'].dt.date <= end_date) & (df['Вид продукта'].isin(selected_products))]
        else:
            filtered_df = df[df['Вид продукта'].isin(selected_products)]

        tab1, tab2, tab3, tab4 = st.tabs(["Динамика продаж", "Анализ продуктов", "Анализ локаций", "Прогноз и рекомендации"])

        with tab1:
            fig = px.line(
                filtered_df.groupby('Дата')['Объем продаж'].sum().reset_index(),
                x='Дата',
                y='Объем продаж',
                title='Динамика продаж',
                labels={'Объем продаж': 'Объем продаж', 'Дата': 'Дата'}
            )
            fig.update_xaxes(tickformat="%d %b", dtick="M15")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                product_sales = filtered_df.groupby('Вид продукта')['Объем продаж'].sum().reset_index()
                fig = px.bar(product_sales, x='Вид продукта', y='Объем продаж', title='Продажи по продуктам')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                price_analysis = filtered_df.groupby('Вид продукта').agg(avg_price=('Сумма', 'mean'), total_sales=('Объем продаж', 'sum')).reset_index()
                fig = px.scatter(price_analysis, x='avg_price', y='total_sales', text='Вид продукта', title='Зависимость объема от цены', labels={'avg_price': 'Средняя цена', 'total_sales': 'Объем продаж'})
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                location_sales = filtered_df.groupby('Местоположение')['Выручка'].sum().reset_index()
                fig = px.pie(location_sales, names='Местоположение', values='Выручка', title='Распределение выручки по локациям')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                location_trend = filtered_df.groupby(['Местоположение', 'Дата'])['Объем продаж'].sum().reset_index()
                fig = px.line(location_trend, x='Дата', y='Объем продаж', color='Местоположение', title='Динамика продаж по локациям', labels={'Дата': 'Дата', 'Объем продаж': 'Объем продаж', 'Местоположение': 'Локация'})
                fig.update_xaxes(tickformat="%d %b", dtick="M15", title="Дата")
                fig.update_layout(hovermode="x unified", legend_title_text='Локация', margin=dict(t=40, b=40, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
    st.subheader("Прогноз продаж на 30 дней")
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
                title='Фактические и прогнозируемые продажи',
                line_dash='Тип',
                color_discrete_map={'Факт': 'blue', 'Прогноз': 'red'}
            )
            
            # Добавляем доверительный интервал (примерный)
            last_actual = forecast_df[forecast_df['Тип'] == 'Факт'].iloc[-1]['Объем продаж']
            fig.add_trace(go.Scatter(
                x=forecast_df[forecast_df['Тип'] == 'Прогноз']['Дата'],
                y=forecast_df[forecast_df['Тип'] == 'Прогноз']['Объем продаж'] * 1.2,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df[forecast_df['Тип'] == 'Прогноз']['Дата'],
                y=forecast_df[forecast_df['Тип'] == 'Прогноз']['Объем продаж'] * 0.8,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255,0,0,0.2)',
                name='Доверительный интервал'
            ))
            
            fig.update_layout(
                xaxis_title='Дата',
                yaxis_title='Объем продаж',
                hovermode='x unified',
                legend_title_text='Тип данных'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Детали прогноза**")
            st.dataframe(
                forecast_df[forecast_df['Тип'] == 'Прогноз'][['Дата', 'Объем продаж']]
                .rename(columns={'Объем продаж': 'Прогноз объема'})
                .style.format({'Прогноз объема': '{:.1f}'}),
                hide_index=True
            )
            
            # Анализ точности прогноза
            st.markdown("**Метод прогнозирования**")
            st.write("""
            Использован метод Хольта-Винтерса с:
            - Учетом тренда
            - Учетом недельной сезонности
            - Демпфированием тренда для более консервативных прогнозов
            """)

            st.subheader("Рекомендации")
            recommendations = generate_recommendations(filtered_df)
            for rec in recommendations:
                st.markdown(f"- {rec}")

        st.download_button(label="Скачать обработанные данные", data=filtered_df.to_csv(index=False).encode('utf-8'), file_name='sales_analysis.csv', mime='text/csv')

