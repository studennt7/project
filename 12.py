import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройки страницы
st.set_page_config(
    page_title="Sales-smart",
    page_icon="📊",
    layout="wide"
)

# Настройка темы
def set_white_theme():
    white_theme = {
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#31333F",
        "font": "sans serif"
    }
    
    # Применяем настройки темы
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {white_theme["backgroundColor"]};
                color: {white_theme["textColor"]};
            }}
            .css-1d391kg, .css-1v3fvcr {{
                background-color: {white_theme["secondaryBackgroundColor"]} !important;
            }}
            .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as {{
                color: {white_theme["textColor"]} !important;
            }}
            .css-1aumxhk {{
                color: {white_theme["textColor"]};
                font-family: {white_theme["font"]};
            }}
            .css-10trblm {{
                color: {white_theme["textColor"]};
            }}
            .st-cn, .st-co, .st-cp {{
                color: {white_theme["textColor"]} !important;
            }}
            .st-bh, .st-bi, .st-bj, .st-bk, .st-bl, .st-bm, .st-bn, .st-bo, .st-bp, .st-bq, .st-br, .st-bs, .st-bt, .st-bu, .st-bv, .st-bw, .st-bx, .st-by, .st-bz {{
                color: {white_theme["textColor"]} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_white_theme()

# Функция загрузки данных
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

# Улучшенная функция прогнозирования
def make_forecast(df, periods=30):
    try:
        daily_data = df.groupby('Дата').agg({'Объем продаж': 'sum'}).reset_index()
        daily_data = daily_data.set_index('Дата').asfreq('D').fillna(0)
        
        if len(daily_data) < 30:
            return None, "Для прогноза требуется минимум 30 дней данных"
        
        # Модель с учетом сезонности и тренда
        model = ExponentialSmoothing(
            daily_data['Объем продаж'],
            seasonal_periods=7,
            trend='add',
            seasonal='add',
            damped_trend=True
        ).fit()
        
        forecast = model.forecast(periods)
        future_dates = pd.date_range(
            start=daily_data.index[-1] + timedelta(days=1),
            periods=periods
        )
        
        forecast_df = pd.DataFrame({
            'Дата': future_dates,
            'Объем продаж': forecast,
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

# Генерация рекомендаций
def generate_recommendations(df):
    recommendations = []
    
    try:
        daily_sales = df.groupby('Дата')['Объем продаж'].sum()
        decomposition = seasonal_decompose(daily_sales, period=7)
        
        if decomposition.seasonal.std() > (daily_sales.mean() * 0.1):
            recommendations.append(
                "🔍 Выявлена недельная сезонность. Оптимизируйте запасы и персонал соответственно."
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
            f"📍 Лучшая локация: {best_loc}, проблемная: {worst_loc}. Изучите причины."
        )
    
    return recommendations if recommendations else ["🔎 Недостаточно данных для рекомендаций"]

# Интерфейс приложения
st.title("📈 Sales-smart")

# Загрузка данных
with st.expander("📁 Загрузить данные", expanded=True):
    uploaded_file = st.file_uploader(
        "Выберите файл продаж (Excel)",
        type="xlsx",
        help="Файл должен содержать колонки: Дата, Объем продаж, Вид продукта, Местоположение, Сумма, Тип покупателя"
    )

if uploaded_file:
    error_msg, df = load_and_analyze_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        # Основные метрики
        total_sales = df['Объем продаж'].sum()
        total_revenue = df['Выручка'].sum()
        avg_price = df['Сумма'].mean()
        unique_products = df['Вид продукта'].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Общий объем", f"{total_sales:,.0f}")
        col2.metric("Выручка", f"{total_revenue:,.2f} руб.")
        col3.metric("Средний чек", f"{avg_price:.2f} руб.")
        col4.metric("Кол-во продуктов", unique_products)
        
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
            "Продукты",
            options=df['Вид продукта'].unique(),
            default=df['Вид продукта'].unique()
        )
        
        locations = st.sidebar.multiselect(
            "Локации",
            options=df['Местоположение'].unique(),
            default=df['Местоположение'].unique()
        )
        
        # Применение фильтров
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[
                (df['Дата'].dt.date >= start_date) & 
                (df['Дата'].dt.date <= end_date) &
                (df['Вид продукта'].isin(products)) &
                (df['Местоположение'].isin(locations))
            ]
        else:
            filtered_df = df[
                (df['Вид продукта'].isin(products)) &
                (df['Местоположение'].isin(locations))
            ]
        
        # Визуализации
        tab1, tab2, tab3, tab4 = st.tabs(["Динамика", "Продукты", "Локации", "Прогноз"])
        
        with tab1:
            fig = px.line(
                filtered_df.groupby('Дата').agg({'Объем продаж': 'sum'}).reset_index(),
                x='Дата',
                y='Объем продаж',
                title='Динамика продаж',
                labels={'Объем продаж': 'Объем', 'Дата': 'Дата'},
                color_discrete_sequence=['#1f77b4']  # Синий цвет для графиков
            )
            fig.update_xaxes(tickformat="%d %b", dtick="M1")
            fig.update_layout(
                hovermode="x unified",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                filtered_df.groupby('Дата').agg({'Объем продаж': 'sum', 'Выручка': 'sum'})
                .style.format({'Объем продаж': '{:,.0f}', 'Выручка': '₽{:,.2f}'}),
                use_container_width=True
            )
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    filtered_df.groupby('Вид продукта')['Объем продаж'].sum().reset_index(),
                    x='Вид продукта',
                    y='Объем продаж',
                    title='Продажи по продуктам',
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    filtered_df.groupby('Вид продукта').agg({
                        'Сумма': 'mean',
                        'Объем продаж': 'sum'
                    }).reset_index(),
                    x='Сумма',
                    y='Объем продаж',
                    size='Объем продаж',
                    color='Вид продукта',
                    title='Объем продаж'
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    filtered_df.groupby('Местоположение')['Выручка'].sum().reset_index(),
                    names='Местоположение',
                    values='Выручка',
                    title='Распределение выручки'
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(
                    filtered_df.groupby(['Местоположение', 'Дата'])['Объем продаж'].sum().reset_index(),
                    x='Дата',
                    y='Объем продаж',
                    color='Местоположение',
                    title='Динамика по локациям'
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Прогноз продаж на 30 дней")
            forecast_df, forecast_error = make_forecast(filtered_df)
            
            if forecast_error:
                st.warning(forecast_error)
            else:
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = go.Figure()
                    
                    # Фактические данные
                    fig.add_trace(go.Scatter(
                        x=forecast_df[forecast_df['Тип'] == 'Факт']['Дата'],
                        y=forecast_df[forecast_df['Тип'] == 'Факт']['Объем продаж'],
                        name='Факт',
                        line=dict(color='#1f77b4')  # Синий
                    ))
                    
                    # Прогноз
                    fig.add_trace(go.Scatter(
                        x=forecast_df[forecast_df['Тип'] == 'Прогноз']['Дата'],
                        y=forecast_df[forecast_df['Тип'] == 'Прогноз']['Объем продаж'],
                        name='Прогноз',
                        line=dict(color='#d62728', dash='dot')  # Красный
                    ))
                    
                    # Доверительный интервал
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
                        fillcolor='rgba(214,39,40,0.1)',
                        name='Доверительный интервал'
                    ))
                    
                    fig.update_layout(
                        title='Прогноз продаж с учетом сезонности',
                        xaxis_title='Дата',
                        yaxis_title='Объем продаж',
                        hovermode='x unified',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='black'),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Детали прогноза**")
                    st.dataframe(
                        forecast_df[forecast_df['Тип'] == 'Прогноз'][['Дата', 'Объем продаж']]
                        .rename(columns={'Объем продаж': 'Прогноз'})
                        .style.format({'Прогноз': '{:,.0f}'}),
                        hide_index=True
                    )
            
            st.subheader("Рекомендации")
            recommendations = generate_recommendations(filtered_df)
            for rec in recommendations:
                st.markdown(f"📌 {rec}")
        
        # Экспорт данных
        st.download_button(
            label="Скачать данные (CSV)",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name="sales_data.csv",
            mime="text/csv"
        )
