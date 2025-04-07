import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta
import warnings
from fpdf import FPDF
from io import BytesIO
import base64
warnings.filterwarnings('ignore')

# Настройки страницы
st.set_page_config(
    page_title="Sales-smart",
    page_icon="📊",
    layout="wide"
)

# Настройка стилей
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
        color: #000000;
    }
    .st-bw {
        background-color: white;
    }
    .st-at {
        background-color: white;
    }
    .css-18e3th9 {
        padding: 1rem 1rem 10rem;
    }
    h1, h2, h3, h4, h5, h6, p, div {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

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

# Функция для создания PDF отчета
def create_pdf_report(df, forecast_df, recommendations, filtered_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Заголовок
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Аналитический отчет по продажам", ln=1, align='C')
    pdf.ln(10)
    
    # Основные метрики
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Ключевые метрики", ln=1)
    pdf.set_font("Arial", size=12)
    
    total_sales = df['Объем продаж'].sum()
    total_revenue = df['Выручка'].sum()
    avg_price = df['Сумма'].mean()
    unique_products = df['Вид продукта'].nunique()
    
    pdf.cell(200, 10, txt=f"Общий объем продаж: {total_sales:,.0f}", ln=1)
    pdf.cell(200, 10, txt=f"Общая выручка: {total_revenue:,.2f} руб.", ln=1)
    pdf.cell(200, 10, txt=f"Средний чек: {avg_price:.2f} руб.", ln=1)
    pdf.cell(200, 10, txt=f"Количество уникальных продуктов: {unique_products}", ln=1)
    pdf.ln(10)
    
    # Графики (сохраняем временные изображения)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Визуализация данных", ln=1)
    
    # Динамика продаж
    fig = px.line(
        filtered_df.groupby('Дата').agg({'Объем продаж': 'sum'}).reset_index(),
        x='Дата',
        y='Объем продаж',
        title='Динамика продаж'
    )
    img_bytes = fig.to_image(format="png")
    pdf.image(BytesIO(img_bytes), x=10, w=190)
    pdf.ln(5)
    
    # Продукты
    fig = px.bar(
        filtered_df.groupby('Вид продукта')['Объем продаж'].sum().reset_index(),
        x='Вид продукта',
        y='Объем продаж',
        title='Продажи по продуктам'
    )
    img_bytes = fig.to_image(format="png")
    pdf.image(BytesIO(img_bytes), x=10, w=190)
    pdf.ln(5)
    
    # Локации
    fig = px.bar(
        filtered_df.groupby('Местоположение')['Выручка'].sum().reset_index(),
        x='Местоположение',
        y='Выручка',
        title='Выручка по локациям'
    )
    img_bytes = fig.to_image(format="png")
    pdf.image(BytesIO(img_bytes), x=10, w=190)
    pdf.ln(10)
    
    # Прогноз
    if forecast_df is not None:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Прогноз продаж", ln=1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df[forecast_df['Тип'] == 'Факт']['Дата'],
            y=forecast_df[forecast_df['Тип'] == 'Факт']['Объем продаж'],
            name='Факт',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df[forecast_df['Тип'] == 'Прогноз']['Дата'],
            y=forecast_df[forecast_df['Тип'] == 'Прогноз']['Объем продаж'],
            name='Прогноз',
            line=dict(color='red', dash='dot')
        ))
        img_bytes = fig.to_image(format="png")
        pdf.image(BytesIO(img_bytes), x=10, w=190)
        pdf.ln(5)
    
    # Рекомендации
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Рекомендации", ln=1)
    pdf.set_font("Arial", size=12)
    
    for rec in recommendations:
        pdf.multi_cell(0, 10, txt=rec)
    
    return pdf

# Интерфейс приложения
st.title("📈 Sales-smart")

# Загрузка данных
with st.expander("📁 Загрузить данные", expanded=True):
    st.markdown("""
    **Требования к данным:**
    
    Для корректной работы приложения загружаемый файл должен соответствовать следующим требованиям:
    
    - Формат файла: **Excel (.xlsx)**
    - Обязательные колонки:
        - **Дата** - дата продажи в формате ДД.ММ.ГГГГ
        - **Объем продаж** - количество проданных единиц (числовое значение)
        - **Вид продукта** - наименование товара или услуги
        - **Местоположение** - точка продажи или филиал
        - **Сумма** - цена за единицу товара
        - **Тип покупателя** - категория клиента (розничный, оптовый и т.д.)
    
    **Рекомендации:**
    - Данные должны быть полными и актуальными
    - Период анализа должен содержать не менее 30 дней данных
    - Избегайте пустых значений в обязательных колонках
    """)
    
    uploaded_file = st.file_uploader(
        "Выберите файл продаж (Excel)",
        type="xlsx",
        help="Загрузите файл, соответствующий указанным требованиям"
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
                labels={'Объем продаж': 'Объем', 'Дата': 'Дата'}
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
                    color='Вид продукта',
                    text_auto=True
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
                    title='Цена vs Объем продаж',
                    hover_name='Вид продукта'
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Анализ продаж по локациям")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    filtered_df.groupby('Местоположение').agg({
                        'Объем продаж': 'sum',
                        'Выручка': 'sum'
                    }).reset_index(),
                    x='Местоположение',
                    y='Выручка',
                    title='Выручка по локациям',
                    color='Местоположение',
                    text_auto='.2s'
                )
                fig.update_traces(
                    textfont_size=12,
                    textangle=0,
                    textposition="outside"
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)

            
            with col2:
                fig = px.bar(
                    filtered_df.groupby(['Местоположение', 'Вид продукта'])['Объем продаж'].sum().reset_index(),
                    x='Местоположение',
                    y='Объем продаж',
                    color='Вид продукта',
                    title='Распределение продаж по продуктам и локациям',
                    barmode='stack'
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Динамика продаж по локациям")
            fig = px.line(
                filtered_df.groupby(['Местоположение', pd.Grouper(key='Дата', freq='W-MON')])['Объем продаж'].sum().reset_index(),
                x='Дата',
                y='Объем продаж',
                color='Местоположение',
                title='Недельная динамика продаж продаж',
                                markers=True,
                line_shape="spline"
            )
            fig.update_xaxes(tickformat="%d %b", dtick="M1")
            fig.update_layout(
                hovermode="x unified",
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
                        line=dict(color='blue')
                    )
                    
                    # Прогноз
                    fig.add_trace(go.Scatter(
                        x=forecast_df[forecast_df['Тип'] == 'Прогноз']['Дата'],
                        y=forecast_df[forecast_df['Тип'] == 'Прогноз']['Объем продаж'],
                        name='Прогноз',
                        line=dict(color='red', dash='dot')
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
                        fillcolor='rgba(255,0,0,0.1)',
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
        
        # Создание и скачивание PDF отчета
        pdf = create_pdf_report(df, forecast_df, recommendations, filtered_df)
        pdf_output = pdf.output(dest='S').encode('latin-1')
        st.download_button(
            label="📥 Скачать отчет (PDF)",
            data=pdf_output,
            file_name="sales_report.pdf",
            mime="application/pdf"
        )

        # Экспорт данных
        st.download_button(
            label="📄 Скачать данные (CSV)",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name="sales_data.csv",
            mime="text/csv"
        )
