import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta
from fpdf import FPDF
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# Настройки страницы
st.set_page_config(
    page_title="Sales Analytics Pro",
    page_icon="📊",
    layout="wide"
)

# Стилизация
st.markdown("""
<style>
.stApp {
    background-color: #f8f9fa;
}
.css-1d391kg {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.st-bb {
    color: #2c3e50;
}
.st-b7 {
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Функция для создания PDF отчета
def create_pdf_report(df, kpis, figures, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Заголовок
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Аналитический отчет Sales Analytics Pro", ln=1, align='C')
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
    
    temp_files = []
    for fig in figures:
        img_path = tempfile.mktemp(suffix='.png')
        fig.write_image(img_path, width=1000, height=600, scale=2)
        pdf.image(img_path, w=190)
        pdf.ln(5)
        temp_files.append(img_path)
    
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
    
    # Очистка временных файлов
    for file in temp_files:
        try:
            os.unlink(file)
        except:
            pass
    
    return pdf_path

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

# Функция прогнозирования
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
            f"🏆 Топ-3 продукта: {', '.join(top_products.index)}. Увеличьте их наличие и продвижение."
        )
    
    customer_stats = df.groupby('Тип покупателя')['Выручка'].agg(['sum', 'count'])
    if len(customer_stats) > 1:
        best_customer = customer_stats['sum'].idxmax()
        recommendations.append(
            f"👥 Основная выручка от '{best_customer}'. Разработайте специальную программу лояльности."
        )
    
    location_stats = df.groupby('Местоположение')['Выручка'].sum()
    if len(location_stats) > 1:
        best_loc = location_stats.idxmax()
        worst_loc = location_stats.idxmin()
        recommendations.append(
            f"📍 Лучшая локация: {best_loc}, проблемная: {worst_loc}. Изучите причины различий."
        )
    
    return recommendations if recommendations else ["🔎 Недостаточно данных для формирования рекомендаций"]

# Основной интерфейс
st.title("📈 Sales Analytics Pro")

# Панель загрузки данных
with st.expander("📁 Загрузить данные", expanded=True):
    st.markdown("""
    **Требования к файлу данных:**
    - Формат: Excel (.xlsx)
    - Обязательные колонки:
        - `Дата` (ГГГГ-ММ-ДД)
        - `Объем продаж` (число)
        - `Вид продукта` (текст)
        - `Местоположение` (текст)
        - `Сумма` (число)
        - `Тип покупателя` (текст)
    
    **Пример данных:**
    | Дата       | Объем | Продукт | Локация   | Сумма | Тип покупателя |
    |------------|-------|---------|-----------|-------|----------------|
    | 2023-01-01 | 10    | A       | Москва    | 100   | Розница        |
    """)
    
    uploaded_file = st.file_uploader(
        "Выберите файл с данными",
        type="xlsx",
        help="Загрузите файл в формате Excel с указанными колонками"
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
        st.sidebar.header("🔍 Фильтры")
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
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Динамика", "🛍️ Продукты", "🏢 Локации", "🔮 Прогноз"])
        
        figures_for_pdf = []
        
        with tab1:
            st.markdown("### Динамика продаж")
            fig1 = px.line(
                filtered_df.groupby('Дата').agg({'Объем продаж': 'sum'}).reset_index(),
                x='Дата',
                y='Объем продаж',
                template='plotly_white'
            )
            fig1.update_layout(
                xaxis_title='Дата',
                yaxis_title='Объем продаж',
                hovermode="x unified"
            )
            st.plotly_chart(fig1, use_container_width=True)
            figures_for_pdf.append(fig1)
            
        with tab2:
            st.markdown("### Анализ по продуктам")
            col1, col2 = st.columns(2)
            
            with col1:
                fig2 = px.bar(
                    filtered_df.groupby('Вид продукта')['Объем продаж'].sum().reset_index(),
                    x='Вид продукта',
                    y='Объем продаж',
                    color='Вид продукта'
                )
                st.plotly_chart(fig2, use_container_width=True)
                figures_for_pdf.append(fig2)
            
            with col2:
                fig3 = px.scatter(
                    filtered_df.groupby('Вид продукта').agg({
                        'Сумма': 'mean',
                        'Объем продаж': 'sum'
                    }).reset_index(),
                    x='Сумма',
                    y='Объем продаж',
                    size='Объем продаж',
                    color='Вид продукта',
                    hover_name='Вид продукта'
                )
                st.plotly_chart(fig3, use_container_width=True)
                figures_for_pdf.append(fig3)
        
        with tab3:
            st.markdown("### Анализ по локациям")
            col1, col2 = st.columns(2)
            
            with col1:
                fig4 = px.pie(
                    filtered_df.groupby('Местоположение')['Выручка'].sum().reset_index(),
                    names='Местоположение',
                    values='Выручка',
                    hole=0.3
                )
                st.plotly_chart(fig4, use_container_width=True)
                figures_for_pdf.append(fig4)
            
            with col2:
                # Улучшенный график по локациям
                location_weekly = filtered_df.groupby([
                    'Местоположение', 
                    pd.Grouper(key='Дата', freq='W-MON')
                ])['Объем продаж'].sum().reset_index()
                
                fig5 = px.line(
                    location_weekly,
                    x='Дата',
                    y='Объем продаж',
                    color='Местоположение',
                    facet_col='Местоположение',
                    facet_col_wrap=2,
                    height=600
                )
                st.plotly_chart(fig5, use_container_width=True)
                figures_for_pdf.append(fig5)
        
        with tab4:
            st.markdown("### Прогноз продаж")
            forecast_df, forecast_error = make_forecast(filtered_df)
            
            if forecast_error:
                st.warning(forecast_error)
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig6 = go.Figure()
                    
                    # Фактические данные
                    fig6.add_trace(go.Scatter(
                        x=forecast_df[forecast_df['Тип'] == 'Факт']['Дата'],
                        y=forecast_df[forecast_df['Тип'] == 'Факт']['Объем продаж'],
                        name='Факт',
                        line=dict(color='#3498db')
                    ))
                    
                    # Прогноз
                    fig6.add_trace(go.Scatter(
                        x=forecast_df[forecast_df['Тип'] == 'Прогноз']['Дата'],
                        y=forecast_df[forecast_df['Тип'] == 'Прогноз']['Объем продаж'],
                        name='Прогноз',
                        line=dict(color='#e74c3c', dash='dash')
                    ))
                    
                    # Доверительный интервал
                    fig6.add_trace(go.Scatter(
                        x=forecast_df[forecast_df['Тип'] == 'Прогноз']['Дата'],
                        y=forecast_df[forecast_df['Тип'] == 'Прогноз']['Объем продаж'] * 1.15,
                        fill=None,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig6.add_trace(go.Scatter(
                        x=forecast_df[forecast_df['Тип'] == 'Прогноз']['Дата'],
                        y=forecast_df[forecast_df['Тип'] == 'Прогноз']['Объем продаж'] * 0.85,
                        fill='tonexty',
                        mode='lines',
                        line=dict(width=0),
                        fillcolor='rgba(231, 76, 60, 0.1)',
                        name='Доверительный интервал'
                    ))
                    
                    fig6.update_layout(
                        title='Прогноз продаж с учетом сезонности',
                        xaxis_title='Дата',
                        yaxis_title='Объем продаж',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig6, use_container_width=True)
                    figures_for_pdf.append(fig6)
                
                with col2:
                    st.markdown("**Детали прогноза**")
                    st.dataframe(
                        forecast_df[forecast_df['Тип'] == 'Прогноз'][['Дата', 'Объем продаж']]
                        .rename(columns={'Объем продаж': 'Прогноз'})
                        .style.format({'Прогноз': '{:,.0f}'}),
                        height=400
                    )
            
            st.markdown("### Рекомендации")
            recommendations = generate_recommendations(filtered_df)
            for rec in recommendations:
                st.success(rec)
        
        # Экспорт данных
        st.sidebar.markdown("---")
        st.sidebar.header("📤 Экспорт")
        
        # CSV
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Скачать CSV",
            data=csv,
            file_name="sales_data.csv",
            mime="text/csv"
        )
        
        # PDF
        if st.sidebar.button("Создать PDF отчет"):
            with st.spinner("Формирование отчета..."):
                pdf_path = create_pdf_report(
                    filtered_df,
                    kpi_text,
                    figures_for_pdf,
                    recommendations
                )
                
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                st.sidebar.download_button(
                    label="Скачать PDF",
                    data=pdf_bytes,
                    file_name="sales_report.pdf",
                    mime="application/pdf"
                )
                
                os.unlink(pdf_path)
