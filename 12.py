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

# Функция для создания PDF
def create_pdf(df, filtered_df, forecast_df, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Заголовок
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Отчет по продажам", ln=1, align='C')
    pdf.ln(10)
    
    # Основные метрики
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Ключевые показатели", ln=1)
    pdf.set_font("Arial", size=12)
    
    total_sales = filtered_df['Объем продаж'].sum()
    total_revenue = filtered_df['Выручка'].sum()
    avg_price = filtered_df['Сумма'].mean()
    unique_products = filtered_df['Вид продукта'].nunique()
    
    pdf.cell(200, 10, txt=f"Общий объем продаж: {total_sales:,.0f}", ln=1)
    pdf.cell(200, 10, txt=f"Общая выручка: {total_revenue:,.2f} руб.", ln=1)
    pdf.cell(200, 10, txt=f"Средний чек: {avg_price:.2f} руб.", ln=1)
    pdf.cell(200, 10, txt=f"Количество продуктов: {unique_products}", ln=1)
    pdf.ln(10)
    
    # Рекомендации
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Рекомендации", ln=1)
    pdf.set_font("Arial", size=12)
    
    for rec in recommendations:
        pdf.cell(200, 10, txt=f"- {rec.replace('📌', '').strip()}", ln=1)
    
    # Прогноз
    if forecast_df is not None:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Прогноз продаж", ln=1)
        pdf.set_font("Arial", size=12)
        
        forecast_values = forecast_df[forecast_df['Тип'] == 'Прогноз']
        for index, row in forecast_values.iterrows():
            pdf.cell(200, 10, txt=f"{row['Дата'].strftime('%Y-%m-%d')}: {row['Объем продаж']:,.0f}", ln=1)
    
    return pdf.output(dest='S').encode('latin1')

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
                "Выявлена недельная сезонность. Оптимизируйте запасы и персонал соответственно."
            )
    except:
        pass
    
    top_products = df.groupby('Вид продукта')['Объем продаж'].sum().nlargest(3)
    if len(top_products) > 0:
        recommendations.append(
            f"Топ-3 продукта: {', '.join(top_products.index)}. Увеличьте их наличие."
        )
    
    customer_stats = df.groupby('Тип покупателя')['Выручка'].agg(['sum', 'count'])
    if len(customer_stats) > 1:
        best_customer = customer_stats['sum'].idxmax()
        recommendations.append(
            f"Основная выручка от '{best_customer}'. Разработайте программу лояльности."
        )
    
    location_stats = df.groupby('Местоположение')['Выручка'].sum()
    if len(location_stats) > 1:
        best_loc = location_stats.idxmax()
        worst_loc = location_stats.idxmin()
        recommendations.append(
            f"Лучшая локация: {best_loc}, проблемная: {worst_loc}. Изучите причины."
        )
    
    return recommendations if recommendations else ["Недостаточно данных для рекомендаций"]

# Интерфейс приложения
st.title("📈 Sales-smart - Аналитика продаж")

# Загрузка данных
with st.expander("📁 Загрузить данные", expanded=True):
    st.markdown("""
    **Требования к данным:**
    - Файл должен быть в формате Excel (.xlsx)
    - Обязательные колонки:
        - `Дата` - дата продажи (формат: ДД.ММ.ГГГГ)
        - `Объем продаж` - количество проданных единиц
        - `Вид продукта` - категория/название продукта
        - `Местоположение` - место продажи (филиал, город и т.д.)
        - `Сумма` - цена за единицу товара
        - `Тип покупателя` - категория покупателя (розница, опт и т.д.)
    - Данные должны быть в одной вкладке (первая вкладка будет использоваться по умолчанию)
    """)
    
    uploaded_file = st.file_uploader(
        "Выберите файл с данными о продажах",
        type="xlsx",
        help="Загрузите файл, соответствующий требованиям выше"
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
                color_discrete_sequence=['#1f77b4']
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
                    title='Цена vs Объем продаж'
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("""
            ### Анализ продаж по локациям
            На графиках ниже представлено распределение выручки и динамика продаж по разным локациям.
            Используйте эту информацию для выявления наиболее и наименее эффективных точек продаж.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    filtered_df.groupby('Местоположение')['Выручка'].sum().reset_index(),
                    names='Местоположение',
                    values='Выручка',
                    title='Доля выручки по локациям (%)',
                    hole=0.3
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Улучшенный график динамики по локациям
                loc_df = filtered_df.groupby(['Местоположение', 'Дата'])['Объем продаж'].sum().reset_index()
                
                fig = px.area(
                    loc_df,
                    x='Дата',
                    y='Объем продаж',
                    color='Местоположение',
                    title='Динамика продаж по локациям',
                    facet_col='Местоположение',
                    facet_col_wrap=2,
                    height=600
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    showlegend=False
                )
                
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
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
                   fig = px.line(
                    forecast_df,
                    x='Дата',
                    y='Объем продаж',
                    color='Тип',
                    title='Прогноз продаж на 30 дней',
                    color_discrete_map={'Факт': '#1f77b4', 'Прогноз': '#ff7f0e'}
                )
                fig.update_layout(
                    hovermode="x unified",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    forecast_df[forecast_df['Тип'] == 'Прогноз'][['Дата', 'Объем продаж']]
                    .set_index('Дата')
                    .style.format({'Объем продаж': '{:,.0f}'}),
                    use_container_width=True
                )
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
                        name='Доверительный интервал (±20%)'
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
                    st.markdown("""
                    **Детали прогноза:**
                    - Метод: Тройное экспоненциальное сглаживание (Holt-Winters)
                    - Учтены: тренд, сезонность (7 дней)
                    - Доверительный интервал: ±20%
                    """)
                    
                    forecast_data = forecast_df[forecast_df['Тип'] == 'Прогноз'][['Дата', 'Объем продаж']]
                    forecast_data = forecast_data.rename(columns={'Объем продаж': 'Прогноз'})
                    forecast_data['Дата'] = forecast_data['Дата'].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(
                        forecast_data.style.format({'Прогноз': '{:,.0f}'}),
                        height=600,
                        hide_index=True
                    )
            
            st.subheader("Рекомендации на основе анализа")
            recommendations = generate_recommendations(filtered_df)
            
            for rec in recommendations:
            st.write(f"📌 {rec}")
        
        # Генерация PDF
        pdf_data = create_pdf(df, filtered_df, forecast_df, recommendations)
        b64 = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="sales_report.pdf">📄 Скачать отчет в PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Дополнительные опции экспорта
        with st.expander("Дополнительные опции экспорта"):
            st.markdown("""
            ### Экспорт данных в различных форматах
            Выберите нужный формат для выгрузки данных:
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Экспорт в CSV
                st.download_button(
                    label="Скачать данные (CSV)",
                    data=filtered_df.to_csv(index=False).encode('utf-8'),
                    file_name="sales_data.csv",
                    mime="text/csv",
                    help="Скачать отфильтрованные данные в формате CSV"
                )
            
            with col2:
                # Экспорт в Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name='Sales Data')
                    if forecast_df is not None:
                        forecast_df.to_excel(writer, index=False, sheet_name='Forecast')
                output.seek(0)
                
                st.download_button(
                    label="Скачать данные (Excel)",
                    data=output,
                    file_name="sales_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Скачать данные и прогноз в формате Excel"
                )
                    
