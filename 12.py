import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# Function to load and analyze data
@st.cache_data
def load_and_analyze_data(file):
    try:
        df = pd.read_excel(file)
        
        # Check required columns
        required_columns = ['Дата', 'Объем продаж', 'Вид продукта', 'Местоположение', 'Сумма', 'Тип покупателя']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return f"В файле отсутствуют колонки: {', '.join(missing_columns)}", None
        
        # Data processing
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Выручка'] = df['Объем продаж'] * df['Сумма']
        
        return None, df
        
    except Exception as e:
        return f"Ошибка обработки файла: {str(e)}", None

# UI Layout
st.title("📊 Sales Smart Analytics")

# File upload section
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

# Main analysis section
if uploaded_file is not None:
    error_msg, df = load_and_analyze_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    elif df is not None:
        # Calculate KPIs
        total_sales = df['Объем продаж'].sum()
        total_revenue = df['Выручка'].sum()
        avg_price = df['Сумма'].mean()
        unique_products = df['Вид продукта'].nunique()
        
        # Display KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Общий объем продаж", f"{total_sales:,.0f}")
        col2.metric("Общая выручка", f"{total_revenue:,.2f} руб.")
        col3.metric("Средняя сумма", f"{avg_price:,.2f} руб.")
        col4.metric("Видов продуктов", unique_products)
        
        # Date range selector
        min_date = df['Дата'].min().date()
        max_date = df['Дата'].max().date()
        selected_dates = st.date_input(
            "Выберите диапазон дат",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data based on date selection
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
            filtered_df = df[(df['Дата'].dt.date >= start_date) & (df['Дата'].dt.date <= end_date)]
        else:
            filtered_df = df
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["Динамика продаж", "Продукты", "Локации"])
        
        with tab1:
            fig = px.line(
                filtered_df, 
                x='Дата', 
                y='Объем продаж',
                title='Динамика продаж',
                labels={'Объем продаж': 'Объем продаж', 'Дата': 'Дата'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            product_sales = filtered_df.groupby('Вид продукта')['Объем продаж'].sum().reset_index()
            fig = px.bar(
                product_sales,
                x='Вид продукта',
                y='Объем продаж',
                title='Продажи по видам продуктов'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            location_sales = filtered_df.groupby('Местоположение')['Выручка'].sum().reset_index()
            fig = px.pie(
                location_sales,
                names='Местоположение',
                values='Выручка',
                title='Выручка по локациям'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data export
        st.download_button(
            label="Скачать обработанные данные",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='processed_sales_data.csv',
            mime='text/csv'
        )
