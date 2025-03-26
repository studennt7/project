import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_data(file):
    """Загружает данные и выполняет базовый анализ"""
    try:
        if file is None:
            return "Ошибка: Файл не был загружен", None, None
        
        # Загрузка данных
        df = pd.read_excel(file.name)
        
        # Проверка наличия необходимых колонок
        required_columns = ['Дата', 'Объем продаж', 'Вид продукта', 'Местоположение', 'Цена', 'Тип покупателя']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return f"В файле отсутствуют колонки: {', '.join(missing_columns)}", None, None
        
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Выручка'] = df['Объем продаж'] * df['Цена']
        
        # Основные показатели
        total_sales = df['Объем продаж'].sum()
        total_revenue = df['Выручка'].sum()
        avg_price = df['Цена'].mean()
        
        info = f"Анализ данных о продажах\n\n" \
               f"Общий объем продаж: {total_sales:,.0f}\n" \
               f"Общая выручка: {total_revenue:,.2f} руб.\n" \
               f"Средняя цена: {avg_price:.2f} руб.\n"
        
        # Динамика продаж
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Дата', y='Объем продаж')
        plt.title('Динамика продаж')
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_path = "time_sales.png"
        plt.savefig(chart_path)
        plt.close()
        
        return info, chart_path
        
    except Exception as e:
        return f"Ошибка обработки файла: {str(e)}", None

def generate_recommendations(df):
    """Простые рекомендации на основе данных"""
    top_product = df.groupby('Вид продукта').agg({
        'Объем продаж': 'sum'
    }).sort_values('Объем продаж', ascending=False).head(1)
    
    recommendations = [
        f"Рекомендуем увеличить маркетинговые усилия для продукта '{top_product.index[0]}'",
        "Оптимизировать ценообразование",
        "Увеличить ассортимент"
    ]
    
    return "\n".join(recommendations)

# Интерфейс Gradio
with gr.Blocks() as app:
    gr.Markdown("# Упрощенная версия системы анализа продаж")
    gr.Markdown("Загрузите файл с данными о продажах, чтобы получить анализ и рекомендации.")
    
    file_input = gr.File(label="Загрузите Excel файл")
    analyze_btn = gr.Button("Анализировать")
    
    with gr.Row():
        with gr.Column():
            output_info = gr.Textbox(label="Общая информация", lines=6)
            output_chart = gr.Image(label="Динамика продаж")
        
        with gr.Column():
            recommendations_btn = gr.Button("Получить рекомендации")
            output_recommendations = gr.Textbox(label="Рекомендации", lines=6)
    
    def analyze(file):
        info, chart = load_and_analyze_data(file)
        return info, chart
    
    def recommendations(file):
        df = pd.read_excel(file.name)
        return generate_recommendations(df)
    
    analyze_btn.click(fn=analyze, inputs=[file_input], outputs=[output_info, output_chart])
    recommendations_btn.click(fn=recommendations, inputs=[file_input], outputs=[output_recommendations])

app.launch(debug=True)
