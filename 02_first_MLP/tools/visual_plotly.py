import pandas as pd
import random
import plotly.express as px
import matplotlib.pyplot as plt


def random_color_hex():
    return "#%02x%02x%02x" % (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )


def grafico_torta(
    df: pd.DataFrame, prediction_index: int, random_colors: bool = False
) -> None:
    ### Variables hardcodeadas
    colors = [
        "#FFD700",
        "#FF4500",
        "#1E90FF",
        "#FF1493",
        "#32CD32",
    ]  # Colores personalizados
    labels = ["sentiment_i", "emotions_6_max_label", "emotions_26_max_label"]
    titles = [
        "Predicciones de Sentimiento",
        "Predicciones de Emociones",
        "Predicciones de Emociones",
    ]
    columnas = ["sentimiento", "emocion", "emocion"]

    if prediction_index == 0:
        label, title, columna = labels[0], titles[0], columnas[0]
    elif prediction_index == 1:
        label, title, columna = labels[1], titles[1], columnas[1]
    elif prediction_index == 2:
        label, title, columna = labels[2], titles[2], columnas[2]
    else:
        raise (ValueError, "Elija un valor entre '0', '1' o '2'")

    df = (
        (df[label].value_counts() / df[label].value_counts().sum() * 100)
        .round(1)
        .to_frame()
    )

    # Resetear el índice del DataFrame
    df = df.reset_index()

    # Renombrar las columnas
    df.columns = [columna, "porcentaje"]

    # colores random
    if random_colors:
        colors = [random_color_hex() for _ in range(df.shape[0])]

    # Variables para personalizar el estilo del gráfico
    title = title

    # Generar el gráfico de torta con Plotly Express
    fig = px.pie(
        df,
        values="porcentaje",
        names=columna,
        title=title,
        color_discrete_sequence=colors,
    )

    # Mostrar el gráfico en la notebook
    return fig
