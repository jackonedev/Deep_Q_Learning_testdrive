import unicodedata
import spacy
from tqdm import tqdm
import pandas as pd

nlp = spacy.load("es_core_news_sm")


def eliminar_caracteres_no_imprimibles(
    input_string: str, conservar_simbolos: bool = False
) -> str:
    def eliminar_tildes(content: str) -> str:
        cleaned_content = ""
        for c in content:
            if c in "áéíóúÁÉÍÓÚáéíóúÁÉÍÓÚó":
                cleaned_content += unicodedata.normalize("NFD", c)[0]
            elif c in ["ñ", "Ñ"]:
                cleaned_content += c
            # estos son los valores que eliminamos
            elif not conservar_simbolos and (
                c in "&'()*+‘<=>[]^`{|}~ýª!?¿¡.,/⁉️‼:\""
            ):  ## linea modificada junto con los parametros de la funcion
                continue
            else:
                cleaned_content += c
        return cleaned_content

    input_string = input_string.lower()

    input_string = eliminar_tildes(input_string)

    lista = input_string.split(" ")
    cleaned_content = []
    for elemento in lista:
        if elemento:
            cleaned_content.append(elemento)
    input_string = " ".join(cleaned_content)
    return input_string


def aplicar_stopwords(contenido: list) -> list:
    corpus_sample = []
    for parrafo in tqdm(contenido):
        doc = nlp(parrafo)
        corpus_sample.append(
            " ".join([token.text for token in doc if not token.is_stop])
        )
    return corpus_sample
