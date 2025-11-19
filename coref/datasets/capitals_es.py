# coref/datasets/capitals_es.py

import os
import re
import numpy as np
import pandas as pd

from coref import COREF_ROOT
from coref.datasets.common import BaseVocab, Statement


class Vocab(BaseVocab):
    """
    Vocabulario equivalente a CAPITALS pero en español.

    Extrae (persona, país) desde un CSV cuyos contextos contienen frases como:

        "Diego_Acosta posee la ciudadanía del país El_Salvador
         Donis_Escober posee la ciudadanía del país Mexico"

    Restricciones:
    - Los ATRIBUTOS (países) se filtran a 1 token (como en capitals original),
      porque se usan como respuestas con encode_single_word.
    - Los NOMBRES pueden tener varios tokens; no pasan por encode_single_word,
      así que no necesitan esa restricción.
    """

    type = "CAPITALS_ES"

    def __init__(
        self,
        tokenizer_type,
        split=None,
        is_parallel=False,
        csv_path=None,
    ):
        super().__init__(tokenizer_type)

        # Ruta por defecto al CSV (ajústala si tu archivo se llama distinto)
        if csv_path is None:
            csv_path = os.path.join(
                COREF_ROOT,
                "coref/datasets/raw/ciudadania.csv",
            )

        df = pd.read_csv(csv_path)

        # Detectar la columna de contexto
        if "prompt" in df.columns:
            context_col = "prompt"
        else:
            context_col = df.columns[0]

        persons = []
        countries = []

        # Regex para capturar: "<Persona> posee la ciudadanía del país <Pais>"
        pattern = r"(\S+)\s+posee la ciudadanía del país\s+(\S+)"

        for row in df.itertuples():
            context = getattr(row, context_col)
            matches = re.findall(pattern, context)
            for person, country in matches:
                persons.append(person)
                countries.append(country)

        # Deduplicar
        persons = sorted(set(persons))
        countries = sorted(set(countries))

        # Estructura de atributos igual que capitals: (country, capital)
        # Aquí no tenemos capital, así que usamos (pais, pais)
        pairs = [(c, c) for c in countries]

        # 👇 IMPORTANTE:
        # - NO filtramos nombres por longitud de token (para no quedarnos sin sujetos).
        # - SÍ filtramos países (atributos) a 1 token, como en capitals original.
        self.filtered_names = persons
        self.filtered_country_capital_pairs = self.filter_countries(pairs)

        # Train/test split igual que capitals.py
        self.simple_train_test_split(split)

        self.is_parallel = is_parallel

    def default_context(self, num_entities, entities=None, attributes=None):
        """
        Igual que en capitals.py:
        crea una lista de Statements con (entity_index, attribute_index, "normal")
        o una versión 'parallel' si is_parallel=True.
        """
        if entities is None:
            entities = range(num_entities)
        if attributes is None:
            attributes = range(num_entities)

        if not self.is_parallel:
            return [Statement(e, s, "normal") for e, s in zip(entities, attributes)]
        else:
            return [Statement(list(entities), list(attributes), "parallel")]
