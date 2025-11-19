# coref/datasets/compound_capitals_es.py

import itertools
from collections import namedtuple

from coref.utils import prepend
from coref.datasets.common import (
    TrackFormatter,
    Substring,
    tokenize_prompt,
    recursive_add_offset,
)


class TEMPLATES:
    @classmethod
    def lookup(cls, template_type):
        if template_type == "normal":
            return cls.DEFAULT
        elif template_type == "noquestion":
            return cls.NOQUESTION
        raise Exception(f"unknown template {template_type}")

    class DEFAULT:
        @classmethod
        def generate_template(cls, *, vocab, template_context, prompt_id, context):
            @lambda f: f(**template_context)
            def ret(query_name, raw_query_name):
                names_perm, attrs_perm = vocab.get_shuffled_labels(prompt_id)
                return (
                    cls.template,
                    dict(
                        qn_subject=vocab.filtered_names[names_perm[query_name]]
                        if raw_query_name is None
                        else raw_query_name,
                    ),
                )

            return ret

        @classmethod
        def extract_template_indices(cls, full_output_indices):
            return {
                "qn_subject": full_output_indices["qn_subject"][0],
                "ans_subject": full_output_indices["qn_subject"][-1],
            }

        @classmethod
        def instantiate(cls, vocab, statement, prompt_id, template_context):
            names_perm, attrs_perm = vocab.get_shuffled_labels(prompt_id)
            if (
                statement.type == "normal"
                or statement.type == "ref"
                or statement.type == "extended"
            ):
                context_template = cls.lookup(statement.type)
                name = vocab.filtered_names[names_perm[statement.name]]
                country, _ = vocab.filtered_country_capital_pairs[
                    attrs_perm[statement.attr]
                ]
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    context_template,
                    subject=name,
                    country=country,
                )
                return (
                    cur_ctx,
                    {
                        "subject": ctx_idx_map["subject"][0],
                        "country": ctx_idx_map["country"][0],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )
            elif statement.type == "coref":
                context_template = cls.lookup(statement.type)
                names = ["primera_persona", "segunda_persona"]
                name = names[statement.name]
                country, _ = vocab.filtered_country_capital_pairs[
                    attrs_perm[statement.attr]
                ]
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    context_template,
                    subject=name,
                    country=country,
                )
                return (
                    cur_ctx,
                    {
                        "subject": ctx_idx_map["subject"][0],
                        "country": ctx_idx_map["country"][0],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )
            elif statement.type == "parallel":
                parallel_template = cls.lookup(statement.type)
                all_names = [
                    vocab.filtered_names[names_perm[name]] for name in statement.name
                ]
                all_countries = [
                    vocab.filtered_country_capital_pairs[attrs_perm[attr]][0]
                    for attr in statement.attr
                ]
                names_format = cls.format_parallel(
                    [f"{{subject_{i}}}" for i in range(len(all_names))]
                )
                countries_format = cls.format_parallel(
                    [f"{{country_{i}}}" for i in range(len(all_names))]
                )
                context_format = parallel_template.format(
                    all_subjects=names_format, all_countries=countries_format
                )
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    context_format,
                    **{f"subject_{i}": name for i, name in enumerate(all_names)},
                    **{
                        f"country_{i}": country
                        for i, country in enumerate(all_countries)
                    },
                )
                return (
                    cur_ctx,
                    {
                        "subject": [
                            ctx_idx_map[f"subject_{i}"][0]
                            for i in range(len(all_names))
                        ],
                        "country": [
                            ctx_idx_map[f"country_{i}"][0]
                            for i in range(len(all_names))
                        ],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )
            else:
                raise Exception(f"unknown statement type {statement.type}")

        @classmethod
        def lookup(cls, context_type):
            if context_type == "normal":
                return cls.context_template
            elif context_type == "extended":
                return cls.context_extended_template
            elif context_type == "coref":
                return cls.coref_template
            elif context_type == "ref":
                return cls.ref_template
            elif context_type == "parallel":
                return cls.parallel_template
            raise Exception(f"unknown context template {context_type}")

        # ==== AQUÍ VAN LOS TEXTOS EN ESPAÑOL ====

        template = """Responde la pregunta basándote en el siguiente contexto. Mantén la respuesta breve.

--- Contexto ---{context}

--- Pregunta ---
Pregunta: ¿De qué país posee la ciudadanía {qn_subject}?

Respuesta: {qn_subject} posee la ciudadanía del país"""

        context_template = (
            """ {subject} posee la ciudadanía del país {country}."""
        )
        context_extended_template = (
            """ {subject} actualmente posee la ciudadanía del país {country}."""
        )
        coref_template = (
            """ La {subject} luego pasa a poseer la ciudadanía del país {country}."""
        )
        ref_template = (
            """ Más tarde {subject} pasa a poseer la ciudadanía del país {country}."""
        )
        parallel_template = (
            """ {all_subjects} poseen la ciudadanía de los países {all_countries} respectivamente."""
        )

        @classmethod
        def format_parallel(cls, things):
            return ", ".join(things[:-1]) + " y " + things[-1]

    class NOQUESTION:
        @classmethod
        def lookup(cls, context_type):
            if context_type == "normal":
                return cls.context_template
            elif context_type == "coref":
                return cls.coref_template
            elif context_type == "ref":
                return cls.ref_template
            elif context_type == "parallel":
                return cls.parallel_template
            raise Exception(f"unknown context template {context_type}")

        template = """Completa la afirmación basándote en el contexto siguiente. Mantén la respuesta breve.

--- Contexto ---{context}

Afirmación: {qn_subject} posee la ciudadanía del país"""

        context_template = (
            """ {subject} posee la ciudadanía del país {country}."""
        )
        coref_template = (
            """ La {subject} luego pasa a poseer la ciudadanía del país {country}."""
        )
        ref_template = (
            """ Más tarde {subject} pasa a poseer la ciudadanía del país {country}."""
        )
        parallel_template = (
            """ {all_subjects} poseen la ciudadanía de los países {all_countries} respectivamente."""
        )

        def format_parallel(things):
            return ", ".join(things[:-1]) + " y " + things[-1]
