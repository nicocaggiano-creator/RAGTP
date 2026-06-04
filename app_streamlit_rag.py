#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

import rag_veterinaria as rag


APP_DIR = Path(__file__).resolve().parent
INDEX_DIR = APP_DIR / "rag_veterinaria_index"


def shorten(text: str, max_chars: int | None = 900) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if max_chars is None:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


@st.cache_resource(show_spinner=False)
def load_rag_index():
    return rag.load_index(INDEX_DIR)


def available_patients(index) -> list[str]:
    names = set()
    for doc in index["docs"]:
        name = doc.metadata.get("nombre_mascota")
        tutor = doc.metadata.get("tutor")
        if name:
            label = name if not tutor else f"{name} | {tutor}"
            names.add(label)
    return sorted(names, key=lambda value: value.lower())


def parse_patient_label(label: str) -> str | None:
    if not label or label == "Todos":
        return None
    return label.split("|", 1)[0].strip()


st.set_page_config(
    page_title="RAG veterinaria",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("RAG veterinaria")

try:
    index = load_rag_index()
except Exception as exc:
    st.error(f"No pude cargar el indice en {INDEX_DIR}: {exc}")
    st.stop()

with st.sidebar:
    st.header("Filtros")
    especie_label = st.selectbox("Especie", ["Todas", "Perro", "Gato"])
    seccion_label = st.selectbox(
        "Seccion",
        [
            "Todas",
            "perfil_paciente",
            "historia_clinica",
            "dieta_actual",
            "dieta_indicada",
            "suplementos_indicados",
            "etiqueta_balanceado",
            "preparacion",
            "indicaciones_generales",
            "seguimientos",
        ],
    )
    patient_options = ["Todos"] + available_patients(index)
    patient_label = st.selectbox("Paciente", patient_options)
    top_k = st.slider("Resultados", min_value=3, max_value=15, value=6, step=1)
    show_context = st.toggle("Mostrar texto completo recuperado", value=False)

question = st.text_input(
    "Pregunta",
    placeholder="Ej: Que pacientes tienen diarrea? / Que dieta indicada tiene Ivi?",
)

examples = [
    "Que pacientes tienen diarrea?",
    "Que gatos tienen indicado calcio?",
    "Que dieta indicada tiene Ivi?",
    "Que perros tienen problemas renales?",
]

cols = st.columns(len(examples))
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        question = example

if not question:
    st.info("Escribi una pregunta o elegi un ejemplo para consultar las fichas.")
    st.stop()

especie = None if especie_label == "Todas" else especie_label.lower()
seccion = None if seccion_label == "Todas" else seccion_label
paciente = parse_patient_label(patient_label)

results = rag.retrieve(
    question=question,
    index=index,
    top_k=top_k,
    especie=especie,
    paciente=paciente,
    seccion=seccion,
)

st.subheader("Resultados")

if not results:
    st.warning("No encontre evidencia suficiente con esos filtros.")
    st.stop()

st.caption(
    "Esta version muestra evidencia recuperada del RAG. Para una respuesta redactada tipo chatbot se puede sumar un LLM por API."
)

for rank, (doc, score) in enumerate(results, start=1):
    meta = doc.metadata
    title = (
        f"{rank}. {meta.get('nombre_mascota') or 'Sin nombre'}"
        f" | {meta.get('especie') or 'sin especie'}"
        f" | {meta.get('seccion')}"
        f" | score {score:.3f}"
    )
    with st.expander(title, expanded=rank <= 3):
        c1, c2, c3 = st.columns(3)
        c1.metric("Mascota", meta.get("nombre_mascota") or "-")
        c2.metric("Tutor", meta.get("tutor") or "-")
        c3.metric("Archivo", meta.get("archivo_origen") or "-")

        st.write(shorten(doc.text, None if show_context else 900))

        st.caption(
            f"Raza: {meta.get('raza') or '-'} | Sexo: {meta.get('sexo') or '-'} | "
            f"Edad: {meta.get('edad') or '-'} | Peso kg: {meta.get('peso_kg') or '-'}"
        )
