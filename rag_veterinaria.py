#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG local para fichas clinicas veterinarias.

Uso rapido:
  python rag_veterinaria.py build
  python rag_veterinaria.py ask "Que pacientes tienen diarrea?"
  python rag_veterinaria.py ask "Que dieta tiene Ivi?" --paciente Ivi
  python rag_veterinaria.py ask "Gatos con calcio" --especie gato --top-k 8

Si Ollama esta corriendo:
  python rag_veterinaria.py ask "Resumen de pacientes con enfermedad renal" --ollama-model llama3.2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_DATA_PATH = Path(
    r"C:\Users\ncaggiano\Desktop\NICO\CAPACITACION\MAESTRIA EN CIENCIAS DE DATOS\MATERIAS\TEXT MINING\TP\resultados_completos.json"
)
DEFAULT_INDEX_DIR = Path("rag_veterinaria_index")

VALID_SECTIONS = {
    "perfil_paciente",
    "historia_clinica",
    "dieta_actual",
    "dieta_indicada",
    "suplementos_indicados",
    "etiqueta_balanceado",
    "preparacion",
    "indicaciones_generales",
    "seguimientos",
}


@dataclass
class RagDocument:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


RagDocument.__module__ = "rag_veterinaria"
sys.modules.setdefault("rag_veterinaria", sys.modules[__name__])


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\ufffd", "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_key(value: Any) -> str:
    text = clean_text(value).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_species(value: Any) -> str:
    text = normalize_key(value)
    if text in {"perro", "canino", "canina"}:
        return "perro"
    if text in {"gato", "felino", "felina"}:
        return "gato"
    return text


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def parse_float(value: Any) -> Optional[float]:
    text = clean_text(value).replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def stable_patient_id(record: Dict[str, Any], idx: int) -> str:
    name = normalize_key(record.get("nombre_mascota"))
    tutor = normalize_key(record.get("tutor"))
    source = normalize_key(record.get("_archivo_origen"))
    raw = "_".join(part for part in [name, tutor, source] if part)
    raw = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return raw.upper()[:80] if raw else f"PACIENTE_{idx + 1}"


def flatten_items(items: Any) -> str:
    if not items:
        return ""
    if isinstance(items, list):
        lines = []
        for item in items:
            if isinstance(item, dict):
                parts = []
                for key in ["tipo", "ingrediente", "cantidad", "unidad", "notas"]:
                    val = clean_text(item.get(key))
                    if val:
                        parts.append(f"{key}: {val}")
                if parts:
                    lines.append("; ".join(parts))
            else:
                text = clean_text(item)
                if text:
                    lines.append(text)
        return "\n".join(lines)
    if isinstance(items, dict):
        parts = [f"{key}: {clean_text(val)}" for key, val in items.items() if clean_text(val)]
        return "\n".join(parts)
    return clean_text(items)


def base_metadata(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "id_paciente": stable_patient_id(record, idx),
        "nombre_mascota": clean_text(record.get("nombre_mascota")),
        "tutor": clean_text(record.get("tutor")),
        "telefono": clean_text(record.get("telefono")),
        "veterinaria": clean_text(record.get("veterinaria_derivadora")),
        "fecha": clean_text(record.get("fecha")),
        "especie": normalize_species(record.get("especie")),
        "raza": clean_text(record.get("raza")),
        "sexo": clean_text(record.get("sexo")),
        "edad": clean_text(record.get("edad")),
        "peso_kg": parse_float(record.get("peso_kg")),
        "archivo_origen": clean_text(record.get("_archivo_origen")),
        "tiene_error": bool(record.get("error")),
    }


def make_section_doc(
    docs: List[RagDocument],
    meta: Dict[str, Any],
    section: str,
    title: str,
    body: str,
) -> None:
    body = clean_text(body)
    if not body:
        return
    full_text = (
        f"Paciente: {meta.get('nombre_mascota') or 'sin nombre'}\n"
        f"Tutor: {meta.get('tutor') or 'sin tutor'}\n"
        f"Especie: {meta.get('especie') or 'sin especie'}\n"
        f"Raza: {meta.get('raza') or 'sin raza'}\n"
        f"Seccion: {section}\n"
        f"{title}:\n{body}"
    )
    section_meta = dict(meta)
    section_meta["seccion"] = section
    section_meta["titulo"] = title
    doc_id = f"{meta['id_paciente']}::{section}::{len(docs)}"
    docs.append(RagDocument(doc_id=doc_id, text=full_text, metadata=section_meta))


def records_to_documents(records: List[Dict[str, Any]]) -> List[RagDocument]:
    docs: List[RagDocument] = []
    for idx, record in enumerate(records):
        if record.get("error"):
            continue
        meta = base_metadata(record, idx)

        profile = "\n".join(
            f"{label}: {value}"
            for label, value in [
                ("Nombre mascota", meta["nombre_mascota"]),
                ("Tutor", meta["tutor"]),
                ("Telefono", meta["telefono"]),
                ("Veterinaria derivadora", meta["veterinaria"]),
                ("Fecha", meta["fecha"]),
                ("Especie", meta["especie"]),
                ("Raza", meta["raza"]),
                ("Sexo", meta["sexo"]),
                ("Edad", meta["edad"]),
                ("Peso kg", meta["peso_kg"]),
            ]
            if value not in {"", None}
        )
        make_section_doc(docs, meta, "perfil_paciente", "Perfil del paciente", profile)

        clinical = "\n".join(
            f"{label}: {value}"
            for label, value in [
                ("Condicion", record.get("condicion")),
                ("Antecedentes", record.get("antecedentes")),
                ("Analisis de sangre", record.get("analisis_sangre")),
                ("Actividad fisica", record.get("actividad_fisica")),
                ("Estado actual", record.get("estado_actual")),
                ("RED", record.get("red")),
            ]
            if clean_text(value)
        )
        make_section_doc(docs, meta, "historia_clinica", "Historia clinica", clinical)
        make_section_doc(docs, meta, "dieta_actual", "Dieta actual", flatten_items(record.get("dieta_actual")))
        make_section_doc(docs, meta, "dieta_indicada", "Dieta indicada", flatten_items(record.get("dieta_indicada")))
        make_section_doc(
            docs,
            meta,
            "suplementos_indicados",
            "Suplementos indicados",
            flatten_items(record.get("suplementos_indicados")),
        )
        make_section_doc(
            docs,
            meta,
            "etiqueta_balanceado",
            "Etiqueta del balanceado",
            flatten_items(record.get("etiqueta_balanceado")),
        )
        make_section_doc(docs, meta, "preparacion", "Preparacion", record.get("preparacion"))
        make_section_doc(
            docs,
            meta,
            "indicaciones_generales",
            "Indicaciones generales",
            record.get("indicaciones_generales"),
        )
        make_section_doc(docs, meta, "seguimientos", "Seguimientos", flatten_items(record.get("seguimientos")))

    return docs


def build_vectorizer() -> FeatureUnion:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=True,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])


def save_sqlite(records: List[Dict[str, Any]], db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS pacientes")
    cur.execute(
        """
        CREATE TABLE pacientes (
            id_paciente TEXT PRIMARY KEY,
            nombre_mascota TEXT,
            tutor TEXT,
            telefono TEXT,
            veterinaria TEXT,
            fecha TEXT,
            especie TEXT,
            raza TEXT,
            sexo TEXT,
            edad TEXT,
            peso_kg REAL,
            archivo_origen TEXT,
            antecedentes TEXT,
            estado_actual TEXT,
            dieta_indicada TEXT,
            suplementos_indicados TEXT,
            indicaciones_generales TEXT
        )
        """
    )
    for idx, record in enumerate(records):
        if record.get("error"):
            continue
        meta = base_metadata(record, idx)
        cur.execute(
            """
            INSERT OR REPLACE INTO pacientes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta["id_paciente"],
                meta["nombre_mascota"],
                meta["tutor"],
                meta["telefono"],
                meta["veterinaria"],
                meta["fecha"],
                meta["especie"],
                meta["raza"],
                meta["sexo"],
                meta["edad"],
                meta["peso_kg"],
                meta["archivo_origen"],
                clean_text(record.get("antecedentes")),
                clean_text(record.get("estado_actual")),
                flatten_items(record.get("dieta_indicada")),
                flatten_items(record.get("suplementos_indicados")),
                clean_text(record.get("indicaciones_generales")),
            ),
        )
    conn.commit()
    conn.close()


def build_index(data_path: Path, index_dir: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {data_path}")

    index_dir.mkdir(parents=True, exist_ok=True)
    with data_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    docs = records_to_documents(records)
    if not docs:
        raise RuntimeError("No se generaron documentos validos para indexar.")

    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform([doc.text for doc in docs])

    with (index_dir / "rag_index.pkl").open("wb") as f:
        pickle.dump({"docs": docs, "vectorizer": vectorizer, "matrix": matrix}, f)

    save_sqlite(records, index_dir / "pacientes.sqlite")

    valid_records = sum(1 for item in records if not item.get("error"))
    print(f"OK: indice construido en {index_dir.resolve()}")
    print(f"Registros validos: {valid_records}")
    print(f"Documentos/chunks: {len(docs)}")
    print(f"Base estructurada: {(index_dir / 'pacientes.sqlite').resolve()}")


def load_index(index_dir: Path) -> Dict[str, Any]:
    index_file = index_dir / "rag_index.pkl"
    if not index_file.exists():
        raise FileNotFoundError(
            f"No existe {index_file}. Primero ejecuta: python rag_veterinaria.py build"
        )
    with index_file.open("rb") as f:
        return pickle.load(f)


def metadata_matches(
    metadata: Dict[str, Any],
    especie: Optional[str],
    paciente: Optional[str],
    seccion: Optional[str],
) -> bool:
    if especie and normalize_species(especie) != metadata.get("especie"):
        return False
    if seccion and seccion != metadata.get("seccion"):
        return False
    if paciente:
        needle = normalize_key(paciente)
        haystack = " ".join(
            [
                normalize_key(metadata.get("id_paciente")),
                normalize_key(metadata.get("nombre_mascota")),
                normalize_key(metadata.get("tutor")),
                normalize_key(metadata.get("archivo_origen")),
            ]
        )
        if needle not in haystack:
            return False
    return True


def retrieve(
    question: str,
    index: Dict[str, Any],
    top_k: int,
    especie: Optional[str] = None,
    paciente: Optional[str] = None,
    seccion: Optional[str] = None,
) -> List[Tuple[RagDocument, float]]:
    docs: List[RagDocument] = index["docs"]
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]

    allowed = [
        i
        for i, doc in enumerate(docs)
        if metadata_matches(doc.metadata, especie=especie, paciente=paciente, seccion=seccion)
    ]
    if not allowed:
        return []

    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix[allowed]).ravel()
    order = np.argsort(scores)[::-1][:top_k]
    return [(docs[allowed[i]], float(scores[i])) for i in order if scores[i] > 0]


def compact_context(results: List[Tuple[RagDocument, float]], max_chars: int = 9000) -> str:
    blocks = []
    total = 0
    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        block = (
            f"[Fuente {rank} | score={score:.3f}]\n"
            f"Paciente: {meta.get('nombre_mascota')} | Tutor: {meta.get('tutor')} | "
            f"Especie: {meta.get('especie')} | Seccion: {meta.get('seccion')} | "
            f"Archivo: {meta.get('archivo_origen')}\n"
            f"{doc.text}"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    return f"""Sos un asistente para consultar fichas clinicas veterinarias.
Responde solo con la informacion del contexto.
Si no hay evidencia suficiente, decilo claramente.
Cita mascota, tutor, seccion y archivo origen cuando corresponda.
No inventes diagnosticos ni indicaciones.

Pregunta:
{question}

Contexto recuperado:
{context}

Respuesta:"""


def call_ollama(prompt: str, model: str, host: str) -> str:
    url = host.rstrip("/") + "/api/generate"
    response = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False, "temperature": 0.1},
        timeout=120,
    )
    response.raise_for_status()
    return clean_text(response.json().get("response", ""))


def extractive_answer(question: str, results: List[Tuple[RagDocument, float]]) -> str:
    if not results:
        return "No encontre evidencia suficiente en el indice para responder esa consulta."

    lines = [
        "No use un LLM generativo; te muestro la evidencia recuperada mas relevante.",
        "",
    ]
    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        snippet = doc.text
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if len(snippet) > 650:
            snippet = snippet[:650].rsplit(" ", 1)[0] + "..."
        lines.append(
            f"{rank}. {meta.get('nombre_mascota') or 'Sin nombre'} "
            f"({meta.get('especie') or 'sin especie'}), tutor {meta.get('tutor') or 'sin tutor'} "
            f"| seccion {meta.get('seccion')} | archivo {meta.get('archivo_origen')} | score {score:.3f}"
        )
        lines.append(f"   {snippet}")
    return "\n".join(lines)


def ask_question(args: argparse.Namespace) -> None:
    index = load_index(args.index_dir)
    results = retrieve(
        question=args.question,
        index=index,
        top_k=args.top_k,
        especie=args.especie,
        paciente=args.paciente,
        seccion=args.seccion,
    )
    context = compact_context(results)

    if args.show_context:
        print("\n=== CONTEXTO RECUPERADO ===\n")
        print(context or "Sin contexto recuperado.")
        print("\n=== RESPUESTA ===\n")

    if args.ollama_model:
        if not results:
            print("No encontre evidencia suficiente en el indice para responder esa consulta.")
            return
        prompt = build_prompt(args.question, context)
        try:
            print(call_ollama(prompt, model=args.ollama_model, host=args.ollama_host))
        except Exception as exc:
            print(f"No pude llamar a Ollama ({exc}). Devuelvo respuesta extractiva.\n")
            print(extractive_answer(args.question, results))
    else:
        print(extractive_answer(args.question, results))


def inspect_index(index_dir: Path) -> None:
    index = load_index(index_dir)
    docs: List[RagDocument] = index["docs"]
    species: Dict[str, int] = {}
    sections: Dict[str, int] = {}
    patients = set()
    for doc in docs:
        species[doc.metadata.get("especie") or ""] = species.get(doc.metadata.get("especie") or "", 0) + 1
        sections[doc.metadata.get("seccion") or ""] = sections.get(doc.metadata.get("seccion") or "", 0) + 1
        patients.add(doc.metadata.get("id_paciente"))

    print(f"Pacientes: {len(patients)}")
    print(f"Documentos/chunks: {len(docs)}")
    print("Especies:")
    for key, val in sorted(species.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {key or 'sin especie'}: {val}")
    print("Secciones:")
    for key, val in sorted(sections.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {key}: {val}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG local para fichas clinicas veterinarias")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Ruta al resultados_completos.json")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="Carpeta del indice")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Construye el indice local")
    subparsers.add_parser("inspect", help="Muestra resumen del indice")

    ask = subparsers.add_parser("ask", help="Consulta el RAG")
    ask.add_argument("question", help="Pregunta en lenguaje natural")
    ask.add_argument("--top-k", type=int, default=6, help="Cantidad de chunks recuperados")
    ask.add_argument("--especie", choices=["perro", "gato"], help="Filtro por especie")
    ask.add_argument("--paciente", help="Filtro por nombre de mascota, tutor, id o archivo")
    ask.add_argument("--seccion", choices=sorted(VALID_SECTIONS), help="Filtro por seccion")
    ask.add_argument("--show-context", action="store_true", help="Muestra el contexto recuperado")
    ask.add_argument("--ollama-model", help="Modelo de Ollama para generar respuesta")
    ask.add_argument("--ollama-host", default="http://localhost:11434", help="Host de Ollama")

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    try:
        if args.command == "build":
            build_index(args.data, args.index_dir)
        elif args.command == "inspect":
            inspect_index(args.index_dir)
        elif args.command == "ask":
            ask_question(args)
        else:
            parser.error(f"Comando desconocido: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
