# RAG veterinaria

Este prototipo arma un RAG local sobre las fichas clinicas extraidas en
`resultados_completos.json`.

## 1. Construir el indice

```powershell
cd "C:\Users\ncaggiano\Documents\TEXT MINING"
python rag_veterinaria.py build
```

Esto genera:

- `rag_veterinaria_index/rag_index.pkl`: indice de recuperacion.
- `rag_veterinaria_index/pacientes.sqlite`: tabla estructurada de pacientes.

## 2. Inspeccionar el indice

```powershell
python rag_veterinaria.py inspect
```

## 3. Hacer consultas sin LLM

Devuelve evidencia recuperada, con paciente, seccion y archivo origen.

```powershell
python rag_veterinaria.py ask "Que pacientes tienen diarrea?" --top-k 5
python rag_veterinaria.py ask "Que dieta indicada tiene Ivi?" --paciente Ivi --seccion dieta_indicada
python rag_veterinaria.py ask "Que gatos tienen indicado calcio?" --especie gato --seccion suplementos_indicados
```

## 4. Hacer consultas con Ollama

Primero hay que tener Ollama corriendo y un modelo descargado. Ejemplo:

```powershell
ollama pull llama3.2
python rag_veterinaria.py ask "Resumen de pacientes con diarrea" --ollama-model llama3.2
```

El modelo recibe solo el contexto recuperado y tiene instrucciones de no inventar
diagnosticos ni indicaciones.

## 5. Arquitectura implementada

Cada ficha se transforma en varios documentos:

- `perfil_paciente`
- `historia_clinica`
- `dieta_actual`
- `dieta_indicada`
- `suplementos_indicados`
- `etiqueta_balanceado`
- `preparacion`
- `indicaciones_generales`
- `seguimientos`

Cada documento conserva metadata: paciente, tutor, especie, raza, sexo, peso,
archivo origen y seccion. Eso permite filtrar por especie, paciente o tipo de
informacion.

## 6. Siguiente mejora recomendada

Esta version funciona sin instalar nada extra porque usa `scikit-learn`.
Para una entrega mas fuerte, el siguiente paso seria reemplazar o complementar
TF-IDF con embeddings multilingues, por ejemplo:

```powershell
python -m pip install sentence-transformers faiss-cpu
```

Modelo sugerido:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Tambien se puede sumar una interfaz Streamlit para presentar el TP.
