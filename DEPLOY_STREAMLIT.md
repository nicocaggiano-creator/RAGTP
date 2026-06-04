# Despliegue en Streamlit Community Cloud

Objetivo: que tus companeros puedan usar el RAG desde un link, sin instalar nada
en sus computadoras.

## Archivos necesarios

Subi estos archivos/carpetas a un repositorio de GitHub:

- `app_streamlit_rag.py`
- `rag_veterinaria.py`
- `rag_veterinaria_index/`
- `requirements.txt`
- `README_RAG_VETERINARIA.md`

La app usa el indice ya construido en `rag_veterinaria_index/`, por eso no hace
falta que Streamlit Cloud vuelva a procesar `resultados_completos.json`.

## Pasos

1. Crear un repositorio en GitHub, por ejemplo `rag-veterinaria`.
2. Subir los archivos listados arriba.
3. Entrar a `https://share.streamlit.io`.
4. Elegir `Create app`.
5. Seleccionar el repositorio y la rama.
6. En `Main file path`, poner:

```text
app_streamlit_rag.py
```

7. Deploy.

Streamlit va a instalar las dependencias desde `requirements.txt` y generar una
URL del estilo:

```text
https://tu-usuario-rag-veterinaria-app-streamlit-rag.streamlit.app
```

Esa URL es la que podes compartir con tus companeros.

## Privacidad

Si las fichas tienen datos sensibles, lo ideal es usar un repositorio privado y
configurar la app como privada, o anonimizar tutores/telefonos antes de subirla.
Para una demo publica, conviene quitar telefonos y datos identificatorios.
