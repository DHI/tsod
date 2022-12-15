from pathlib import Path
import streamlit as st
import mikeio

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file:
    data = uploaded_file.getvalue()
    TEMP_FILE = Path("test.dfs0")

    with TEMP_FILE.open("wb") as f:
        f.write(data)

    test = mikeio.read("test.dfs0")

    st.write(test.to_dataframe())
    TEMP_FILE.unlink()
