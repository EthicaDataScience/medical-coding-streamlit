import streamlit as st
import pandas as pd
import pickle
import openai
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# 📦 Chargement des embeddings MedDRA
with open("data/meddra_embeddings.pkl", "rb") as f:
    meddra_embeddings = pickle.load(f)

# 🔑 Clé OpenAI depuis secrets.toml
openai.api_key = st.secrets["openai_api_key"]

# 🔍 Fonction d'embedding
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=[text.replace("\n", " ")]
    )
    return response.data[0].embedding

# 🧠 Interface
st.title("🔎 Automatic Medical Coding (MedDRA)")
uploaded_file = st.file_uploader("📤 Upload your AE file (.txt tabulated)", type=["txt"])

if uploaded_file:
    df_ae = pd.read_csv(uploaded_file, sep="\t")
    results = []

    for _, row in df_ae.iterrows():
        ae_term = row.get("AETERM", "")
        if pd.isna(ae_term) or ae_term.strip() == "":
            continue

        ae_embedding = get_embedding(ae_term)
        best_sim = 0
        best_match = None

        for entry in meddra_embeddings:
            sim = cosine_similarity([ae_embedding], [entry["embedding"]])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_match = entry

        combined_row = row.to_dict()
        if best_match:
            combined_row.update(best_match["row"])  # Colonnes MedDRA
            combined_row["matched_term"] = best_match["term"]
            combined_row["matched_source"] = best_match["source"]
            combined_row["similarity"] = round(best_sim, 4)
        else:
            combined_row.update({
                "matched_term": None,
                "matched_source": None,
                "similarity": 0.0
            })

        results.append(combined_row)

    df_result = pd.DataFrame(results)
    st.success("✅ Codage terminé avec succès")
    st.dataframe(df_result)

    # 📁 Création d'un fichier Excel dans un buffer
    output = BytesIO()
    df_result.to_excel(output, index=False)
    output.seek(0)

    # 📥 Téléchargement
    st.download_button(
        label="📥 Download Excel results",
        data=output,
        file_name="AE_CODING.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
