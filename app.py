import streamlit as st
import pandas as pd
import pickle
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Liste des utilisateurs autorisés
AUTHORIZED_EMAILS = {
    "skaba@ethicacro.com",
    "data.management@ethicacro.com",
    "data.science@ethicacro.com"
}

# Authentification Streamlit Cloud
if not hasattr(st, "experimental_user") or "email" not in st.experimental_user:
    st.stop()

user_email = st.experimental_user["email"]

if user_email not in AUTHORIZED_EMAILS:
    st.error("⛔ Accès refusé. Tu n'es pas autorisé à utiliser cette application.")
    st.stop()

st.sidebar.success(f"🔐 Connecté en tant que : {user_email}")

# Chargement des embeddings MedDRA
with open("data/meddra_embeddings.pkl", "rb") as f:
    meddra_embeddings = pickle.load(f)

# Configuration OpenAI
openai.api_key = st.secrets["openai_api_key"]

# Fonction d'embedding OpenAI
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=[text.replace("\n", " ")]
    )
    return response.data[0].embedding

# Interface principale
st.title("🔎 Codage Médical Automatique (MedDRA + OpenAI)")
uploaded_file = st.file_uploader("📤 Dépose ton fichier AE (.txt tabulé)", type=["txt"])

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
    st.download_button(
        "📥 Télécharger les résultats Excel",
        df_result.to_excel(index=False),
        file_name="AE_CODING.xlsx"
    )
