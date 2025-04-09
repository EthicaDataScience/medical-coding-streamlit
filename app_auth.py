import streamlit as st
import pandas as pd
import pickle
import openai
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# 🔐 Utilisateurs autorisés (email: Password)
AUTHORIZED_USERS = {
    "skaba@ethicacro.com": "Kaba19",
    "data.management@ethicacro.com": "z8K!ef92mT",
    "data.science@ethicacro.com": "T2b*Qp57xV"
}

# ⚙️ Init état de session
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

# 🔓 Bouton de déconnexion
if st.session_state.authenticated:
    with st.sidebar:
        st.write(f"👤 Logged in as **{st.session_state.user_email}**")
        if st.button("🔓 Log out"):
            st.session_state.authenticated = False
            st.session_state.user_email = None
            st.rerun()

# 🔐 Formulaire de connexion
if not st.session_state.authenticated:
    st.title("🔐 Authentication required")
    with st.form("login_form"):
        email = st.text_input("📧 Email")
        password = st.text_input("🔑 Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if email in AUTHORIZED_USERS and AUTHORIZED_USERS[email] == password:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.success(f"✅ Bienvenue, {email} !")
            st.rerun()
        else:
            st.error("❌ Email ou Password incorrect.")
    st.stop()

# ✅ Si connecté, suite de l’application
st.title("🔎 Automatic Medical Coding (MedDRA)")

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

# 📤 Upload du fichier AE
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
            combined_row.update(best_match["row"])
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

    # 📁 Génération du fichier Excel
    output = BytesIO()
    df_result.to_excel(output, index=False)
    output.seek(0)

    # 📥 Bouton de téléchargement
    st.download_button(
        label="📥 Download Excel results",
        data=output,
        file_name="AE_CODING.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
