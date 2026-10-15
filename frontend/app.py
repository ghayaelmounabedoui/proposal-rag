
import json
import os

import requests
import streamlit as st


API_URL = os.getenv(
    "PROPOSAL_RAG_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

API_KEY = os.getenv("PROPOSAL_RAG_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

TIMEOUT = 180

st.set_page_config(
    page_title="Proposal RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Proposal RAG")
st.caption(
    "Assistant intelligent pour répondre aux appels d'offres"
)


def api_request(method, endpoint, raw=False, **kwargs):
    try:
        response = requests.request(
            method,
            f"{API_URL}{endpoint}",
            timeout=TIMEOUT,
            headers=HEADERS,
            **kwargs,
        )
        response.raise_for_status()
        return response.content if raw else response.json()

    except requests.exceptions.HTTPError as exc:
        st.error(f"Erreur API : {exc.response.text}")
        return None

    except requests.exceptions.RequestException as exc:
        st.error(f"Connexion API impossible : {exc}")
        return None


def show_sources(sources):
    """Afficher les passages utilisés pour générer la réponse."""

    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            location = source["document"]
            if source.get("section"):
                location += f" - {source['section']}"
            if source.get("page"):
                location += f" (p. {source['page']})"

            st.markdown(f"**{source['id']}** {location}")
            st.caption(source["excerpt"])


# --------------------------------------------------
# Vérification de l'API
# --------------------------------------------------

try:
    health = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    api_online = health.ok

except requests.exceptions.RequestException:
    api_online = False


# --------------------------------------------------
# Navigation
# --------------------------------------------------

with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Choisir une page",
        [
            "Assistant RAG",
            "Documents",
            "Analyse RFP",
        ],
    )

    st.divider()

    if api_online:
        st.success("API connectée")
    else:
        st.error("API indisponible")
        st.caption(
            "Démarre FastAPI sur le port 8000."
        )


# --------------------------------------------------
# Page Assistant RAG
# --------------------------------------------------

if page == "Assistant RAG":

    st.subheader("Assistant appels d'offres")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_sources(message.get("sources", []))

    question = st.chat_input(
        "Pose une question sur tes documents..."
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Recherche des documents et génération..."
            ):
                result = api_request(
                    "POST",
                    "/ask",
                    json={
                        "question": question,
                        "top_k": 3,
                    },
                )

            if result:
                answer = result["answer"]
                sources = result.get("sources", [])
                st.markdown(answer)
                show_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Page Documents
# --------------------------------------------------

elif page == "Documents":

    st.subheader("Gestion des documents")

    st.markdown("### Importer un document")

    uploaded_file = st.file_uploader(
        "Sélectionne un fichier",
        type=["pdf", "docx", "txt", "md"],
    )

    if uploaded_file is not None:

        st.write(
            f"Fichier sélectionné : {uploaded_file.name}"
        )

        if st.button(
            "Indexer le document",
            type="primary",
        ):
            with st.spinner(
                "Importation et indexation en cours..."
            ):
                result = api_request(
                    "POST",
                    "/documents/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                        )
                    },
                )

            if result:
                st.success(
                    "Document indexé avec succès !"
                )
                st.json(result)

    st.divider()
    st.markdown("### Documents indexés")

    if st.button("Actualiser la liste"):
        st.rerun()

    result = api_request(
        "GET",
        "/documents",
    )

    if result:
        documents = result.get("documents", [])

        st.write(
            f"Nombre de documents : {len(documents)}"
        )

        if not documents:
            st.info("Aucun document indexé.")

        for document in documents:
            filename = document["filename"]
            chunks = document["chunks"]

            with st.container(border=True):
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"**{filename}**")
                    st.caption(
                        f"{chunks} chunks indexés"
                    )

                with col2:
                    if st.button(
                        "Supprimer",
                        key=f"delete_{filename}",
                    ):
                        delete_result = api_request(
                            "DELETE",
                            "/documents/"
                            + requests.utils.quote(
                                filename,
                                safe="",
                            ),
                        )

                        if delete_result:
                            st.success(
                                "Document supprimé."
                            )
                            st.rerun()


# --------------------------------------------------
# Page Analyse RFP
# --------------------------------------------------

elif page == "Analyse RFP":

    st.subheader(
        "Analyse automatique d'un appel d'offres"
    )

    st.write(
        "Importe un appel d'offres pour extraire "
        "ses exigences, analyser la conformité et "
        "générer une proposition technique."
    )

    # ----------------------------------------------
    # 1. Importer et analyser le RFP
    # ----------------------------------------------

    rfp_file = st.file_uploader(
        "Sélectionne un appel d'offres",
        type=["pdf", "docx", "txt", "md"],
        key="rfp_file",
    )

    if rfp_file is not None:
        if st.button(
            "Analyser l'appel d'offres",
            type="primary",
        ):
            with st.spinner(
                "Extraction des exigences..."
            ):
                result = api_request(
                    "POST",
                    "/rfp/analyze",
                    files={
                        "file": (
                            rfp_file.name,
                            rfp_file.getvalue(),
                        )
                    },
                )

            if result:
                st.session_state["rfp_analysis"] = result

                # Réinitialiser les résultats précédents
                st.session_state.pop(
                    "compliance_matrix",
                    None,
                )
                st.session_state.pop(
                    "technical_proposal",
                    None,
                )

    analysis = st.session_state.get(
        "rfp_analysis"
    )

    if analysis:

        st.success(
            f"{analysis['total_requirements']} "
            "exigences extraites."
        )

        st.info(
            f"Document analysé en "
            f"{analysis.get('total_batches', 1)} lot(s)."
        )

        requirements = analysis["requirements"]

        st.dataframe(
            requirements,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Télécharger les exigences (JSON)",
            data=json.dumps(
                analysis,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="rfp_requirements.json",
            mime="application/json",
        )

        # ------------------------------------------
        # 2. Matrice de conformité
        # ------------------------------------------

        st.divider()
        st.subheader("Matrice de conformité")

        if st.button(
            "Générer la matrice de conformité",
            type="primary",
        ):
            with st.spinner(
                "Recherche des preuves et génération..."
            ):
                result = api_request(
                    "POST",
                    "/rfp/compliance",
                    json={
                        "requirements": requirements,
                    },
                )

            if result:
                st.session_state[
                    "compliance_matrix"
                ] = result

                # Une nouvelle matrice invalide
                # l'ancienne proposition.
                st.session_state.pop(
                    "technical_proposal",
                    None,
                )

        matrix = st.session_state.get(
            "compliance_matrix"
        )

        if matrix:

            st.dataframe(
                matrix["results"],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Télécharger la matrice (JSON)",
                data=json.dumps(
                    matrix,
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name="compliance_matrix.json",
                mime="application/json",
            )

            # --------------------------------------
            # 3. Génération de proposition technique
            # --------------------------------------

            st.divider()
            st.subheader("Proposition technique")

            st.write(
                "Génère une proposition technique "
                "structurée à partir des exigences "
                "et des résultats de conformité."
            )

            if st.button(
                "Générer la proposition technique",
                type="primary",
            ):
                with st.spinner(
                    "Rédaction de la proposition..."
                ):
                    result = api_request(
                        "POST",
                        "/rfp/proposal",
                        json={
                            "results": matrix["results"],
                        },
                    )

                if result:
                    st.session_state[
                        "technical_proposal"
                    ] = result["proposal"]
                    st.session_state.pop("proposal_docx", None)

            # --------------------------------------
            # 4. Affichage de la proposition
            # --------------------------------------

            proposal = st.session_state.get(
                "technical_proposal"
            )

            if proposal:

                st.success(
                    "Proposition technique générée !"
                )

                st.markdown(proposal)

                col_md, col_docx = st.columns(2)

                with col_md:
                    st.download_button(
                        "Télécharger (Markdown)",
                        data=proposal,
                        file_name="technical_proposal.md",
                        mime="text/markdown",
                    )

                with col_docx:
                    # Conversion faite une seule fois par proposition.
                    if "proposal_docx" not in st.session_state:
                        st.session_state["proposal_docx"] = api_request(
                            "POST",
                            "/rfp/proposal/export",
                            raw=True,
                            json={"proposal": proposal},
                        )

                    docx_bytes = st.session_state["proposal_docx"]

                    if docx_bytes:
                        st.download_button(
                            "Télécharger (Word)",
                            data=docx_bytes,
                            file_name="technical_proposal.docx",
                            mime=(
                                "application/vnd.openxmlformats-"
                                "officedocument.wordprocessingml.document"
                            ),
                        )
