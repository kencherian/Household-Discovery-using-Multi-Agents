import streamlit as st
import os
import logging
import pandas as pd
import re
import io
import hashlib # --- SECURITY ADDITION: Level 2 (Integrity) ---
import tempfile
from utils.parser import Parsers
from utils.chunking import Chunking
from rag.embeddings import Embeddings
from rag.vectorstore import VectorStore
from rag.llm import LLM
from multiagents.graph import build_graph
from langchain.retrievers.multi_query import MultiQueryRetriever
from multiagents import prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SECURITY CONFIGURATION (Level 1) ---
ACCESS_PASSWORD = "MySecurePassword123" 
# ----------------------------------------

# --- SECURITY UTILS (Level 2) ---
def get_file_hash(file_path):
    """Calculates the SHA-256 hash of a file for integrity verification."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        return f"Error: {e}"
# --------------------------------

def llm_clean_dataframe(df, llm, batch_size=50):
    cleaned_batches = []
    num_rows = len(df)
    progress_bar = st.progress(0, text="Cleaning data with LLM...")
    llm_class = llm.__class__.__name__.lower()
    for i, start in enumerate(range(0, num_rows, batch_size)):
        batch = df.iloc[start:start+batch_size]
        table_str = batch.to_markdown(index=False)
        prompt = prompts.data_cleaning_prompt.format(content=table_str)
        # Explicitly check for Gemini/Google model
        if 'google' in llm_class or 'gemini' in llm_class:
            response = llm.invoke(prompt)
        else:
            response = llm.invoke([{"role": "system", "content": prompt}])
        match = re.search(r'(\|.*\|\n)(\|[-| ]+\|\n)((\|.*\|\n)+)', response.content if hasattr(response, 'content') else str(response))
        if match:
            table_md = match.group(0)
            try:
                cleaned_df = pd.read_csv(io.StringIO(table_md.replace('|', ',')), skipinitialspace=True)
                cleaned_df.columns = [c.strip() for c in cleaned_df.columns]
                cleaned_batches.append(cleaned_df)
            except Exception as e:
                st.warning(f"Failed to parse cleaned table for batch {i+1}: {e}. Using original data for this batch.")
                cleaned_batches.append(batch)
        else:
            st.warning(f"LLM did not return a valid markdown table for batch {i+1}. Using original data for this batch.")
            cleaned_batches.append(batch)
        progress_bar.progress(min((start+batch_size)/num_rows, 1.0), text=f"Cleaning data with LLM... Batch {i+1}/{(num_rows-1)//batch_size+1}")
    progress_bar.empty()
    return pd.concat(cleaned_batches, ignore_index=True)

def hybrid_clean_dataframe(df, llm, batch_size=50):
    # Helper functions
    def looks_like_name(val):
        return bool(re.match(r"^[A-Za-z .,'-]+$", str(val).strip())) and len(str(val).split()) <= 4
    def looks_like_address(val):
        return bool(re.search(r"\d+|Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Blvd|Boulevard|Court|Ct|Apt|Unit|Suite|Ste|PO Box", str(val), re.IGNORECASE))
    # Split into easy and ambiguous
    easy_rows = []
    ambiguous_rows = []
    for idx, row in df.iterrows():
        name, address = str(row.get('Name', '')), str(row.get('Address', ''))
        if (looks_like_name(name) and looks_like_address(address) and name and address):
            easy_rows.append(row)
        else:
            ambiguous_rows.append(row)
    easy_df = pd.DataFrame(easy_rows)
    ambiguous_df = pd.DataFrame(ambiguous_rows)
    # Clean ambiguous rows with LLM
    if not ambiguous_df.empty:
        cleaned_ambiguous = llm_clean_dataframe(ambiguous_df, llm, batch_size=batch_size)
        cleaned_df = pd.concat([easy_df, cleaned_ambiguous], ignore_index=True)
    else:
        cleaned_df = easy_df
    return cleaned_df

def rag_pipeline(available_files, embedding_model, vector_store, llm_model, use_llm_cleaning=True, save_cleaned=False, load_cleaned=False, cleaning_llm=None, use_hybrid_cleaning=False):
    combined_df = pd.DataFrame()
    for selected_file in available_files:
        if selected_file.endswith('.csv'):
            combined_df = pd.concat([combined_df, Parsers.csv_parser([selected_file])])
        elif selected_file.endswith('.xlsx'):
            combined_df = pd.concat([combined_df, Parsers.xlsx_parser([selected_file])])
        else:
            raise ValueError("Unsupported file type. Please select CSV or XLSX files.")

    if combined_df.empty:
        raise ValueError("No data extracted from selected files.")

    cleaned_path = os.path.join("output", "cleaned_data.csv")

    if load_cleaned and os.path.exists(cleaned_path):
        cleaned_df = pd.read_csv(cleaned_path)
        st.success("Loaded cleaned data from cache.")
    elif use_llm_cleaning:
        if use_hybrid_cleaning:
            cleaned_df = hybrid_clean_dataframe(combined_df, cleaning_llm)
        else:
            cleaned_df = llm_clean_dataframe(combined_df, cleaning_llm)
        if save_cleaned:
            cleaned_df.to_csv(cleaned_path, index=False)
            st.success(f"Saved cleaned data to {cleaned_path}")
    else:
        cleaned_df = combined_df
        st.info("LLM cleaning skipped. Using raw data.")

    st.markdown('### Cleaned Data')
    st.dataframe(cleaned_df)

    # Chunking and embedding
    text_chunks = Chunking.get_chunks(cleaned_df.to_string(index=False))
    embeddings = Embeddings.get_embeddings(embedding_model)
    vectorstore = VectorStore.vectorization(vector_store, text_chunks, embeddings)

    # Multi-query retriever integration
    llm = LLM.get_llm(llm_model) if isinstance(llm_model, str) else llm_model
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

    # Build graph using retriever
    graph = build_graph(llm, embedding_model, vector_store, text_chunks)

    initial_state = {"task": "", "result": "", "truth_set_path": None}
    thread = {"configurable": {"thread_id": "entity_resolution"}}

    final_results = {}
    final_state = None
    for state in graph.stream(initial_state, thread):
        agent_name = list(state.keys())[0]
        agent_result = state[agent_name]

        if agent_name in ["direct_agent", "indirect_agent", "household_agent", "household_moves_agent"]:
            if isinstance(agent_result, dict) and agent_result.get("result"):
                st.subheader(f"{agent_name.replace('_', ' ').title()} Result:")
                st.markdown(agent_result["result"])
                final_results[agent_name] = agent_result["result"]
            else:
                st.error(f"Unexpected agent response: {agent_result}")
        final_state = state

    return final_results

def test_llm_connection(llm):
    try:
        llm_class = llm.__class__.__name__.lower()
        prompt = "Hello, world!"
        if 'google' in llm_class or 'gemini' in llm_class:
            response = llm.invoke(prompt)
        else:
            response = llm.invoke([{"role": "system", "content": prompt}])
        st.success(f"LLM test successful! Response: {response.content if hasattr(response, 'content') else str(response)}")
    except Exception as e:
        st.error(f"LLM test failed: {e}")

def main():
    st.set_page_config(layout="wide")
    st.title("Household Discovery with Multi-Agent RAG System")

    # --- SECURITY LEVEL 1: ACCESS CONTROL ---
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("### 🔒 Login Required")
        password_input = st.text_input("Enter Access Password:", type="password")
        if st.button("Login"):
            if password_input == ACCESS_PASSWORD:
                st.session_state.authenticated = True
                st.success("Authenticated successfully!")
                st.rerun()  # Rerun the app to show the hidden content
            else:
                st.error("Incorrect password. Access Denied.")
        st.stop() # Stop execution here so the rest of the app doesn't load
    # ----------------------------------------

    input_folder = "input"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # List available files in input/ directory
    all_files = [f for f in os.listdir(input_folder) if f.endswith(('.csv', '.xlsx')) and not f.startswith('~$')]
    if not all_files:
        st.warning("No CSV or XLSX files found in input/ directory.")
        return

    selected_files = st.multiselect("Select dataset(s) to process:", all_files, default=all_files[:1])
    
    # --- SECURITY LEVEL 2: INTEGRITY CHECKS ---
    if selected_files:
        st.sidebar.markdown("### 🛡️ Data Integrity Audit")
        st.sidebar.info("SHA-256 Signatures of selected files:")
        for f_name in selected_files:
            f_path = os.path.join(input_folder, f_name)
            f_hash = get_file_hash(f_path)
            # Display first 8 chars for brevity, full hash on hover
            st.sidebar.text(f"{f_name}:")
            st.sidebar.code(f"{f_hash[:10]}...", language="text")
    # ------------------------------------------

    if not selected_files:
        st.warning("Please select at least one dataset to process.")
        return
    available_files = [os.path.join(input_folder, f) for f in selected_files]

    embedding_model = st.selectbox("Select Embedding Model:", Embeddings.get_available_embeddings())
    vector_store = st.selectbox("Select VectorStore:", VectorStore.get_available_vectorstores())
    llm_model = st.selectbox("Select LLM Model:", LLM.get_available_llm())

    use_llm_cleaning = st.checkbox("Use LLM Data Cleaning", value=True)
    use_hybrid_cleaning = st.checkbox("Use Hybrid Cleaning (fast + LLM)", value=False)
    save_cleaned = st.checkbox("Save cleaned data for reuse", value=False)
    load_cleaned = st.checkbox("Load cleaned data if available", value=False)

    cleaning_llm_model = None
    if use_llm_cleaning:
        cleaning_llm_model = st.selectbox("Select LLM for Data Cleaning:", LLM.get_available_llm(), index=LLM.get_available_llm().index('mistral') if 'mistral' in LLM.get_available_llm() else 0)
        cleaning_llm = LLM.get_llm(cleaning_llm_model)
    else:
        cleaning_llm = None

    if st.button("Test LLM Connection"):
        if use_llm_cleaning and cleaning_llm:
            test_llm_connection(cleaning_llm)
        else:
            test_llm_connection(LLM.get_llm(llm_model))

    if st.button("Run Pipeline"):
        try:
            results = rag_pipeline(available_files, embedding_model, vector_store, llm_model, use_llm_cleaning, save_cleaned, load_cleaned, cleaning_llm, use_hybrid_cleaning)

            output_file_path = os.path.join(output_folder, "output_summary.md")
            with open(output_file_path, "w") as f:
                f.write("# Summary\n\n")
                
                # --- SECURITY ADDITION: Add Integrity Log to Output ---
                f.write("## 🛡️ Data Integrity Log\n")
                f.write("| File Name | SHA-256 Hash |\n")
                f.write("|---|---|\n")
                for f_name in selected_files:
                    f_hash = get_file_hash(os.path.join(input_folder, f_name))
                    f.write(f"| {f_name} | `{f_hash}` |\n")
                f.write("\n---\n\n")
                # -----------------------------------------------------

                for agent_name, result_markdown in results.items():
                    f.write(f"### {agent_name.replace('_', ' ').title()} Result:\n\n")
                    f.write(result_markdown)
                    f.write("\n\n")

            st.success(f"Output summary saved to `{output_file_path}`")

            with open(output_file_path, "r") as f:
                st.markdown(f.read(), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error running pipeline: {e}")
            st.warning("If you are using Gemini or Mistral, check your API key, quota, and model name. For Gemini, ensure the prompt is not too large.")
            logger.exception(e)

if __name__ == "__main__":
    main()