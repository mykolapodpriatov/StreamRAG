import streamlit as st

def main():
    st.set_page_config(page_title="StreamRAG Dashboard", layout="wide")

    st.title("StreamRAG Dashboard")
    st.write("Monitoring streaming data ingestion and RAG system.")

    col1, col2 = st.columns(2)
    with col1:
        # TODO: Fetch real document count from Qdrant
        st.metric(label="Total Documents", value="1,200", delta="12")
    with col2:
        # TODO: Fetch real active streams from Redis
        st.metric(label="Active Streams", value="4")

    st.subheader("Query the Knowledge Base")
    query = st.text_input("Enter your query:")
    if query:
        st.write(f"Searching for: {query}...")
        # Placeholder for RAG pipeline
        st.info("Answer will appear here.")

if __name__ == "__main__":
    main()
