import os
import time
from glob import glob
from tqdm import tqdm
import argparse
import warnings

from common.misc_utils import LOCAL_CACHE_DIR, get_logger, get_txt_tab_filenames, get_model_endpoints
from common.db_utils import MilvusVectorStore
from ingest.doc_utils import extract_document_data, hierarchical_chunk_with_token_split, create_chunk_documents

logger = get_logger("ingest")

def reset_db():
    emb_model_dict, llm_model_dict = get_model_endpoints()
    vector_store = MilvusVectorStore(emb_name=emb_model_dict["emb_model"], llm_name=llm_model_dict["llm_model"])
    collection_name = vector_store._generate_collection_name()

    vector_store.reset_collection()
    logger.info(f"Cleaned Vector DB: {collection_name}")

def ingest(directory_path, include_meta_info_in_main_text):
    emb_model_dict, llm_model_dict = get_model_endpoints()
    # Initialize/reset the database before processing any files
    vector_store = MilvusVectorStore(emb_name=emb_model_dict["emb_model"], llm_name=llm_model_dict["llm_model"])
    collection_name = vector_store._generate_collection_name()

    # Process each document in the directory
    allowed_file_types = ['pdf']
    file_paths = []
    for f_type in allowed_file_types:
        file_paths.extend(glob(f'{directory_path}/**/*.{f_type}', recursive=True))
    files_being_processed = '\n'.join(f for f in file_paths)
    logger.info(f"Processing the following files: {files_being_processed}")

    out_path = os.path.join(LOCAL_CACHE_DIR, f'{collection_name}_cache')
    os.makedirs(out_path, exist_ok=True)

    start_time = time.time()
    extract_document_data(
        file_paths, out_path, llm_model_dict['llm_model'], llm_model_dict['llm_endpoint'])

    original_filenames, input_txt_files, input_tab_files = get_txt_tab_filenames(file_paths, out_path)
    output_chunk_files = [f.replace('_clean_text.json', '_clean_chunk.json') for f in input_txt_files]
    hierarchical_chunk_with_token_split(
        input_txt_files, output_chunk_files, llm_model_dict["llm_endpoint"],
        max_tokens=emb_model_dict['max_tokens'] - 100 if include_meta_info_in_main_text else emb_model_dict['max_tokens']
    )
    combined_filtered_chunks = []
    for in_txt_f, in_tab_f, orig_fn, out_txt_f in tqdm(zip(
        input_txt_files, input_tab_files, original_filenames, output_chunk_files
    ), total=len(input_txt_files), desc='Creating Chunks'):
        # Combine all chunks (text, image summaries, table summaries)
        filtered_chunks, stats = create_chunk_documents(
            in_txt_f, in_tab_f, orig_fn, include_meta_info_in_main_text, collection_name)
        combined_filtered_chunks.extend(filtered_chunks)

    # Insert data into Milvus
    vector_store.insert_chunks(
        emb_model=emb_model_dict['emb_model'],
        emb_endpoint=emb_model_dict['emb_endpoint'],
        max_tokens=emb_model_dict['max_tokens'],
        chunks=combined_filtered_chunks
    )

    logger.info(f"Inserted {len(combined_filtered_chunks)} chunks to the vector DB: {collection_name}")

    # Log time taken for the file
    end_time = time.time()  # End the timer for the current file
    file_processing_time = end_time - start_time
    logger.info(f"Time taken to ingest data in vector DB is: {file_processing_time:.2f} seconds")

    logger.info(f"Vector DB ({collection_name}) creation completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Data Ingestion CLI", formatter_class=argparse.RawTextHelpFormatter)
    common_parser = parser.add_subparsers(dest="command", required=True)

    ingest_parser = common_parser.add_parser("ingest", help="Ingest the DOCs", description="Ingest the DOCs into Milvus after all the processing\n", formatter_class=argparse.RawTextHelpFormatter)
    ingest_parser.add_argument("--path", type=str, default="/var/docs", help="Path to the documents that needs to be ingested into the RAG")
    ingest_parser.add_argument("--include-meta", action="store_true", help="Include meta info while ingesting the docs")

    common_parser.add_parser("clean-db", help="Clean the DB", description="Clean the Milvus DB\n", formatter_class=argparse.RawTextHelpFormatter)

    command_args = parser.parse_args()
    if command_args.command == "ingest":
        ingest(command_args.path, command_args.include_meta)
    elif command_args.command == "clean-db":
        reset_db()

if __name__ == "__main__":
    main()
