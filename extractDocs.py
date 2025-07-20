import mysql.connector
from langchain_core.documents import Document 

def fetch_pdf_chunks():
    chunk_counter = 0  # Initialize a counter for chunks
    all_documents = [] # To store Document objects

    conn = None # Initialize conn to None for proper error handling
    cursor = None # Initialize cursor to None
    
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Password123@",
            database="content"
        )
        cursor = conn.cursor(buffered=True)  # Use buffered cursor to avoid unread results

        cursor.execute("SELECT id, content FROM pdf_chunks ORDER BY id") # Assuming 'id' is a primary key for ordering
        
        # Fetch all results at once to avoid unread result issues
        rows = cursor.fetchall()
        
        # Iterate through the fetched rows to create Document objects and count
        for row in rows:
            chunk_id_from_db = row[0] # Assuming the first column is the ID
            content = row[1]          # Assuming the second column is the content
            
            chunk_counter += 1
            
            # Create a Document object with metadata including the chunk_id
            doc = Document(
                page_content=content,
                metadata={"source": "mysql_db", "db_id": chunk_id_from_db, "chunk_order": chunk_counter}
            )
            all_documents.append(doc)
            
            print(f"  Processed chunk {chunk_counter} (DB ID: {chunk_id_from_db}, Content Snippet: '{content[:50]}...')") # Print progress
            
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        # Depending on your application, you might want to re-raise the exception
        # or return an empty list.
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []
    finally:
        # Proper cleanup order: cursor first, then connection
        if cursor is not None:
            try:
                cursor.close()
            except Exception as e:
                print(f"Error closing cursor: {e}")
        
        if conn is not None:
            try:
                if conn.is_connected():
                    conn.close()
                    print("Database connection closed.")
            except Exception as e:
                print(f"Error closing connection: {e}")
            
    print(f"Step 2: Finished fetching PDF chunks. Found {len(all_documents)} total chunks.")
    return all_documents


if __name__ == "__main__":
    fetch_pdf_chunks()