import numpy as np
from face_engine import get_connection

EMBEDDING_FILE = "pins_Sarah Wayne Callies.npy"


def insert_single_embedding():

    embedding = np.load(EMBEDDING_FILE)

    print("Original shape:", embedding.shape)

    # Ensure correct shape
    embedding = embedding.flatten()

    print("Flattened shape:", embedding.shape)

    # Convert to float (important)
    embedding_list = [float(x) for x in embedding]

    print("Length:", len(embedding_list))
    print("First 5 values:", embedding_list[:5])

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO cctv_embeddings
        (video_name, camera_name, frame_number, embedding, location)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        "Sarah_Wayne_Callies",
        "Camera 01",
        0,
        embedding_list,
        "Cheemeni"
    ))

    conn.commit()
    cur.close()
    conn.close()

    print("Embedding inserted successfully!")


if __name__ == "__main__":
    insert_single_embedding()