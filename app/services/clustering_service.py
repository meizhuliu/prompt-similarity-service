import numpy as np


def cluster_duplicates(store, threshold=0.9):
    """
    Groups similar prompts into duplicate clusters using cosine similarity.
    """

    # Get all prompt IDs from the vector store
    ids = list(store.vectors.keys())

    # Track visited prompts to avoid duplicate clustering
    visited = set()

    # Final list of clusters (each cluster = similar prompts)
    clusters = []

    # Iterate through all prompts
    for i, a in enumerate(ids):

        # Skip if already assigned to a cluster
        if a in visited:
            continue

        # Start a new cluster with current prompt
        cluster = [a]
        visited.add(a)

        # Compare with remaining prompts
        for b in ids[i+1:]:

            # Only check unvisited prompts
            if b not in visited:

                # Compute similarity (dot product assumes normalized embeddings)
                sim = float(np.dot(store.vectors[a], store.vectors[b]))

                # Add to cluster if similarity exceeds threshold
                if sim > threshold:
                    cluster.append(b)
                    visited.add(b)

        # Only keep meaningful clusters (duplicates > 1 item)
        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters
