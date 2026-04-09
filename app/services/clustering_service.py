import numpy as np

def cluster_duplicates(store, threshold=0.9):
    ids = list(store.vectors.keys())
    visited = set()
    clusters = []

    for i, a in enumerate(ids):
        if a in visited:
            continue

        cluster = [a]
        visited.add(a)

        for b in ids[i+1:]:
            if b not in visited:
                sim = float(np.dot(store.vectors[a], store.vectors[b]))
                if sim > threshold:
                    cluster.append(b)
                    visited.add(b)

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters
