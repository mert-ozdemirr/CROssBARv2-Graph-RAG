import networkx as nx
import re
import pickle
import os
from typing import List, Tuple
import numpy as np
import chromadb
from neo4j import GraphDatabase
from itertools import product
from fastapi import Request


import llm_analysis
import voyage_embedder
import neo4j_drive

def get_bm25_collection(request: Request):
    global all_bm25s_g
    all_bm25s_g = request.app.state.all_bm25s


def graph_retriever(user_query, candidate_count, graph_match_count, request: Request):
    # make params global
    global candidate_node_count_determiner
    candidate_node_count_determiner = candidate_count
    global graph_match_count_determiner
    graph_match_count_determiner = graph_match_count

    #
    get_bm25_collection(request)

    # subgraph extraction
    llm_output = llm_analysis.llm_subgraph_pattern_extractor(user_query)
    llm_output_triples = llm_analysis.get_triples_str(llm_output)
    triples_parsed = llm_analysis.parse_triples(llm_output_triples)

    # create and form graphs
    seperate_graphs = graph_constructor(triples_parsed)
    for seperate_graph in seperate_graphs:
        graph_type_seperator(seperate_graph)
    
    #
    global relationship_mappings
    relationship_mapping_file_path = os.path.join(os.path.dirname(__file__), "local_files", "relationship_mappings.pkl") 
    with open(relationship_mapping_file_path, "rb") as f:
        relationship_mappings = pickle.load(f)

    # get chromadb collection names
    global chromadb_names
    chromadb_names = chromadb_storage_name_pull()

    # neo4j set
    global driver_n4j
    driver_n4j = neo4j_drive.neo4j_driver_set()

    # run retrieval pipeline for each separate graph
    generation_data_prompt = []
    if len(seperate_graphs) == 0:
        generation_data_prompt += no_graph_search(user_query)
    for seperate_graph in seperate_graphs:
        print(seperate_graph.edges(data=True))
        generation_data_prompt += process_graph(seperate_graph)

    # post-process the raw retrieval data
    deduplicated_generation_data = deduplicate_result(generation_data_prompt)

    



    return deduplicated_generation_data








def graph_constructor(parsed_triples):
    if not parsed_triples:
        return []
    
    G = nx.Graph()
    for subj, rel, obj in parsed_triples:
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, label=rel)

    # Split G into weakly connected components (disconnected subgraphs)
    intermediate_list = [
        G.subgraph(component).copy() 
        for component in nx.connected_components(G)
    ]
    return intermediate_list


def graph_type_seperator(G): # updates_graphs
    nodes_list = G.nodes
    graph_known_node_count = 0

    # extract info
    party_infos = []
    for node in nodes_list:
        if (node[:8] == "UNKNOWN "):
            party_info_str = re.sub(r'[^A-Za-z0-9]+', '', (node.split(",")[-1]).lower()[1:])
            print(party_info_str)
            # catch organism difference
            if (party_info_str == "organism"):
                party_info_str = "organismtaxon"
            party_info_labeled = {
                "type": party_info_str,
                "label": 0 #unknown category
            }
            G.nodes[node]["metadata"] = party_info_labeled
        else:
            known_type = re.sub(r'[^A-Za-z0-9]+', '', node.split(",")[-1].lower()[1:])
            if known_type == "organism":
                known_type = "organismtaxon"
            G.nodes[node]["metadata"] = {"info": node.split(",")[0], "label": 1, "type": known_type} # known category
            #print(known_type)
            graph_known_node_count += 1

    G.graph["known_count"] = graph_known_node_count


def chromadb_storage_name_pull():
    base_dir = os.path.join(os.path.dirname(__file__), "local_files")
    embeddings_dir = os.path.join(base_dir, "Embeddings")
    file_names = [f for f in os.listdir(embeddings_dir) if os.path.isfile(os.path.join(embeddings_dir, f))]
    chromadb_names = file_names.copy()
    chromadb_names.append("all_nodes")
    return chromadb_names

def dynamic_subchunk_to_parent_mapping(length):
    counts = [1] * length
    parent_to_subchunk_mapping = {}
    current_index = 0
    for line_idx, count in enumerate(counts):
        parent_to_subchunk_mapping[line_idx] = list(range(current_index, current_index + count))
        current_index += count

    subchunk_to_parent = {}
    for parent_idx, subchunk_list in parent_to_subchunk_mapping.items():
        for subchunk_idx in subchunk_list:
            subchunk_to_parent[subchunk_idx] = parent_idx

    return subchunk_to_parent

def query_chromadb(query, collection, subchunk_to_parent_map, search_info, top_k=1000):
    # Generate embedding for query text
    if (search_info == "question"):
        query_embedding = voyage_embedder.question_or_info_embedder(query)
    else:
        query_embedding = voyage_embedder.question_or_info_embedder(search_info)
    if query_embedding:
        # 1) Query top_k sub-chunks from your ChromaDB collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # 2) Extract the relevant fields from the result
        distances = results.get("distances", [[]])[0]  # list of distances
        idss = results.get("ids", [[]])[0]             # list of sub-chunk IDs
        idss = list(map(int, idss))                    # ensure they're integers (if needed)

        # 3) Build a list of (subchunk_id, parent_id, distance) or a dict
        subchunk_info_list = []
        for subchunk_id, distance in zip(idss, distances):
            parent_id = subchunk_to_parent_map[subchunk_id]
            subchunk_info_list.append({
                "subchunk_id": subchunk_id,
                "parent_id": parent_id,
                "distance": distance
            })

        return subchunk_info_list
    else:
        print("Failed to generate query embedding.")
        return []
    
def group_by_parent(subchunk_info_list):
    grouped = {}
    for info in subchunk_info_list:
        parent_id = info["parent_id"]
        distance = info["distance"]
        
        # If parent_id not in grouped, store it
        if parent_id not in grouped:
            grouped[parent_id] = {
                "distance": distance,
                "subchunks": [info["subchunk_id"]]
            }
        else:
            # Update min distance if needed
            if distance < grouped[parent_id]["distance"]:
                grouped[parent_id]["distance"] = distance
            grouped[parent_id]["subchunks"].append(info["subchunk_id"])
            
    # Convert dict to list of (parent_id, distance, subchunks)
    # Then sort by distance ascending
    parent_info_list = []
    for parent_id, val in grouped.items():
        parent_info_list.append({
            "parent_id": parent_id,
            "min_distance": val["distance"],
            "subchunks": val["subchunks"]
        })
    
    parent_info_list.sort(key=lambda x: x["min_distance"])
    return parent_info_list

def rank_documents(query_text, doc_vectors, the_bm25):
    query_sparse_vector = the_bm25.encode_queries(query_text)

    # Extract query indices and values
    query_indices = query_sparse_vector["indices"]
    query_values = np.array(query_sparse_vector["values"])  # Convert to numpy array

    rankings = []
    
    for doc_id, doc_vec in doc_vectors.items():
        doc_indices = doc_vec["indices"]
        doc_values = np.array(doc_vec["values"])

        # Compute similarity: dot product of matching indices
        matching_indices = set(query_indices) & set(doc_indices)  # Common terms
        query_mask = [query_indices.index(i) for i in matching_indices if i in query_indices]
        doc_mask = [doc_indices.index(i) for i in matching_indices if i in doc_indices]

        if matching_indices:
            similarity_score = np.dot(query_values[query_mask], doc_values[doc_mask])
        else:
            similarity_score = 0  # No match

        rankings.append((doc_id, similarity_score))

    # Sort documents by score (descending)
    rankings.sort(key=lambda x: x[1], reverse=True)

    # Extract ordered document indices (ranked)
    return rankings

def filter_by_similarity_topk(
    results: List[Tuple[int, float]],
    topk: int = 35
) -> List[Tuple[int, float]]:
    """
    Returns the top-k most similar results.

    Parameters:
        results: List of (index, similarity) tuples sorted by similarity (descending)
        topk: Number of top results to return

    Returns:
        List of top-k (index, similarity) tuples
    """
    return results[:topk]

def hybrid_search(embedding_result, kw_result):
    alpha = 0.5  # Weight for keyword similarity vs. embedding similarity

    hybrid_result = []

    # Convert list of (parent_id, score) into {parent_id: score}
    kw_dict = dict(kw_result)

    # Collect all unique parent_ids from both retrievals
    all_parent_ids = set(item["parent_id"] for item in embedding_result) | set(kw_dict.keys())

    for pid in all_parent_ids:
        # Get embedding similarity (convert distance if available)
        embedding_entry = next((item for item in embedding_result if item["parent_id"] == pid), None)
        similarity_score_embedding = 1.0 - embedding_entry["min_distance"] if embedding_entry else 0.0

        # Get keyword score (default to 0 if not found)
        kw_score = kw_dict.get(pid, 0.0)

        # Compute hybrid score
        hybrid_score = alpha * kw_score + (1 - alpha) * similarity_score_embedding

        # Store results
        combined_item = {
            "parent_id": pid,
            "kw_score": kw_score,
            "embedding_similarity": similarity_score_embedding,
            "hybrid_score": hybrid_score,
        }
        hybrid_result.append(combined_item)

    # Sort by hybrid score (higher is better)
    hybrid_result.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return(hybrid_result)

def candidate_nodes_pid_finder(info, search_category):
    local_files_dir = os.path.join(os.path.dirname(__file__), "local_files")
    chroma_db_dir = os.path.join(local_files_dir, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_db_dir)
    # embedding search
    category_found = False
    if (search_category == "all"):
        collection = client.get_collection(name="all_nodes")
        subchunk_to_parent_map = dynamic_subchunk_to_parent_mapping(collection.count())
        retrieved_chunks_embedding = group_by_parent(query_chromadb(info, collection, subchunk_to_parent_map, info))
        category_found = True
    else:
        for chromadb_name in chromadb_names:
            if (chromadb_name.split("_")[0] == search_category):
                # selecting right collection to search within
                collection = client.get_collection(name=chromadb_name)
                subchunk_to_parent_map = dynamic_subchunk_to_parent_mapping(collection.count())
                retrieved_chunks_embedding = group_by_parent(query_chromadb(info, collection, subchunk_to_parent_map, info))
                category_found = True
    if (not category_found):
        return []
    # keyword search
    for bm25_module in all_bm25s_g:      
        # selecting right bm25 module
        if (bm25_module["name"] == search_category):
            if (info == "question"):
                #chunk_rankings_kw = rank_documents(questions[question_index], bm25_module["vectors"], bm25_module["data"]) # all scores for hybrid search
                pass
            else:
                chunk_rankings_kw = rank_documents(info, bm25_module["vectors"], bm25_module["data"]) # all scores for hybrid search
            #print(filter_by_similarity_topk(chunk_rankings_kw, 10))
            retrieved_chunks_kw = filter_by_similarity_topk(chunk_rankings_kw, int(candidate_node_count_determiner/4)) # top 5 from kw search
            print(retrieved_chunks_kw)
    # hybrid search
    retrieved_chunks_hybrid = filter_by_similarity_topk(hybrid_search(retrieved_chunks_embedding, chunk_rankings_kw), int(3*candidate_node_count_determiner/4))
    print(retrieved_chunks_hybrid)
    # pre-retrieval
    pid_list_root = []
    for kw_chunk in retrieved_chunks_kw:
        if kw_chunk[0] not in pid_list_root:
            entry = {
                "pid": kw_chunk[0],
                "kw_score": kw_chunk[1],
                "self_category": search_category
            }
            pid_list_root.append(entry)
    for hybrid_chunk in retrieved_chunks_hybrid:
        if hybrid_chunk["parent_id"] not in pid_list_root:
            entry = {
                "pid": hybrid_chunk["parent_id"],
                "kw_score": hybrid_chunk["kw_score"],
                "hybrid_score": hybrid_chunk["hybrid_score"],
                "self_category": search_category
            }
            pid_list_root.append(entry)
    return pid_list_root

def candidate_nodes_pid_finder_for_blind(info, search_category):
    local_files_dir = os.path.join(os.path.dirname(__file__), "local_files")
    chroma_db_dir = os.path.join(local_files_dir, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_db_dir)
    # embedding search
    category_found = False
    if (search_category == "all"):
        collection = client.get_collection(name="all_nodes")
        subchunk_to_parent_map = dynamic_subchunk_to_parent_mapping(collection.count())
        retrieved_chunks_embedding = group_by_parent(query_chromadb(info, collection, subchunk_to_parent_map, info))
        category_found = True
    else:
        for chromadb_name in chromadb_names:
            if (chromadb_name.split("_")[0] == search_category):
                # selecting right collection to search within
                collection = client.get_collection(name=chromadb_name)
                subchunk_to_parent_map = dynamic_subchunk_to_parent_mapping(collection.count())
                retrieved_chunks_embedding = group_by_parent(query_chromadb(info, collection, subchunk_to_parent_map, info))
                category_found = True
    if (not category_found):
        return []
    # keyword search
    for bm25_module in all_bm25s_g:      
        # selecting right bm25 module
        if (bm25_module["name"] == search_category):
            if (info == "question"):
                #chunk_rankings_kw = rank_documents(questions[question_index], bm25_module["vectors"], bm25_module["data"]) # all scores for hybrid search
                pass
            else:
                chunk_rankings_kw = rank_documents(info, bm25_module["vectors"], bm25_module["data"]) # all scores for hybrid search
            #print(filter_by_similarity_topk(chunk_rankings_kw, 10))
            retrieved_chunks_kw = filter_by_similarity_topk(chunk_rankings_kw, 125) # top 5 from kw search
            print(retrieved_chunks_kw)
    # hybrid search
    retrieved_chunks_hybrid = filter_by_similarity_topk(hybrid_search(retrieved_chunks_embedding, chunk_rankings_kw), 375)
    print(retrieved_chunks_hybrid)
    # pre-retrieval
    pid_list_root = []
    for kw_chunk in retrieved_chunks_kw:
        if kw_chunk[0] not in pid_list_root:
            entry = {
                "pid": kw_chunk[0],
                "kw_score": kw_chunk[1],
                "self_category": search_category
            }
            pid_list_root.append(entry)
    for hybrid_chunk in retrieved_chunks_hybrid:
        if hybrid_chunk["parent_id"] not in pid_list_root:
            entry = {
                "pid": hybrid_chunk["parent_id"],
                "kw_score": hybrid_chunk["kw_score"],
                "hybrid_score": hybrid_chunk["hybrid_score"],
                "self_category": search_category
            }
            pid_list_root.append(entry)
    return pid_list_root

def get_original_class_name(lower_name):
    original_classes = [
        "Drug", "SmallMolecule", "Pathway", "Disease", "CellularComponent",
        "GOTerm", "Compound", "BiologicalProcess", "Protein", "EcNumber",
        "Reaction", "Phenotype", "Gene", "OrganismTaxon", "ProteinDomain",
        "MolecularFunction", "SideEffect"
    ]
    class_map = {cls.lower(): cls for cls in original_classes}

    return class_map.get(lower_name, lower_name)  # fallback to original if not found

def find_name_from_pid(pid, search_category):
    local_files_dir = os.path.join(os.path.dirname(__file__), "local_files")
    chroma_db_dir = os.path.join(local_files_dir, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_db_dir)

    for chromadb_name in chromadb_names:
        if (search_category == chromadb_name.split("_")[0]):
            collection = client.get_collection(name=chromadb_name)
            result = collection.get(ids=str(pid))
            document = result["documents"][0][1:-3]
            return document.split(" is ")[0]
        
def match_pattern_around_node(G, center_node_name_in_db, center_node_name, center_node_category):
    n0_category = get_original_class_name(center_node_category)
    var_map = {center_node_name: "n0"}
    match_parts = []
    var_index = 1
    rel_index = 1

    for u, v, data in G.edges(data=True):
        rel = data.get("label", "RELATED_TO")

        if u not in var_map:
            var_map[u] = f"n{var_index}"
            var_index += 1
        if v not in var_map:
            var_map[v] = f"n{var_index}"
            var_index += 1

        u_label = n0_category if u == center_node_name else get_original_class_name(G.nodes[u].get("metadata", {}).get("type", ""))
        v_label = n0_category if v == center_node_name else get_original_class_name(G.nodes[v].get("metadata", {}).get("type", ""))

        u_var = var_map[u]
        v_var = var_map[v]

        # Handle loopback (u == v)
        if u == v:
            v_var = f"{u_var}_loop"

        rel_var = f"r{rel_index}"
        rel_index += 1

        match_parts.append(f"({u_var}:{u_label})-[{rel_var}:{rel}]-({v_var}:{v_label})")

    match_clause = "MATCH " + ",\n      ".join(match_parts)

    if center_node_category == "gene":
        the_property = "gene_symbol"
    elif center_node_category == "protein":
        the_property = "primary_protein_name"
    elif center_node_category == "compound":
        the_property = "smiles"
    elif center_node_category == "organismtaxon":
        the_property = "organism_name"
    else:
        the_property = "name"

    where_clause = f"WHERE n0.{the_property} = '{center_node_name_in_db}'"

    full_query = f"""
    {match_clause}
    {where_clause}
    RETURN *
    LIMIT {graph_match_count_determiner}
    """.strip()

    #print(full_query)

    excluded_keys = {
        "esm2_embedding", "prott5_embedding", "biokeen_embedding", "doc2vec_embedding",
        "anc2vec_embedding", "cada_embedding", "nt_embedding", "rxnfp_embedding", "selformer_embedding"
    }

    result_graphs = []
    with driver_n4j.session() as session:
        result = session.run(full_query)
        for record in result:
            result_graph = nx.Graph()

            for value in record.values():
                if isinstance(value, dict):
                    continue
                elif hasattr(value, "nodes") and hasattr(value, "relationships"):
                    for node in value.nodes:
                        node_dict = dict(node)
                        filtered_node_data = {k: v for k, v in node_dict.items() if k not in excluded_keys}
                        filtered_node_data["__neo4j_id"] = node.id
                        filtered_node_data["__labels"] = list(node.labels)
                        result_graph.add_node(node.id, **filtered_node_data)

                    for rel in value.relationships:
                        result_graph.add_edge(
                            rel.start_node.id,
                            rel.end_node.id,
                            label=rel.type,
                            **dict(rel)
                        )
                elif hasattr(value, "type"):
                    rel = value
                    result_graph.add_edge(
                        rel.start_node.id,
                        rel.end_node.id,
                        label=rel.type,
                        **dict(rel)
                    )
                elif hasattr(value, "id"):
                    node = value
                    node_dict = dict(node)
                    filtered_node_data = {k: v for k, v in node_dict.items() if k not in excluded_keys}
                    filtered_node_data["__neo4j_id"] = node.id
                    filtered_node_data["__labels"] = list(node.labels)
                    result_graph.add_node(node.id, **filtered_node_data)

            result_graphs.append(result_graph)

    return result_graphs

def get_primary_identifier(node_attrs):
    labels = node_attrs.get("__labels", ["Entity"])
    # Assume the first label is the main category
    main_label = labels[0].lower() if labels else "entity"

    if main_label == "gene":
        return node_attrs.get("gene_symbol") or node_attrs.get("id") or "Unnamed"
    elif main_label == "protein":
        return node_attrs.get("primary_protein_name") or node_attrs.get("id") or "Unnamed"
    elif main_label == "compound":
        return node_attrs.get("smiles") or node_attrs.get("id") or "Unnamed"
    else:
        return node_attrs.get("name") or node_attrs.get("id") or "Unnamed"

def format_property_value(value):
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)

def textualize_graph_fully(graph):
    lines = []

    # Node descriptions
    for node_id, attrs in graph.nodes(data=True):
        identifier = get_primary_identifier(attrs)
        labels = attrs.get("__labels", ["Entity"])
        label_str = ", ".join(labels) if labels else "Entity"
        prop_str = "; ".join(
            f"{k}: {format_property_value(v)}"
            for k, v in attrs.items()
            if k not in {"name", "gene_symbol", "smiles", "primary_protein_name", "__labels", "__neo4j_id"}
        )
        lines.append(f"- {identifier} is a {label_str} with the following properties: {prop_str}.")

    # Relationship descriptions
    for u, v, data in graph.edges(data=True):
        u_attrs = graph.nodes[u]
        v_attrs = graph.nodes[v]

        u_name = get_primary_identifier(u_attrs)
        v_name = get_primary_identifier(v_attrs)

        rel_type = data.get("label", "is related to").replace("_", " ").lower()
        full_rel_desc = f"{', '.join(u_attrs.get('__labels', ['entity']))} {rel_type} {', '.join(v_attrs.get('__labels', ['entity']))}"

        rel_props = {k: v for k, v in data.items() if k != "label"}
        rel_prop_str = "; ".join(f"{k}: {format_property_value(v)}" for k, v in rel_props.items()) or "no additional properties"

        lines.append(f"- {u_name} has a relationship with {v_name} as \"{full_rel_desc}\" with {rel_prop_str}.")

    return "\n".join(lines)

def node_type_checker_from_relationship(type_to_check, relationship, relationship_map):
    for unique_rel in relationship_map:
        if (relationship == unique_rel["name"]):
            if (unique_rel["party_a"] == type_to_check):
                return unique_rel["party_a"]
            elif (unique_rel["party_b"] == type_to_check):
                return unique_rel["party_b"]
            else:
                return "corrupted"
    return "corrupted"

def match_pattern_blind(G):
    var_map = {}
    match_parts = []
    var_index = 0
    rel_index = 0

    # Assign variable names to nodes
    for node in G.nodes:
        var_map[node] = f"n{var_index}"
        var_index += 1

    # Create match patterns for edges
    for u, v, data in G.edges(data=True):
        u_label = get_original_class_name(G.nodes[u].get("metadata", {}).get("type", ""))
        v_label = get_original_class_name(G.nodes[v].get("metadata", {}).get("type", ""))
        rel_label = data.get("label", "RELATED_TO")

        u_var = var_map[u]
        v_var = var_map[v]

        if u == v:
            # Loopback detected: create a new variable name for the second occurrence
            v_var = f"{var_map[v]}_loop"

        rel_var = f"r{rel_index}"
        rel_index += 1

        match_parts.append(f"({u_var}:{u_label})-[{rel_var}:{rel_label}]-({v_var}:{v_label})")

    # Assemble the MATCH clause
    match_clause = "MATCH " + ",\n      ".join(match_parts)

    # Full query (LIMIT 2000 manually added)
    full_query = f"""
    {match_clause}
    RETURN *
    LIMIT 2000
    """.strip()

    #print(full_query)

    # Exclude heavy attributes
    excluded_keys = {
        "esm2_embedding", "prott5_embedding", "biokeen_embedding", "doc2vec_embedding",
        "anc2vec_embedding", "cada_embedding", "nt_embedding", "rxnfp_embedding", "selformer_embedding"
    }

    result_graphs = []
    with driver_n4j.session() as session:
        result = session.run(full_query)
        for record in result:
            result_graph = nx.Graph()

            for value in record.values():
                if isinstance(value, dict):
                    continue
                elif hasattr(value, "nodes") and hasattr(value, "relationships"):
                    for node in value.nodes:
                        node_dict = dict(node)
                        filtered_node_data = {k: v for k, v in node_dict.items() if k not in excluded_keys}
                        filtered_node_data["__neo4j_id"] = node.id
                        filtered_node_data["__labels"] = list(node.labels)
                        result_graph.add_node(node.id, **filtered_node_data)

                    for rel in value.relationships:
                        result_graph.add_edge(
                            rel.start_node.id,
                            rel.end_node.id,
                            label=rel.type,
                            **dict(rel)
                        )
                elif hasattr(value, "type"):
                    rel = value
                    result_graph.add_edge(
                        rel.start_node.id,
                        rel.end_node.id,
                        label=rel.type,
                        **dict(rel)
                    )
                elif hasattr(value, "id"):
                    node = value
                    node_dict = dict(node)
                    filtered_node_data = {k: v for k, v in node_dict.items() if k not in excluded_keys}
                    filtered_node_data["__neo4j_id"] = node.id
                    filtered_node_data["__labels"] = list(node.labels)
                    result_graph.add_node(node.id, **filtered_node_data)

            result_graphs.append(result_graph)

    return result_graphs

def match_pattern_around_two_nodes(G, center1_name_in_db, center1_name, center1_category,
                                    center2_name_in_db, center2_name, center2_category):
    var_map = {
        center1_name: "n0",
        center2_name: "n1"
    }
    match_parts = []
    var_index = 2
    rel_index = 1

    for u, v, data in G.edges(data=True):
        rel = data.get("label", "RELATED_TO")

        if u not in var_map:
            var_map[u] = f"n{var_index}"
            var_index += 1
        if v not in var_map:
            var_map[v] = f"n{var_index}"
            var_index += 1

        u_label = get_original_class_name(center1_category if u == center1_name else (
            center2_category if u == center2_name else G.nodes[u].get("metadata", {}).get("type", "")))
        v_label = get_original_class_name(center1_category if v == center1_name else (
            center2_category if v == center2_name else G.nodes[v].get("metadata", {}).get("type", "")))

        u_var = var_map[u]
        v_var = var_map[v]

        if u == v:
            v_var = f"{u_var}_loop"

        rel_var = f"r{rel_index}"
        rel_index += 1

        match_parts.append(f"({u_var}:{u_label})-[{rel_var}:{rel}]-({v_var}:{v_label})")

    match_clause = "MATCH " + ",\n      ".join(match_parts)

    def get_property(cat):
        return {
            "gene": "gene_symbol",
            "protein": "primary_protein_name",
            "compound": "smiles",
            "organismtaxon": "organism_name"
        }.get(cat, "name")

    where_clause = (
        f"WHERE n0.{get_property(center1_category)} = '{center1_name_in_db}' AND "
        f"n1.{get_property(center2_category)} = '{center2_name_in_db}'"
    )

    full_query = f"""
    {match_clause}
    {where_clause}
    RETURN *
    LIMIT {graph_match_count_determiner}
    """.strip()

    excluded_keys = {
        "esm2_embedding", "prott5_embedding", "biokeen_embedding", "doc2vec_embedding",
        "anc2vec_embedding", "cada_embedding", "nt_embedding", "rxnfp_embedding", "selformer_embedding"
    }

    result_graphs = []
    with driver_n4j.session() as session:
        result = session.run(full_query)
        for record in result:
            result_graph = nx.Graph()

            for value in record.values():
                if isinstance(value, dict):
                    continue
                elif hasattr(value, "nodes") and hasattr(value, "relationships"):
                    for node in value.nodes:
                        node_dict = dict(node)
                        filtered_node_data = {k: v for k, v in node_dict.items() if k not in excluded_keys}
                        filtered_node_data["__neo4j_id"] = node.id
                        filtered_node_data["__labels"] = list(node.labels)
                        result_graph.add_node(node.id, **filtered_node_data)

                    for rel in value.relationships:
                        result_graph.add_edge(
                            rel.start_node.id,
                            rel.end_node.id,
                            label=rel.type,
                            **dict(rel)
                        )
                elif hasattr(value, "type"):
                    rel = value
                    result_graph.add_edge(
                        rel.start_node.id,
                        rel.end_node.id,
                        label=rel.type,
                        **dict(rel)
                    )
                elif hasattr(value, "id"):
                    node = value
                    node_dict = dict(node)
                    filtered_node_data = {k: v for k, v in node_dict.items() if k not in excluded_keys}
                    filtered_node_data["__neo4j_id"] = node.id
                    filtered_node_data["__labels"] = list(node.labels)
                    result_graph.add_node(node.id, **filtered_node_data)

            result_graphs.append(result_graph)

    return result_graphs

def find_doc_from_pid(pid, search_category):
    local_files_dir = os.path.join(os.path.dirname(__file__), "local_files")
    chroma_db_dir = os.path.join(local_files_dir, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_db_dir)
    collection = client.get_collection(name=search_category)
    result = collection.get(ids=str(pid))
    document = result["documents"][0][1:-3]
    print(document)
    return document

# pipeline distributor
def process_graph(G):
    all_retrievals = []
    # no candidates, blind search in the db; including 1 noded looped back graphs
    if (G.graph["known_count"] == 0):
        possible_graphs = match_pattern_blind(G)
        for possible_graph in possible_graphs:
            all_retrievals.append(textualize_graph_fully(possible_graph))
    # single candidate creating node, get candidates, then try pattern matching
    elif (G.graph["known_count"] == 1):
        # get the info and search category from info node
        info_node_name = ""
        info = ""
        search_category = ""
        for node_name, node in G.nodes(data=True):
            if (node.get("metadata")["label"] == 1):
                info_node_name = node_name
                info = node.get("metadata")["info"]
                # get one of the neighbors of the info node with its relationship; tuple form = node, neighbor, {"label": rel}
                triple_tuple = next(iter(G.edges(node_name, data=True)), None)
                search_category = node_type_checker_from_relationship(node.get("metadata")["type"], triple_tuple[2]["label"], relationship_mappings)
        # no search, llm output corrupted
        if (search_category == "corrupted"):
            all_retrievals = []
        else:
            candidate_pids_as_list_of_dicts = candidate_nodes_pid_finder(info, search_category)
            for candidate_node in candidate_pids_as_list_of_dicts:
                candidate_name = find_name_from_pid(candidate_node["pid"], search_category)
                possible_graphs = match_pattern_around_node(G, candidate_name, info_node_name, search_category)
                for possible_graph in possible_graphs:
                    all_retrievals.append(textualize_graph_fully(possible_graph))
    # multiple candidate creating node, 
    else:
        info_node_names = []
        infos = []
        search_categories = []
        for node_name, node in G.nodes(data=True):
            if (node.get("metadata")["label"] == 1):
                info_node_name = node_name
                info_node_names.append(info_node_name)
                info = node.get("metadata")["info"]
                infos.append(info)
                # get one of the neighbors of the info-node with its relationship; tuple form = node, neighbor, {"label": rel}
                triple_tuple = next(iter(G.edges(node_name, data=True)), None)
                search_category = node_type_checker_from_relationship(node.get("metadata")["type"], triple_tuple[2]["label"], relationship_mappings)
                search_categories.append(search_category)
        all_corrupted = all(element == "corrupted" for element in search_categories)
        if (all_corrupted):
            all_retrievals = []
        else:
            # combined search
            candidate_pids_higher_list_of_dicts = []
            for the_info_node_name, the_info, the_search_category in zip(info_node_names, infos, search_categories):
                candidate_pids_higher_list_of_dicts.append(candidate_nodes_pid_finder(the_info, the_search_category))
            binary_combination_first_set_elements = []
            binary_combination_second_set_elements = []
            for i, candidate_pids_inner_list_of_dicts in enumerate(candidate_pids_higher_list_of_dicts):
                if i == 2:
                    break
                elif i == 0:
                    for candidate_node in candidate_pids_inner_list_of_dicts:
                        binary_combination_first_set_elements.append({"candidate_node": candidate_node, "candidate_name":find_name_from_pid(candidate_node["pid"], candidate_node["self_category"])})
                elif i == 1:
                    for candidate_node in candidate_pids_inner_list_of_dicts:
                        binary_combination_second_set_elements.append({"candidate_node": candidate_node, "candidate_name":find_name_from_pid(candidate_node["pid"], candidate_node["self_category"])})
            binary_combination_sets = [(d1, d2) for d1, d2 in product(binary_combination_first_set_elements, binary_combination_second_set_elements)]
            for combination in binary_combination_sets:
                for i, j, k in zip(range(len(info_node_names)), range(len(infos)), range(len(search_categories))):
                    possible_graphs = match_pattern_around_two_nodes(G, combination[0]["candidate_name"], info_node_names[i], search_categories[k],
                                                                     combination[1]["candidate_name"], info_node_names[i+1], search_categories[k+1])
                    for possible_graph in possible_graphs:
                        all_retrievals.append(textualize_graph_fully(possible_graph))
                    break

            # separate search
            for the_info_node_name, the_info, the_search_category in zip(info_node_names, infos, search_categories):
                candidate_pids_as_list_of_dicts = candidate_nodes_pid_finder(the_info, the_search_category)
                for candidate_node in candidate_pids_as_list_of_dicts:
                    candidate_name = find_name_from_pid(candidate_node["pid"], the_search_category)
                    possible_graphs = match_pattern_around_node(G, candidate_name, the_info_node_name, the_search_category)
                    for possible_graph in possible_graphs:
                        all_retrievals.append(textualize_graph_fully(possible_graph))

    return all_retrievals

def no_graph_search(user_prompt):
    all_retrievals = []
    candidate_pids_as_list_of_dicts = candidate_nodes_pid_finder(user_prompt, "all")
    print((candidate_pids_as_list_of_dicts))
    for candidate_node in candidate_pids_as_list_of_dicts:
        candidate_text = find_doc_from_pid(candidate_node["pid"], "all_nodes")
        all_retrievals.append(candidate_text)
    return all_retrievals

def deduplicate_result(generation_data):
    seen = set()
    deduped_prompt = []
    for block in generation_data:
        lines = block.splitlines()
        is_a_lines = [line for line in lines if " is a " in line]
        signature = tuple(sorted(is_a_lines))

        if signature not in seen:
            seen.add(signature)
            deduped_prompt.append(block)
    return (deduped_prompt)


        
