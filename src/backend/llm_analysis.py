import google.generativeai as genai
import re
import json

client = genai.configure(api_key='AIzaSyDbMReeHt_IAjjLqDe2OVTJPFPClJLshBQ')

model=genai.GenerativeModel(
model_name="gemini-2.5-flash-preview-04-17",
system_instruction="""You need to segment the given query then extract the knowledge graph structures.
    Notes)
    1). Use the original description in the query with enough context, NEVER use unspecific words like ’in’, ’appear in’,
    ’for’, ’of’ etc.
    2). For nodes that are completely unknown, not partially unknown, you must use the keyword ’UNKNOWN’ with a unique ID, e.g.,
    ’UNKNOWN protein 1, protein’. NEVER present an generic node with it's class name only like "Drug" etc. present it with the correct syntax of e.g. "UNKNOWN drug 1, drug". Other than that use the full or partial info as a name even though it's not a name.
    3). Return the segmented query and extracted graph structures strictly following the format:
    { "divided": [ "segment 1", ... ], "triples": [ ("head", "relation", "tail"), ... ] }
    4). NEVER provide extra descriptions or explanations, such as something like ’Here is the extracted knowledge
    graph structure’.
    5) Always add node type after info in triples in a structure like "head, protein" or "tail, protein" for each node.
    6) Select relations from the list below and create triples by using appropriate relationships. NEVER give me a triplet with a relationship not present in the list below.:
    Biological_process_is_a_biological_process
    Biological_process_negatively_regulates_biological_process
    Biological_process_negatively_regulates_molecular_function
    Biological_process_part_of_biological_process
    Biological_process_positively_regulates_biological_process
    Biological_process_positively_regulates_molecular_function
    Cellular_component_is_a_cellular_component
    Cellular_component_part_of_cellular_component
    Compound_targets_protein
    Disease_is_a_disease
    Disease_is_associated_with_disease
    Disease_is_comorbid_with_disease
    Disease_is_treated_by_drug
    Disease_modulates_pathway
    Drug_downregulates_gene
    Drug_has_side_effect
    Drug_has_target_in_pathway
    Drug_interacts_with_drug
    Drug_targets_protein
    Drug_upregulates_gene
    Ec_number_is_a_ec_number
    Gene_encodes_protein
    Gene_is_orthologous_with_gene
    Gene_is_related_to_disease
    Gene_regulates_gene
    Molecular_function_is_a_molecular_function
    Molecular_function_negatively_regulates_molecular_function
    Molecular_function_part_of_molecular_function
    Molecular_function_positively_regulates_molecular_function
    Organism_causes_disease
    Pathway_is_equivalent_to_pathway
    Pathway_is_ortholog_to_pathway
    Pathway_is_part_of_pathway
    Pathway_participates_pathway
    Phenotype_is_a_phenotype
    Phenotype_is_associated_with_disease
    Protein_belongs_to_organism
    Protein_catalyzes_ec_number
    Protein_contributes_to_molecular_function
    Protein_domain_enables_molecular_function
    Protein_domain_involved_in_biological_process
    Protein_domain_located_in_cellular_component
    Protein_enables_molecular_function
    Protein_has_domain
    Protein_interacts_with_protein
    Protein_involved_in_biological_process
    Protein_is_active_in_cellular_component
    Protein_is_associated_with_phenotype
    Protein_located_in_cellular_component
    Protein_part_of_cellular_component
    Protein_take_part_in_pathway
    Side_effect_is_a_side_effect
    7) Below is a list of information attributes per node type. Know that you've reached those attribute info if you reached a type of node and that means you are not creating a triplet for this. And a node type cannot be outside of this list, do not give any other things, exactly use those names.:
    Drug: atc_codes, bindingdb, cas_number, chebi, chembl, clinicaltrials, drugcentral, general_references, groups, id, inchi, inchikey, kegg_drug, name, pdb, pharmgkb, pubchem, rxcui, selformer_embedding, smiles, zinc
    SmallMolecule: alogp, atc_codes, bindingdb, cas_number, chebi, chembl, clinicaltrials, drugcentral, full_mwt, general_references, groups, heavy_atoms, id, inchi, inchikey, kegg_drug, name, pdb, pharmgkb, pubchem, qed_score, rxcui, selformer_embedding, smiles, species, type, zinc
    Pathway: biokeen_embedding, id, name, organism
    Disease: doc2vec_embedding, doid, efo, hp, icd10cm, icd9, id, meddra, mesh, name, ncit, omim, orphanet, synonyms, umls
    CellularComponent: anc2vec_embedding, id, name
    GOTerm: anc2vec_embedding, id, name
    Compound: alogp, full_mwt, heavy_atoms, id, inchi, inchikey, qed_score, selformer_embedding, smiles, species, type
    BiologicalProcess: anc2vec_embedding, id, name
    Protein: esm2_embedding, id, length, mass, organism_id, primary_protein_name, protein_names, prott5_embedding, sequence, xref_proteomes
    EcNumber: id, name, rxnfp_embedding
    Reaction: id, name, rxnfp_embedding
    Phenotype: cada_embedding, id, name, synonyms
    Gene: ensembl_gene_ids, ensembl_transcript_ids, gene_names, gene_symbol, id, kegg_ids, nt_embedding
    OrganismTaxon: id, organism_name
    ProteinDomain: child_list, dom2vec_embedding, ec, id, name, parent_list, pdb, pfam, protein_count, type
    MolecularFunction: anc2vec_embedding, id, name
    SideEffect: id, name, synonyms
    8) Please structure the graph in the plainest shape as you can, do not involve unnecessary triplets. NEVER repeat a triplet directly or changing orders of nodes.
    9) A correct example for you:
    Query: Which proteins are targeted by more than one drugs and more than one compounds?
    Answer:
    { "divided": [ "Which proteins are targeted by more than one drugs", "and more than one compounds?" ], "triples": [ ("UNKNOWN drug 1, drug", "Drug_targets_protein", "UNKNOWN protein 1, protein"), ("UNKNOWN drug 2, drug", "Drug_targets_protein", "UNKNOWN protein 1, protein"), ("UNKNOWN compound 1, compound", "Compound_targets_protein", "UNKNOWN protein 1, protein"), ("UNKNOWN compound 2, compound", "Compound_targets_protein", "UNKNOWN protein 1, protein") ] }}"""
)


def llm_subgraph_pattern_extractor(user_query):
    whole_prompt =  "Query: " + user_query
    response = model.generate_content(whole_prompt)
    raw_json = response.text
    raw_json = re.sub(r"^```json\n|\n```$", "", raw_json.strip())
    raw_json = raw_json.replace("(", "[").replace(")", "]")
    try:
        parsed_json = json.loads(raw_json)
        return str(json.dumps(parsed_json, indent=4))  # Pretty print JSON
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        print("Fixed JSON String:", raw_json)


def get_triples_str(llm_output):
    return re.search(r'"triples"\s*:\s*\[.*?\}\s*', llm_output, re.DOTALL).group(0).strip()[11:-1]

def parse_triples(str_triple):
    stack = []
    s = str_triple
    triples = []
    triple_element_histroy = ""
    while (len(s) > 0):
        char_under_inspection = s[0]
        if (char_under_inspection == "[" and len(stack) <= 2):
            stack.append(char_under_inspection)
            if (len(stack) > 2):
                triple_element_histroy += char_under_inspection
        elif (char_under_inspection == "]"):
            if (len(stack) > 2):
                triple_element_histroy += char_under_inspection
            stack.pop()
        elif (len(stack) >= 2 and char_under_inspection != "\n"):
            triple_element_histroy += char_under_inspection
        elif (len(stack) >= 2 and len(triple_element_histroy) > 0):
            triples.append(triple_element_histroy.strip(" ,\""))
            triple_element_histroy = ""
        else:
            triple_element_histroy = ""
        
        s = s[1:]

    triples_grouped = []
    sub_div = []
    for i in range(len(triples)):
        if (i%3 == 0):
            sub_div = []
        sub_div.append(triples[i])
        if ((i+1)%3 == 0):
            if (i != 0):
                triples_grouped.append(sub_div)

    return triples_grouped
