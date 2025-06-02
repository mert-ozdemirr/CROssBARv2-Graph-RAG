import voyageai

vo = voyageai.Client(api_key="pa-Ke2T1z9FHzQlzpPkZW_X8wVElvOmfKvXwKy_sAZVbRC") 

def question_or_info_embedder(info):
    response = vo.embed(info, model="voyage-3-large") 
    return response.embeddings[0]