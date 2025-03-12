from sentence_transformers import SentenceTransformer
from .text_processing import extract_features_from_pdf
import numpy as np
from .utils import find_n_closest_vectors, most_frequent_element
from .text_processing import extract_abstract
from .LLM import get_rationale_from_LLM

def predict_paper_publishability(pdf_path, length_range=[0,160], density_range=[0,1.8], coherence_range=[0.40, 1], include_coherence=True):
    if include_coherence:
        sentence_length, density, coherence = extract_features_from_pdf(pdf_path, include_coherence=include_coherence)
        if (length_range[0]<=sentence_length<=length_range[1]) and (density_range[0]<=density<=density_range[1]) and (coherence_range[0]<=coherence<=coherence_range[1]):
            return True
        return False 
    else:
        sentence_length, density = extract_all_from_pdf(pdf_path, include_coherence=False)
        if (length_range[0]<=sentence_length<=length_range[1]) and (density_range[0]<=density<=density_range[1]):
            return True
        return False   
    
def predict_conference_reasoning(pdf_path, api_key): # this is the final funciton that gives the predicted conference of the research paper pdf
    titles_vector_base = np.load('KNN_train_data.npy') # Vector store can be used here
    titles_vector_labels = np.load('KNN_train_labels.npy')
    class_label = ['CVPR', 'EMNLP', 'KDD', 'NeurIPS', 'TMLR']
    abstract = extract_abstract(pdf_path)
    model_name = 'all-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    abstract_embed = model.encode(abstract)
    index = find_n_closest_vectors(titles_vector_base, abstract_embed, n=21)
    l = [titles_vector_labels[i] for i in index]
    conference_class = most_frequent_element(l)
    conference = class_label[conference_class]
    rationale = get_rationale_from_LLM(abstract, conference)
    return conference, rationale