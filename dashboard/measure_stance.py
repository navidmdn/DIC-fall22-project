from sentence_transformers import SentenceTransformer
from numpy import linalg as LA
import numpy as np


def normalize(wv):
    # normalize vectors
    norms = np.apply_along_axis(LA.norm, 1, wv)
    wv = wv / (norms[:, np.newaxis]+1e-6)
    return wv

def ripa(w, b):
    return w.dot(b) / LA.norm(b)


political_dimension = [
    ['republican', 'trump supporter', 'conservative'],
    ['democrat', 'biden', 'harris']
]

covid_dimension = [
    ['no masks',],
    ['wear the masks',]
]

immigration_dimension = [
    ['build the wall', 'stop immigrants'],
    ['immigration activist', 'help immigrants']
]

blacklivesmatter_dimension = [
    ['bluelivesmatter', 'blue lives matter'],
    ['blm', 'black lives matter',]
]

gunpolicy_dimension = [
    ['gun rights'],
    ['gun control',]
]


class StanceMeasure:
    def __init__(self, model_path: str):
        self.emb_model = SentenceTransformer(model_path)
        self.dims = [political_dimension, covid_dimension, immigration_dimension, blacklivesmatter_dimension,
                     gunpolicy_dimension]
        self.dim_poles = [
            ['partisanship: republican - democrat'],
            ['covid: anti vaccine/masking - pro vaccine/masking'],
            ['immigration: anti immigration - pro immigration'],
            ['blacklivesmatter movement: doesnt support blm - supports blm'],
            ['gun policy: anti gun policy - pro gun policy'],
        ]

    def get_dim_score(self, bio, dim):
        pole1 = self.emb_model.encode(','.join(dim[0])).reshape(1, -1)
        pole2 = self.emb_model.encode(','.join(dim[1])).reshape(1, -1)
        bio_emb = self.emb_model.encode(bio).reshape(1, -1)

        pole1_norm = normalize(pole1)[0]
        pole2_norm = normalize(pole2)[0]
        bio_norm = normalize(bio_emb)

        diff = pole2_norm - pole1_norm
        score = ripa(bio_norm, diff)
        return score[0]


stance_measure = StanceMeasure('../models/distilroberta-twitter-freq100/')
