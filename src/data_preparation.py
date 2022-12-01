import pickle
from sklearn.model_selection import train_test_split
from typing import List
from tqdm import tqdm
from collections import Counter
import numpy as np
from random import randint
import csv
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers import evaluation
from torch.utils.data import DataLoader


def load_raw_list_of_pis(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_train_test_splits(dataset_name, dataset_path, lower_pis=True):

    bios = load_raw_list_of_pis(dataset_path)

    if lower_pis:
        processed_pis = []
        for bio in bios:
            lower_bio = []
            for pi in bio:
                lower_bio.append(pi.lower())
            processed_pis.append(lower_bio)
    else:
        processed_pis = bios

    train, test = train_test_split(processed_pis, test_size=0.2, shuffle=True)
    print(f"train size: {len(train)} test size: {len(test)}")

    with open(f'../data/{dataset_name}_test_bios.pkl', 'wb') as f:
        pickle.dump(test, f)

    with open(f'../data/{dataset_name}_train_bios.pkl', 'wb') as f:
        pickle.dump(train, f)

    return train, test


def create_train_dataset(bios: List[List[str]], dataset_name: str, min_pi_freq: int):

    pi_cnt = Counter()
    for bio in tqdm(bios):
        pi_cnt.update(bio)

    result = []
    for pis in tqdm(bios):
        current_pi = set()
        for pi in pis:
            # don't keep pis which are characters!
            if len(pi) >= 2 and pi_cnt[pi] >= min_pi_freq:
                current_pi.add(pi)
        # bio must contain at least 2 pis
        if len(current_pi) > 1:
            result.append(list(current_pi))

    print(f"number of bios after cleanup: {len(result)}")

    #persist data
    with open(f'../data/{dataset_name}_cleaned_train_bios_{min_pi_freq}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return result


def pair_in_list(current_pair, l):
    for pair in l:
        if current_pair[0] in pair and current_pair[1] in pair:
            return True
    return False


def generate_triplets(bios, neighbors, k=3):

    pi_set = list(neighbors.keys())
    samples = []
    for idx, bio in tqdm(enumerate(bios), total=len(bios)):
        if len(bio) != len(set(bio)):
            continue
        iters = min(len(bio)-1, k)
        chosen_pis = []
        for i in range(iters):
            pos1, pos2 = np.random.choice(bio, size=2, replace=False)
            while pair_in_list([pos1, pos2], chosen_pis):
                pos1, pos2 = np.random.choice(bio, size=2, replace=False)
            chosen_pis.append([pos1, pos2])
            neg_idx = randint(0, len(pi_set)-1)
            while pi_set[neg_idx] in neighbors[pos1] or pi_set[neg_idx] in neighbors[pos2]:
                neg_idx = randint(0, len(pi_set)-1)
            samples.append([pos1, pos2, pi_set[neg_idx]])
    return samples


def generate_contrastive_learning_dataset(bios: List[List[str]], k=3, validation_split=0.01):
    neighbors = {}

    print("generating pi neighbors...")
    for bio in tqdm(bios):
        for pi in bio:
            if pi not in neighbors:
                neighbors[pi] = Counter()

            rest = [b for b in bio if b != pi]
            neighbors[pi].update(rest)

    print("generating triples...")
    triples = generate_triplets(bios, neighbors, k=k)

    train_set, valid_set = train_test_split(triples, test_size=validation_split, shuffle=True)
    return train_set, valid_set


def prepare_contrastive_learning_dataset(raw_dataset_path, dataset_name, min_pi_frequency):
    train, _ = create_train_test_splits(dataset_name, raw_dataset_path, lower_pis=True)
    cleaned_train_bios = create_train_dataset(train, dataset_name, min_pi_frequency)

    train_triplets, validation_triplets = generate_contrastive_learning_dataset(cleaned_train_bios,
                                                                                validation_split=0.01)

    with open(f'../data/{dataset_name}-triplets_{min_pi_frequency}.pkl', 'wb') as f:
        pickle.dump(train_triplets, f)

    with open(f'../data/{dataset_name}-valid-{min_pi_frequency}.csv', 'w') as f:
        write = csv.writer(f, delimiter='\t')
        write.writerows(validation_triplets)

    with open(f'../data/{dataset_name}-train-{min_pi_frequency}.csv', 'w') as f:
        write = csv.writer(f, delimiter='\t')
        write.writerows(validation_triplets)


def get_data_loader(dataset_name, min_pi_frequency, batch_size=256):

    train_examples = []
    with open(f'../data/{dataset_name}-train-{min_pi_frequency}.csv', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader):
            train_examples.append(InputExample(texts=[row[0], row[1]], label=1.0))
            train_examples.append(InputExample(texts=[row[0], row[2]], label=0.0))

    with open(f'../data/{dataset_name}-valid-{min_pi_frequency}.csv', newline='') as f:
        sent1s = []
        sent2s = []
        scores = []
        i = 0
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader):
            sent1s.append(row[0])
            sent1s.append(row[0])
            sent2s.append(row[1])
            sent2s.append(row[2])
            scores.append(1.0)
            scores.append(0.0)
            i += 1

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sent1s, sent2s, scores)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    return train_dataloader, evaluator


if __name__ == "__main__":
    prepare_contrastive_learning_dataset(
        raw_dataset_path='../data/pis2020.pkl',
        dataset_name='twitter',
        min_pi_frequency=100
    )