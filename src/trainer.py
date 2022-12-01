import os

from sentence_transformers import SentenceTransformer, losses
import data_preparation


def train(base_model='all-distilroberta-v1', dataset_name='twitter', output_model_base_path='../models',
          min_pi_frequency=100, batch_size=256):
    model = SentenceTransformer(base_model)
    output_model_path = os.path.join(output_model_base_path, f"{base_model}_{dataset_name}_{min_pi_frequency}")

    train_dataloader, evaluator = data_preparation.get_data_loader(dataset_name=dataset_name,
                                                                   min_pi_frequency=min_pi_frequency,
                                                                   batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=10,
        evaluation_steps=5000,
        warmup_steps=5000,
        output_path=output_model_path
    )


if __name__ == '__main__':
    train()
