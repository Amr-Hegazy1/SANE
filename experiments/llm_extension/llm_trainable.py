import logging

import torch

from SANE.models.def_AE_trainable import AE_trainable
from llm_data_utils import build_llm_dataloaders


class LLM_AE_trainable(AE_trainable):
    """
    LLM-specific Trainable that uses bucketed sampling and mask-aware windowing.
    Keeps the rest of the AE_trainable training loop intact.
    """

    def load_datasets(self):
        if "dataset.pt" in str(self.config["dataset::dump"]):
            windowsize = self.config.get("training::windowsize", 128)
            batch_size = self.config["trainset::batchsize"]
            bucket_size = self.config.get("llm::bucket_size", None)
            num_workers = self.config.get("trainloader::workers", 2)

            logging.info("Load LLM dataset with bucketed sampling and windowing")
            dataset = torch.load(self.config["dataset::dump"], weights_only=False)

            (
                trainset,
                testset,
                valset,
                trainloader,
                testloader,
                valloader,
            ) = build_llm_dataloaders(
                dataset_dict=dataset,
                batch_size=batch_size,
                windowsize=windowsize,
                bucket_size=bucket_size,
                num_workers=num_workers,
                drop_last=True,
            )

            return trainset, testset, valset, trainloader, testloader, valloader

        raise NotImplementedError(
            f'could not load dataset from {self.config["dataset::dump"]}'
        )
