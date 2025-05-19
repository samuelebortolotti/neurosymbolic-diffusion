from argparse import Namespace

from typing import Tuple

import torch
from torch._C import Size
from datasets.utils.base_dataset import BaseDataset, BOIA_get_loader
from datasets.utils.boia_creation import BOIADataset
from datasets.utils.sddoia_creation import CONCEPTS_ORDER
from backbones.boia_linear import BOIAConceptizer, BOIAConceptizerMLP
from backbones.boia_mlp import BOIAMLP
import time

from expressive.experiments.rsbench.models.sddoiadpl import SDDOIADPL
from torch.nn import functional as F
import torch.nn as nn

class BOIA(BaseDataset):
    NAME = "boia"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.dpl_model = SDDOIADPL(
            None, n_images=1, c_split=(), args=args, model_dict=None, n_facts=21, nr_classes=4, nesy_diff_knowledge=True
        )

    def get_data_loaders(self):
        start = time.time()

        image_dir = "data/bdd2048/"
        train_data_path = "data/bdd2048/train_BDD_OIA.pkl"
        val_data_path = "data/bdd2048/val_BDD_OIA.pkl"
        test_data_path = "data/bdd2048/test_BDD_OIA.pkl"

        self.dataset_train = BOIADataset(
            pkl_file_path=train_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "train",
            n_class_attr=2,
            transform=None,
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
        )
        self.dataset_val = BOIADataset(
            pkl_file_path=val_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "val",
            n_class_attr=2,
            transform=None,
        )
        self.dataset_test = BOIADataset(
            pkl_file_path=test_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "test",
            n_class_attr=2,
            transform=None,
        )

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))

        self.train_loader = BOIA_get_loader(
            self.dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = BOIA_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = BOIA_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.backbone == "neural":
            return BOIAMLP(), None

        return BOIAConceptizer(din=2048, nconcept=21), None

    def get_split(self):
        return 1, ()

    def get_concept_labels(self):
        sorted_concepts = sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get)
        return sorted_concepts

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train))
        print("Validation samples", len(self.dataset_val))
        print("Test samples", len(self.dataset_test))

    def get_w_dim(self) -> Size:
        return (21, 2)
    
    def get_y_dim(self) -> Size:
        return (3, 4)

    def y_from_w(self, w_SBW: torch.Tensor) -> torch.Tensor:
        # Stupid hack: Do full probabilistic inference on deterministic 'distributions' over concepts, then pick argmax
        one_hot_w_SBW2 = F.one_hot(w_SBW, num_classes=2)
        one_hot_w_StBW2 = one_hot_w_SBW2.reshape(-1, w_SBW.shape[-1], 2)
        one_hot_w_StBWt2 = one_hot_w_StBW2.reshape(-1, w_SBW.shape[-1]*2)

        y_StB8 = self.dpl_model.problog_inference(one_hot_w_StBWt2.float())
        y_SB8 = y_StB8
        if len(w_SBW.shape) > 2:
            y_SB8 = y_StB8.reshape(w_SBW.shape[:-1] + (8,))
        fs_SB = torch.max(y_SB8[..., :4], dim=-1)[1]
        l_SB = torch.max(y_SB8[..., 4:6], dim=-1)[1]
        r_SB = torch.max(y_SB8[..., 6:8], dim=-1)[1]

        return torch.stack([fs_SB, l_SB, r_SB], dim=-1)

    def get_backbone_nesydiff(self) -> Tuple[nn.Module, nn.Module]:
        n_concepts = self.get_w_dim()[0]

        # Embedding size + Number of concepts * 3 (true, false, mask)
        return lambda x: x, BOIAConceptizerMLP(din=2048+n_concepts*3, nconcept=n_concepts)
