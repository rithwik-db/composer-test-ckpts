from composer.models import ComposerClassifier
import torch
from utils import RandomClassificationDataset, SimpleMLP, MonolithicCheckpointSaver
import itertools
from composer.trainer import Trainer
from torch.utils.data import DataLoader
from composer.utils import dist
import sys
import composer

s3_bucket = 'mosaicml-internal-checkpoints-test' #
if __name__ == "__main__":

    (
        fsdp_state_dict_type, # ['full', 'local', 'sharded']
        precision, # ['amp_fp16', 'amp_bf16']
        sharding_strategy, # ['FULL_SHARD'], 'SHARD_GRAD_OP']
        gpu_num
    ) = sys.argv[1:]

    num_features=32
    num_classes=8
    save_folder=f's3://{s3_bucket}/read_only/elastic_test/{sharding_strategy.lower()}_{fsdp_state_dict_type}_{precision}_{gpu_num}'

    model = SimpleMLP(num_features=num_features, num_classes=num_classes)
    dataset = RandomClassificationDataset(shape=(num_features, ), size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=32)
    optim = torch.optim.Adam(params=model.parameters())
    #rank = 0 if fsdp_state_dict_type == 'full' else '{rank}'
    trainer = Trainer(
        model=model,
        optimizers=optim,
        train_dataloader=dataloader,
        fsdp_config={
            'state_dict_type': fsdp_state_dict_type,
            'sharding_strategy': sharding_strategy,
            'sharded_ckpt_prefix_dir': 'ba{batch}',
        },
        save_folder=save_folder,
        max_duration='2ba',
        save_interval='2ba',
        save_filename='ba{batch}_rank{rank}.pt',
        save_overwrite=False,
        precision=precision,
        #load_path=f'./foo/{sharding_strategy.lower()}_{fsdp_state_dict_type}_{precision}/rank{rank}.pt',
        progress_bar=False,
        log_to_console=False,
        autoresume=False,
        save_num_checkpoints_to_keep=0,
        callbacks=[MonolithicCheckpointSaver(save_folder=save_folder,
                                             batch_interval=2,
                                             filename='mono.pt', keep_optimizers=True) ]
    )
    trainer.fit()
    trainer.close()
