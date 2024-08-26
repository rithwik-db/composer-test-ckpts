import itertools
import subprocess
import time
from mcli.sdk import RunConfig, create_run, wait_for_run_status

 # if not autoresume else 'my-cool-autoresume'
gpu_num=8 # 1
cluster='r1z1'
images = ['mosaicml/pytorch:2.4.0_cu124-python3.11-ubuntu20.04']
composer_versions = ['v0.24.0']


manual_test_integration = {
    'integration_type': 'git_repo',
    'git_repo': 'snarayan21/composer-test-ckpts',
    'git_branch': 'main',
    'path': '/tmp/composer-test-ckpts'
}

fsdp_state_dict_types = ['sharded', 'full']
precisions = ['amp_fp16', 'amp_bf16']
sharding_strategies = ['SHARD_GRAD_OP', 'FULL_SHARD']

for fsdp_state_dict_type, precision, sharding_strategy, image, composer_version in itertools.product(fsdp_state_dict_types,
                                                                            precisions,
                                                                            sharding_strategies, images, composer_versions):
    composer_integration = {
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/composer',
        'git_branch': f'{composer_version}',
        'pip_install': '-e .[all]',
        'path': '/tmp/composer'
    }

    integrations = [manual_test_integration, composer_integration]
    pt_version = image.split(':')[1].split('_')[0]
    cmd = f'pip install pydantic==1.10.12; cd /tmp/composer-test-ckpts && composer -n 2 make_test_ckpt.py {fsdp_state_dict_type} {precision} {sharding_strategy}'
   
    run_name = f"bcompat-{fsdp_state_dict_type}-{precision.split('_')[-1]}-pt-{pt_version.replace('.', '-')}-cp-{composer_version.replace('.', '-')}"
    print(run_name)
    cfg = RunConfig(
        name=run_name,
        gpu_num=gpu_num,
        cluster=cluster,
        image=image,
        integrations=integrations,
        command=cmd,
        scheduling={'priority': 'highest'}
    )
    run = create_run(cfg)
