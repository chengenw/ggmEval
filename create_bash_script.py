import numpy as np
np.random.seed(0)

seeds = list(range(5))
permutation_exps = ['mixing-random', 'mode-collapse', 'mode-dropping', 'rewiring-edges']
datasets = ['lobster', 'grid', 'ego', 'community', 'proteins', 'zinc']
use_echo = True

# seeds = [0]

# permutation_exps = ['mode-collapse', 'mode-dropping', 'rewiring-edges']
# permutation_exps = ['mixing-random']
# permutation_exps = ['rewiring-edges']

# datasets = ['lobster', 'grid', 'ego', 'community', 'zinc']
datasets = ['PROTEINS']
# datasets = ['MUTAG']
# datasets = ['REDDIT-MULTI-5K']
# datasets = ['DBLP_v1']

# seeds = [0]
# seeds = list(range(10))

deg_clus_arr = [(False, False), (True, False), (True, True)]
# deg_clus_arr = [(False, False), (True, False)]
# deg_clus_arr = [(False, False)]

# permutation_exps = ['mixing-random', 'mode-collapse']
# deg_clus_arr = [(False, False), (True, False)]

save_perturb = False
load_perturb = False

use_predfined_hyperparams = True

is_parallel = False

def create_commands():
    bash_cmds = ['#!/bin/bash']

    def generate_gnn_commands():
        # gnns = ['graphcl', 'infograph', 'gin-random']
        gnns = ['gin-random', 'graphcl']
        # gnns = ['graphcl']
        # deg_clus_arr = [(False, False), (True, False), (True, True)]
        commands = []
        id = 0
        for exp in permutation_exps:
            for dataset in datasets:
                for gnn in gnns:
                    for deg_feats, clus_feats in deg_clus_arr:
                        results_directory = 'testing'
                        if clus_feats:
                            results_directory += '_clus_feats'
                        elif deg_feats:
                            results_directory += '_deg_feats'
                        else:
                            results_directory += '_no_feats'
                        for seed in seeds:
                            command = f'python main.py --seed={seed} --permutation_type={exp} --dataset={dataset}' \
                                      f' --feature_extractor={gnn} --results_directory={results_directory}'
                            if deg_feats:
                                command += ' --deg_feats'
                            if clus_feats:
                                command += ' --clus_feats --orbit_feats'
                            if save_perturb:
                                command += ' --save_perturb=True'
                            if load_perturb:
                                command += ' --load_perturb=True'
                            if use_predfined_hyperparams:
                                command += ' --use_predefined_hyperparams=True'
                            if is_parallel:
                                command += ' --is_parallel=True'
                            if use_echo:
                                commands += [f'echo "id {id}: {command}"']
                                id += 1
                            commands += [command]
        return commands

    def generate_mmd_commands():
        commands = []
        results_directory = 'testing_no_feats'
        statistics = ['spectral']
        kernel = 'gaussian_rbf'
        for dataset in datasets:
            for statistic in statistics:
                for seed in seeds:
                    for exp in permutation_exps:
                        command = f'python main.py --seed={seed} --permutation_type={exp} --dataset={dataset} --results_directory={results_directory}'
                        command += f' mmd-structure --statistic={statistic} --kernel={kernel}'
                        commands += [command]
        return commands

    def generate_mae_commands(gae='graphmae'):
        # deg_clus_arr = [(False, False), (True, False), (True, True)]
        node_label = False
        commands = []
        id = 0
        for deg_feats, clus_feats in deg_clus_arr:
            results_directory = 'testing'
            if clus_feats:
                results_directory += '_clus_feats'
            elif deg_feats:
                results_directory += '_deg_feats'
            else:
                results_directory += '_no_feats'
            for seed in seeds:
                for exp in permutation_exps:
                    for dataset in datasets:
                        command = f'python main.py --seed={seed} --permutation_type={exp} --dataset={dataset} --results_directory={results_directory}'
                        if deg_feats:
                            command += ' --deg_feats'
                        if clus_feats:
                            command += ' --clus_feats --orbit_feats'
                        command += f' {gae}'
                        if node_label:
                            command += ' --node_label'
                        if use_echo:
                            commands += [f'echo "id {id}: {command}"']
                            id += 1
                        commands += [command]
        return commands

    bash_cmds += generate_gnn_commands()
    bash_cmds += generate_mae_commands()

    # bash_cmds += generate_mmd_commands()
    # bash_cmds += generate_mae_commands('maskgae')

    return bash_cmds

bash_cmds = create_commands()
open('all_commands.sh', 'w').write('\n'.join(bash_cmds))


# if __name__ == '__main__':
#     create_commands()