import copy
from dataclasses import dataclass
import os
import pickle
from typing import Literal
import tqdm
import wandb
from wandb.sdk.wandb_run import Run
import click
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
})


ilfm_color = '#120789'
rwfm_color = '#C23D80'
grpo_color = '#FA9E3B'


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

truncated_plasma = truncate_colormap(plt.get_cmap('plasma'), 0.0, 0.96)

################################### RUN NAME PARSING ###################################

def parse_run_name(run_name: str) -> dict[str, str]:
    run_name = '_' + run_name
    run_params = {}
    for key, key_name in run_name_keys.items():
        if f'_{key}' not in run_name:
            continue
        
        value = run_name.split(f'_{key}')[1]
        if key != 'reward':
            value = value.split('_')[0]
        run_params[key_name] = parse_run_name_value(value, key)
        
    return run_params

run_name_keys = {
    'reward': 'Reward',

    'ra': 'RWFM Alpha',
    'cea': 'Collection Explore Amplitude',
    'rvf': 'RWFM Value Function',

    'ga': 'GRPO Alpha',
    'gea': 'GRPO Explore Amplitude',
    'gqf': 'GRPO Reward Surrogate',
}

def parse_run_name_value(value: str, key: str) -> str:
    if key == 'reward':
        assert value.startswith('---') and value.endswith('---')
        value = value[3:-3]
        reward_params = {}
        for param_str in value.split('-'):
            i = [x.isdigit() for x in param_str].index(True)
            reward_params[param_str[:i]] = float(param_str[i:])
        
        if reward_params['final_velocity_weight'] > 0.0:
            return 'Position & Velocity'
        elif reward_params['total_time_weight'] > 0.0:
            return 'Position & Time'
        elif reward_params['pointing_north_weight'] > 0.0:
            return 'Position & Heading'
        elif reward_params['wall_contact_weight'] > 0.0:
            return 'Position & Wall'
        elif reward_params['control_magnitude_weight'] > 0.0:
            return 'Position & Control'
        else:
            return 'Position'

    return value


################################# RESULT EXTRACTION ####################################

@dataclass
class RunResults:
    train_samples: list[int]
    test_performances: list[float]
    test_subcomponent_performances: list[dict[str, float]]
    run_params: dict[str, str]
    run_name: str
    expert_performance: dict[str, float]

def parse_run_results(run: Run) -> RunResults:
    log_datas = list(run.scan_history())

    train_samples = []
    train_sample_key = 'misc/train_samples'
    for log_data in log_datas:
        if train_sample_key not in log_data or log_data[train_sample_key] is None:
            continue
        
        if log_data[train_sample_key] not in train_samples:
            train_samples.append(log_data[train_sample_key])
            
    test_performances = []
    test_subcomponent_performances = []
    for collect_iter in range(len(train_samples)):
        collect_iter_key = f'model_eval_test/collect_iter_{collect_iter}/reward'
        for log_data in log_datas:
            if collect_iter_key not in log_data or log_data[collect_iter_key] is None:
                continue
            
            test_performances.append(log_data[collect_iter_key])
            test_subcomponent_performances.append({
                k.split('/')[-1]: v for k, v in log_data.items() if
                k.startswith(f'model_eval_test/collect_iter_{collect_iter}/') and
                k.endswith('reward')
            })
            break
    
    assert 'expert/test_reward' in log_datas[0]
    expert = {k: v for k, v in log_datas[0].items() if k.startswith('expert/')}

    return RunResults(
        train_samples=train_samples,
        test_performances=test_performances,
        test_subcomponent_performances=test_subcomponent_performances,
        run_params=parse_run_name(run.name),
        run_name=run.name,
        expert_performance=expert,
    )


################################ RESULT MANIPULATION ###################################

def reduce_run_results(run_results: list[RunResults], key: str) -> list[RunResults]:
    groups = {}
    for result in run_results:
        group_params = {k: v for k, v in result.run_params.items() if k != key}
        group_key = tuple(sorted(group_params.items()))
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(result)
    
    reduced_results = []
    for group in groups.values():
        try:
            best_result = max(group, key=lambda x: max(x.test_performances))
        except ValueError:
            import pdb; pdb.set_trace()
        best_result = copy.deepcopy(best_result)
        del best_result.run_params[key]
        reduced_results.append(best_result)
    
    return reduced_results
    
def split_run_results(
    run_results: list[RunResults],
    key: str
) -> dict[str, list[RunResults]]:

    groups = {}
    for result in run_results:
        value = result.run_params[key]
        if value not in groups:
            groups[value] = []

        result = copy.deepcopy(result)
        del result.run_params[key]
        groups[value].append(result)
    
    return groups
    
def filter_run_results(
    run_results: list[RunResults],
    key: str,
    value: str
) -> list[RunResults]:

    assert type(run_results[0].run_params[key]) == str
    filtered = []
    for result in run_results:
        if result.run_params[key] == value:
            result = copy.deepcopy(result)
            del result.run_params[key]
            filtered.append(result)
    return filtered

def get_run_results(sweep_id: str) -> list[RunResults]:
    api = wandb.Api()
    sweep = api.sweep(f'flowmatchingrl/{sweep_id}')
    sweep_runs = sweep.runs

    run_results = []
    for run in tqdm.tqdm(sweep_runs):
        if run.state != 'finished':
            print(f'Run {run.id} is not finished')
            continue
            
        result = parse_run_results(run)
        assert len(result.train_samples) == len(result.test_performances)
        run_results.append(result)
    
    return run_results
    
################################ PLOTTING HELPERS ######################################

def get_plot_label(result: RunResults) -> str:
    label = ''
    for key, value in result.run_params.items():
        label += f'{key}: {value} | '

    return label[:-3]

def get_expert_performance(run_result: RunResults) -> float:
    return run_result.expert_performance['expert/test_reward']

def plot_expert_performance(ax: plt.Axes, expert_performance: float) -> None:
    ax.axhline(
        y=expert_performance,
        color=lighten_color('#000000', 0.25),
        linestyle='--',
        linewidth=1.5,
        label=r'$\pi_D$',
    )
    
def plot_stacked_expert_performance(
    ax: plt.Axes,
    expert_subcomponent_performance: float,
    expert_total_performance: float,
    subcomponent_label: str,
    total_label: str,
) -> None:
    
    ax.axhline(
        y=expert_total_performance,
        color=lighten_color('#000000', 0.25),
        linestyle='--',
        linewidth=1.5,
        alpha=0.5,
        label=total_label,
    )

    ax.axhline(
        y=expert_subcomponent_performance,
        color=lighten_color('#000000', 0.0),
        linestyle='--',
        linewidth=1.5,
        label=subcomponent_label,
    )
    
def plot_stacked_lollipop(
    ax: plt.Axes,
    train_samples: list[int],
    subcomponent_performance: list[float],
    total_performance: list[float],
    subcomponent_label: str,
    total_label: str,
    color: str,
) -> None:

    lightened_color = lighten_color(color, 0.5)
    darkened_color = lighten_color(color, 0.0)
    
    markerline, stemlines, baseline = ax.stem(
        train_samples,
        total_performance,
        label=total_label,
        basefmt=' ',
    )
    plt.setp(stemlines, color=lightened_color, linewidth=1.5)
    plt.setp(
        markerline,
        color=lightened_color,
        markersize=3,
        markerfacecolor=lightened_color,
    )
    
    markerline, stemlines, baseline = ax.stem(
        train_samples,
        subcomponent_performance,
        label=subcomponent_label,
        basefmt=' ',
    )
    plt.setp(stemlines, color=darkened_color, linewidth=1.6, alpha=1.0)
    plt.setp(
        markerline,
        color=darkened_color,
        markersize=6,
        markerfacecolor=darkened_color,
        marker='_',
    )

    
def setup_axes(
    ax: plt.Axes,
    train_samples: list[int],
    title: str,
    include_xlabel: bool = True,
    include_ylabel: bool = True,
) -> None:

    plt.autoscale()
    
    ax.set_ylim(top=0)

    ax.set_title(title)
        
    if include_xlabel:
        ax.set_xlabel('Train trajectories')
    if include_ylabel:
        ax.set_ylabel('Test reward')

    ax.grid(True, which='both', linewidth=0.5)

    ax.set_xticks(train_samples)
    ax.set_xticklabels(
        [
            f'{round(x / 1000)}k' if i % 5 == 0 else ''
            for i, x in enumerate(train_samples)
        ]
    )
    
    ax.tick_params(labelsize=6)
    
LINE_PLOT_ARGS = {
    'marker': 'o',
    'linestyle': '-',
    'markersize': 3,
    'alpha': 1.0,
}

REWARD_VALUES = [
    'Position',
    'Position & Time',
    'Position & Velocity',
    'Position & Wall',
    'Position & Heading',
    'Position & Control',
]

def darken_color(color: str, mix: float) -> str:
    mixed_color = [
        (1 - mix) * a + mix * b for a, b in
        zip(colors.to_rgb(color), colors.to_rgb('#000000'))
    ]
    mixed_color = colors.to_hex(mixed_color)
    return mixed_color
    
def lighten_color(color: str, mix: float) -> str:
    mixed_color = [
        (1 - mix) * a + mix * b for a, b in
        zip(colors.to_rgb(color), colors.to_rgb('#FFFFFF'))
    ]
    mixed_color = colors.to_hex(mixed_color)
    return mixed_color

#################################### PLOTTING ##########################################


def main_plots(
    rwfm_results: list[RunResults],
    grpo_results: list[RunResults],
) -> None:
    
    ilfm_results = filter_run_results(rwfm_results, 'RWFM Alpha', '0.0')
    ilfm_results = filter_run_results(ilfm_results, 'Collection Explore Amplitude', '0.0')
    ilfm_results = filter_run_results(ilfm_results, 'RWFM Value Function', 'False')
    
    rwfm_results = reduce_run_results(rwfm_results, 'RWFM Alpha')
    rwfm_results = reduce_run_results(rwfm_results, 'Collection Explore Amplitude')
    rwfm_results = filter_run_results(rwfm_results, 'RWFM Value Function', 'False')

    grpo_results = reduce_run_results(grpo_results, 'GRPO Alpha')
    grpo_results = reduce_run_results(grpo_results, 'GRPO Explore Amplitude')
    grpo_results = filter_run_results(grpo_results, 'GRPO Reward Surrogate', 'True')
    
    ilfm_split_results = split_run_results(ilfm_results, 'Reward')
    rwfm_split_results = split_run_results(rwfm_results, 'Reward')
    grpo_split_results = split_run_results(grpo_results, 'Reward')
    
    fig, axes = plt.subplots(2, 3, figsize=(6, 4))

    for i, reward_value in enumerate(REWARD_VALUES):
        ilfm_results = ilfm_split_results[reward_value]
        rwfm_results = rwfm_split_results[reward_value]
        grpo_results = grpo_split_results[reward_value]
        
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        train_samples = grpo_results[0].train_samples
        expert_performance = get_expert_performance(grpo_results[0])
        
        for ilfm_result in ilfm_results:
            assert tuple(ilfm_result.train_samples) == tuple(train_samples)
            ax.plot(
                ilfm_result.train_samples,
                ilfm_result.test_performances,
                label='ILFM',
                color=ilfm_color,
                **LINE_PLOT_ARGS,
            )
            assert abs(get_expert_performance(ilfm_result) - expert_performance) < 0.03

        for grpo_result in grpo_results:
            assert tuple(grpo_result.train_samples) == tuple(train_samples)
            ax.plot(
                grpo_result.train_samples,
                grpo_result.test_performances,
                label='GRPO',
                color=grpo_color,
                **LINE_PLOT_ARGS,
            )
            assert abs(get_expert_performance(grpo_result) - expert_performance) < 0.03

        for rwfm_result in rwfm_results:
            assert tuple(rwfm_result.train_samples) == tuple(train_samples)
            ax.plot(
                rwfm_result.train_samples,
                rwfm_result.test_performances,
                label='RWFM',
                color=rwfm_color,
                **LINE_PLOT_ARGS,
            )
            assert abs(get_expert_performance(rwfm_result) - expert_performance) < 0.03
            
        plot_expert_performance(ax, expert_performance)
        setup_axes(
            ax,
            train_samples,
            title=reward_value,
            include_xlabel=row == 1,
            include_ylabel=col == 0,
        )

    lines_labels = axes[0, 0].get_legend_handles_labels()
    permutation = [3, 0, 2, 1]
    lines = [lines_labels[0][i] for i in permutation]
    labels = [lines_labels[1][i] for i in permutation]
    fig.legend(lines, labels, loc='lower center', ncol=4, fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(
        f'output_plotting/plots/main.png',
        dpi=300,
    )


def alpha_sweep_plots(
    rwfm_results: list[RunResults],
) -> None:
    
    rwfm_results = reduce_run_results(rwfm_results, 'Collection Explore Amplitude')
    rwfm_results = filter_run_results(rwfm_results, 'RWFM Value Function', 'False')

    rwfm_split_results = split_run_results(rwfm_results, 'Reward')
    
    fig, axes = plt.subplots(2, 3, figsize=(6, 4))

    for i, reward_value in enumerate(REWARD_VALUES):
        rwfm_results = rwfm_split_results[reward_value]
        rwfm_results.sort(key=lambda result: float(result.run_params['RWFM Alpha']))
        
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        train_samples = rwfm_results[0].train_samples
        
        expert_performance = get_expert_performance(rwfm_results[0])
        
        for j, rwfm_result in enumerate(rwfm_results):
            assert tuple(rwfm_result.train_samples) == tuple(train_samples)
            ax.plot(
                rwfm_result.train_samples,
                rwfm_result.test_performances,
                label=rf'$\alpha={int(float(rwfm_result.run_params["RWFM Alpha"]))}$',
                color=truncated_plasma(j / len(rwfm_results)),
                **LINE_PLOT_ARGS,
            )
            assert abs(get_expert_performance(rwfm_result) - expert_performance) < 0.03
            
        plot_expert_performance(ax, expert_performance)
        setup_axes(
            ax,
            train_samples,
            title=reward_value,
            include_xlabel=row == 1,
            include_ylabel=col == 0,
        )

    lines_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(lines_labels[0], lines_labels[1], loc='lower center', ncol=6, fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(
        f'output_plotting/plots/rwfm_alpha_sweep.png',
        dpi=300,
    )
    

def explore_amplitude_sweep_plots(
    results: list[RunResults],
    method: Literal['rwfm', 'grpo']
) -> None:
    
    alpha_key = 'RWFM Alpha' if method == 'rwfm' else 'GRPO Alpha'
    qv_key = 'RWFM Value Function' if method == 'rwfm' else 'GRPO Reward Surrogate'
    qv_value = 'False' if method == 'rwfm' else 'True'
    explore_amplitude_key = 'Collection Explore Amplitude' if method == 'rwfm' else 'GRPO Explore Amplitude'

    results = reduce_run_results(results, alpha_key)
    results = filter_run_results(results, qv_key, qv_value)

    split_results = split_run_results(results, 'Reward')
    
    fig, axes = plt.subplots(2, 3, figsize=(6, 4))

    for i, reward_value in enumerate(REWARD_VALUES):
        results = split_results[reward_value]
        results.sort(key=lambda result: float(result.run_params[explore_amplitude_key]))
        
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        train_samples = results[0].train_samples
        
        expert_performance = get_expert_performance(results[0])
        
        for j, rwfm_result in enumerate(results):
            assert tuple(rwfm_result.train_samples) == tuple(train_samples)
            ax.plot(
                rwfm_result.train_samples,
                rwfm_result.test_performances,
                label=rf'$M={float(rwfm_result.run_params[explore_amplitude_key])}$',
                color=truncated_plasma(j / len(results)),
                **LINE_PLOT_ARGS,
            )
            assert abs(get_expert_performance(rwfm_result) - expert_performance) < 0.03
            
        plot_expert_performance(ax, expert_performance)
        setup_axes(
            ax,
            train_samples,
            title=reward_value,
            include_xlabel=True,
            include_ylabel=True,
        )

    lines_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(lines_labels[0], lines_labels[1], loc='lower center', ncol=6, fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(
        f'output_plotting/plots/{method}_explore_amplitude_sweep.png',
        dpi=300,
    )

    
def time_optimize_plots(
    rwfm_results: list[RunResults],
    grpo_results: list[RunResults],
) -> None:
    
    ilfm_results = filter_run_results(rwfm_results, 'RWFM Alpha', '0.0')
    ilfm_results = filter_run_results(ilfm_results, 'Collection Explore Amplitude', '0.0')
    ilfm_results = filter_run_results(ilfm_results, 'RWFM Value Function', 'False')
    ilfm_results = filter_run_results(ilfm_results, 'Reward', 'Position & Time')
    
    rwfm_results = reduce_run_results(rwfm_results, 'RWFM Alpha')
    rwfm_results = reduce_run_results(rwfm_results, 'Collection Explore Amplitude')
    rwfm_results = filter_run_results(rwfm_results, 'RWFM Value Function', 'False')
    rwfm_results = filter_run_results(rwfm_results, 'Reward', 'Position & Time')

    grpo_results = reduce_run_results(grpo_results, 'GRPO Alpha')
    grpo_results = reduce_run_results(grpo_results, 'GRPO Explore Amplitude')
    grpo_results = filter_run_results(grpo_results, 'GRPO Reward Surrogate', 'True')
    grpo_results = filter_run_results(grpo_results, 'Reward', 'Position & Time')


    fig, axes = plt.subplots(1, 3, figsize=(6, 2.3))
    
    for col, results in enumerate([ilfm_results, rwfm_results, grpo_results]):
        ax = axes[col]
        method = {0: 'ILFM', 1: 'RWFM', 2: 'GRPO'}[col]
        color = {0: ilfm_color, 1: rwfm_color, 2: grpo_color}[col]
    
        train_samples = results[0].train_samples
        
        expert_performance = get_expert_performance(results[0])
        expert_time_performance = results[0].expert_performance['expert/test_total_time_reward']
        plot_stacked_expert_performance(
            ax=ax,
            expert_subcomponent_performance=expert_time_performance,
            expert_total_performance=expert_performance,
            subcomponent_label=r'$\pi_D$ (time reward)',
            total_label=r'$\pi_D$ (total reward)',
        )
        
        for j, result in enumerate(results):
            assert tuple(result.train_samples) == tuple(train_samples)
            
            test_time_performances = [
                subcomponent_performance['total_time_reward']
                for subcomponent_performance in result.test_subcomponent_performances
            ]
            
            plot_stacked_lollipop(
                ax=ax,
                train_samples=result.train_samples,
                subcomponent_performance=test_time_performances,
                total_performance=result.test_performances,
                subcomponent_label=rf'{method} (time reward)',
                total_label=rf'{method} (total reward)',
                color=color,
            )

            assert abs(get_expert_performance(result) - expert_performance) < 0.03
            
        setup_axes(
            ax,
            train_samples,
            title='',
            include_xlabel=True,
            include_ylabel=col == 0,
        )
        ax.set_ylim(-0.9, 0.0)

    lines, labels = axes[0].get_legend_handles_labels()
    lines += axes[1].get_legend_handles_labels()[0][2:]
    labels += axes[1].get_legend_handles_labels()[1][2:]
    lines += axes[2].get_legend_handles_labels()[0][2:]
    labels += axes[2].get_legend_handles_labels()[1][2:]
    fig.legend(lines, labels, loc='lower center', ncol=4, fontsize=6)
    plt.tight_layout(rect=[0, 0.13, 1, 1])

    plt.savefig(
        f'output_plotting/plots/time_optimize.png',
        dpi=300,
    )

    
def velocity_optimize_plots(
    rwfm_results: list[RunResults],
    grpo_results: list[RunResults],
) -> None:
    
    ilfm_results = filter_run_results(rwfm_results, 'RWFM Alpha', '0.0')
    ilfm_results = filter_run_results(ilfm_results, 'RWFM Value Function', 'False')
    ilfm_results = filter_run_results(ilfm_results, 'Reward', 'Position & Velocity')
    ilfm_results_noexplore = filter_run_results(ilfm_results, 'Collection Explore Amplitude', '0.0')
    ilfm_results_explore = filter_run_results(ilfm_results, 'Collection Explore Amplitude', '0.2')

    rwfm_results = reduce_run_results(rwfm_results, 'RWFM Alpha')
    rwfm_results = filter_run_results(rwfm_results, 'RWFM Value Function', 'False')
    rwfm_results = filter_run_results(rwfm_results, 'Reward', 'Position & Velocity')
    rwfm_results_noexplore = filter_run_results(rwfm_results, 'Collection Explore Amplitude', '0.0')
    rwfm_results_explore = filter_run_results(rwfm_results, 'Collection Explore Amplitude', '0.2')

    grpo_results = reduce_run_results(grpo_results, 'GRPO Alpha')
    grpo_results = filter_run_results(grpo_results, 'GRPO Reward Surrogate', 'True')
    grpo_results = filter_run_results(grpo_results, 'Reward', 'Position & Velocity')
    grpo_results_noexplore = filter_run_results(grpo_results, 'GRPO Explore Amplitude', '0.0')
    grpo_results_explore = filter_run_results(grpo_results, 'GRPO Explore Amplitude', '0.2')
    
    all_results = [
        [ilfm_results_noexplore, rwfm_results_noexplore, grpo_results_noexplore],
        [ilfm_results_explore, rwfm_results_explore, grpo_results_explore],
    ]


    fig, axes = plt.subplots(2, 3, figsize=(6, 4))
    
    for col in range(3):
        for row in range(2):
            ax = axes[row, col]
            results = all_results[row][col]
            method = {0: 'ILFM', 1: 'RWFM', 2: 'GRPO'}[col]
            color = {0: ilfm_color, 1: rwfm_color, 2: grpo_color}[col]
        
            train_samples = results[0].train_samples
            
            expert_performance = get_expert_performance(results[0])
            expert_time_performance = results[0].expert_performance['expert/test_final_velocity_reward']
            plot_stacked_expert_performance(
                ax=ax,
                expert_subcomponent_performance=expert_time_performance,
                expert_total_performance=expert_performance,
                subcomponent_label=r'$\pi_D$ (velocity reward)',
                total_label=r'$\pi_D$ (total reward)',
            )
            
            for j, result in enumerate(results):
                assert tuple(result.train_samples) == tuple(train_samples)
                
                test_velocity_performances = [
                    subcomponent_performance['final_velocity_reward']
                    for subcomponent_performance in result.test_subcomponent_performances
                ]
                
                plot_stacked_lollipop(
                    ax=ax,
                    train_samples=result.train_samples,
                    subcomponent_performance=test_velocity_performances,
                    total_performance=result.test_performances,
                    subcomponent_label=rf'{method} (velocity reward)',
                    total_label=rf'{method} (total reward)',
                    color=color,
                )

                assert abs(get_expert_performance(result) - expert_performance) < 0.03
                
            setup_axes(
                ax,
                train_samples,
                title='',
                include_xlabel=row == 1,
                include_ylabel=col == 0,
            )
            ax.set_ylim(-1.2, 0.0)
            ax.set_title(rf'$M=0.0$' if row == 0 else rf'$M=0.2$')

    lines, labels = axes[0,0].get_legend_handles_labels()
    lines += axes[0,1].get_legend_handles_labels()[0][2:]
    labels += axes[0,1].get_legend_handles_labels()[1][2:]
    lines += axes[0,2].get_legend_handles_labels()[0][2:]
    labels += axes[0,2].get_legend_handles_labels()[1][2:]
    fig.legend(lines, labels, loc='lower center', ncol=4, fontsize=6)
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(
        f'output_plotting/plots/velocity_optimize.png',
        dpi=300,
    )
    

@click.command()
@click.argument('rwfm_sweep_id', type=str)
@click.argument('grpo_sweep_id', type=str)
@click.option('--cache', type=bool, default=False)
def main(rwfm_sweep_id: str, grpo_sweep_id: str, cache: bool) -> None:
    if not os.path.exists('output_plotting'):
        os.makedirs('output_plotting')
    if not os.path.exists('output_plotting/plots'):
        os.makedirs('output_plotting/plots')
        
    if not cache:
        rwfm_sweep_results = get_run_results(rwfm_sweep_id)
        grpo_sweep_results = get_run_results(grpo_sweep_id)
        pickle.dump(
            rwfm_sweep_results,
            open('output_plotting/rwfm_sweep_results.pkl', 'wb'),
        )
        pickle.dump(
            grpo_sweep_results,
            open('output_plotting/grpo_sweep_results.pkl', 'wb'),
        )
    else:
        rwfm_sweep_results = pickle.load(
            open('output_plotting/rwfm_sweep_results.pkl', 'rb'),
        )
        grpo_sweep_results = pickle.load(
            open('output_plotting/grpo_sweep_results.pkl', 'rb'),
        )

    main_plots(rwfm_sweep_results, grpo_sweep_results)
    alpha_sweep_plots(rwfm_sweep_results)
    explore_amplitude_sweep_plots(rwfm_sweep_results, 'rwfm')
    explore_amplitude_sweep_plots(grpo_sweep_results, 'grpo')
    time_optimize_plots(rwfm_sweep_results, grpo_sweep_results)
    velocity_optimize_plots(rwfm_sweep_results, grpo_sweep_results)

if __name__ == "__main__":
    main()