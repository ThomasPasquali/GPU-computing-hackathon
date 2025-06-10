from math import ceil
import os
import json
from operator import itemgetter
import sys
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib

FONT_TITLE = 18
FONT_AXES = 18
FONT_TICKS = 16
FONT_LEGEND = 14

plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)    # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title
# matplotlib.rcParams['font.family'] = 'Noto Color Emoji'

graphs = {
    "wikipedia-20070206": {"N": 3_000_000, "M": 45_000_000, "Category": "Small-diameter"},
    "soc-LiveJournal1": {"N": 5_000_000, "M": 69_000_000, "Category": "Small-diameter"},
    "hollywood-2009": {"N": 1_000_000, "M": 114_000_000, "Category": "Small-diameter"},
    "roadNet-CA": {"N": 1_000_000, "M": 5_000_000, "Category": "Small-diameter"},
    # "GAP-twitter": {"N": 61_000_000, "M": 1_500_000_000, "Category": "Small-diameter"},
    # "GAP-web": {"N": 50_000_000, "M": 1_900_000_000, "Category": "Small-diameter"},

    "GAP-road": {"N": 23_000_000, "M": 58_000_000, "Category": "Large-diameter"},
    "rgg_n_2_22_s0": {"N": 4_000_000, "M": 60_000_000, "Category": "Large-diameter"},
    "europe_osm": {"N": 50_000_000, "M": 108_000_000, "Category": "Large-diameter"},
    "rgg_n_2_24_s0": {"N": 18_000_000, "M": 265_000_000, "Category": "Large-diameter"},

    "graph500_20_8": {"N": 1_000_000, "M": 8_000_000, "Category": "Graph500"},
    "graph500_21_8": {"N": 2_000_000, "M": 17_000_000, "Category": "Graph500"},
    "graph500_20_32": {"N": 1_000_000, "M": 33_000_000, "Category": "Graph500"},
    "graph500_21_16": {"N": 2_000_000, "M": 33_000_000, "Category": "Graph500"},
}


CATEGORIES_LABELS_DICT = {
    'BFS_largeD': 'Large-diameter',
    'BFS_smallD': 'Small-diameter',
    'BFS_g500': 'Graph500',
}

GROUPS_ALIASES_DICT = {
    'GpuComputingOnRustToAnnoyTheProff': 'GCORTATP',
}

# SHARED_DIR = os.environ.get('SHARED_DIR')
# if not SHARED_DIR:
#     print('Environment not set. Exiting...')
#     exit(1)

def plot_ranking(ax: matplotlib.axes.Axes, x, y, title, y_label=''):
    colors = ['gold', 'silver', 'goldenrod'] + ['cornflowerblue'] * (len(x) - 3)
    bars = ax.bar(x, y, color=colors)
    ax.set_title(title)
    if y_label: ax.set_ylabel(y_label)
    ax.grid(True, axis='y')
    ax.set_xticks(range(len(x)), [GROUPS_ALIASES_DICT.get(g, g) for g in x], rotation=45)
    # ax.set_ylim(0.9, 1.5)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2)

    # Add medal emojis to the first three bars
    # for i, bar in enumerate(bars):
    #     if i == 0:
    #         medal = '🥇'
    #     elif i == 1:
    #         medal = '🥈'
    #     elif i == 2:
    #         medal = '🥉'
    #     else:
    #         continue
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         bar.get_height(),
    #         medal,
    #         ha='center',
    #         va='bottom',
    #         fontsize=14
    #     )

# f'{SHARED_DIR}/gpu-computing-hackathon-results.json'
if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} <results-json-file>')
    exit(1)

with open(sys.argv[1], 'r') as sout_file:
    input = sout_file.read()
    # Old for jsonl -- data = [json.loads(line) for line in input.strip().split('\n')]
    data = json.loads(input)
    
    # Filter to only submissions that have results for ALL datasets
    all_datasets = set(graphs.keys())
    data = [
        entry for entry in data
        if 'speedups' in entry and set(entry['speedups'].keys()) == all_datasets
    ]

    # Filter data to keep only the submission with the highest geomean for each group
    grouped_data = {}
    for entry in data:
        group = entry['group']
        if group not in grouped_data or entry['geomean'] > grouped_data[group]['geomean']:
            grouped_data[group] = entry
    data = list(grouped_data.values())

    # Rankings
    ## Global
    ranking_global = sorted(data, key=itemgetter('geomean'), reverse=True)
    ## by Graph type
    ranking_by_type = {}
    for entry in data:
        for category, value in entry['geomeans'].items():
            if category not in ranking_by_type:
                ranking_by_type[category] = []
            ranking_by_type[category].append({'group': entry['group'], 'geomean': value})
    for category in ranking_by_type:
        ranking_by_type[category] = sorted(ranking_by_type[category], key=itemgetter('geomean'), reverse=True)
    ## by Graph
    ranking_by_graph = {}
    for entry in data:
        for dataset, value in entry['speedups'].items():
            if dataset not in ranking_by_graph:
                ranking_by_graph[dataset] = []
            ranking_by_graph[dataset].append({'group': entry['group'], 'speedup': value})
    for dataset in ranking_by_graph:
        ranking_by_graph[dataset] = sorted(ranking_by_graph[dataset], key=itemgetter('speedup'), reverse=True)

    # print(ranking_global)
    # print()
    # print(ranking_by_type)
    # print()
    # print(ranking_by_graph)

    # Plotting
    fig, axes = plt.subplots(ceil(len(ranking_by_graph.keys())/3), 4, figsize=(25, 20))
    axes = axes.flatten()

    # Global ranking plot
    plot_ranking(
        axes[0],
        [entry['group'] for entry in ranking_global],
        [entry['geomean'] for entry in ranking_global],
        'Global Ranking',
        'Speedups Geomean',
    )

    # Ranking by category plots
    ax_i = 1
    for category, ranking in ranking_by_type.items():
        plot_ranking(
            axes[ax_i],
            [entry['group'] for entry in ranking],
            [entry['geomean'] for entry in ranking],
            f'Ranking for {CATEGORIES_LABELS_DICT.get(category, category)}',
        )
        ax_i += 1

    # Ranking by dataset plots
    for dataset, ranking in ranking_by_graph.items():
        plot_ranking(
            axes[ax_i],
            [entry['group'] for entry in ranking],
            [entry['speedup'] for entry in ranking],
            f'Graph: "{dataset}"',
            'Speedup' if ax_i % 4 == 0 else '',
        )
        ax_i += 1

    plt.tight_layout()
    output_path = 'ranking_plot.png'
    plt.savefig(output_path)
    print(f'Ranking plot saved to {output_path}')