from src.train_gnn import run_experiment

experiments = [
    ("corr", True),
    ("js", True),
]

all_results = []

for graph_type, use_edge_attr in experiments:
    exp_name = f"{graph_type}_edge_{use_edge_attr}"

    result = run_experiment(
        graph_type=graph_type,
        use_edge_attr=use_edge_attr,
        exp_name=exp_name
    )

    all_results.append(result)

print("\nFINAL RESULTS:")
for r in all_results:
    print(r)