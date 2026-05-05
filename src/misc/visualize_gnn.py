import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

from src.utils import PROCESSED_DIR

def load_data():
    """Carga todos los archivos reales del pipeline."""
    ticker_to_node = pd.read_parquet(PROCESSED_DIR / "ticker_to_node.parquet")
    corr_edges = pd.read_parquet(PROCESSED_DIR / "graph_corr_edges.parquet")
    js_edges = pd.read_parquet(PROCESSED_DIR / "graph_div_edges.parquet")
    snapshots = pd.read_parquet(PROCESSED_DIR / "gnn_snapshots_index.parquet")
    X_gnn = np.load(PROCESSED_DIR / "X_gnn.npy")
    y_gnn = np.load(PROCESSED_DIR / "y_gnn.npy")
    
    # Metadata adicional de panel para sector y market_cap
    panel = pd.read_parquet(PROCESSED_DIR / "panel_with_splits.parquet")
    latest_panel = panel.groupby('ticker').tail(1)[['ticker', 'sector', 'market_cap']]
    
    # Merge metadata en nodes
    nodes = ticker_to_node.merge(latest_panel, on='ticker')
    nodes['market_cap_scaled'] = (nodes['market_cap'] - nodes['market_cap'].min()) / (nodes['market_cap'].max() - nodes['market_cap'].min())
    
    # Layout fijo para todos los grafos (spring_layout una vez)
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes['node_id'])
    all_edges = pd.concat([corr_edges[['src', 'dst']], js_edges[['src', 'dst']]]).drop_duplicates()
    G.add_edges_from(all_edges.values)
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    nodes[['x', 'y']] = [pos[nid] for nid in nodes['node_id']]
    
    # Snapshots con targets y retorno medio de ventana
    snapshot_data = []
    for i, row in snapshots.iterrows():
        target_i = y_gnn[i]
        mean_ret_i = X_gnn[i, :, -1, 2].mean(axis=1)  # log_ret_1d último timestep, media ventana
        snapshot_data.append({
            'date': row['date'], 'split': row['split'],
            'target': target_i.tolist(), 'mean_ret': mean_ret_i.tolist()
        })
    
    return {
        'nodes': nodes.to_dict('records'),
        'graphs': {
            'corr': {'edges': corr_edges.to_dict('records')},
            'js': {'edges': js_edges.to_dict('records')}
        },
        'snapshots': snapshot_data
    }

def create_gnn_dashboard():
    """Genera la app HTML conectada a datos reales."""
    data = load_data()
    
    html_template = Path('src/templates/gnn_dashboard.html').read_text()  # ver abajo
    html_filled = html_template.replace('__DATA__', json.dumps(data))
    
    out_path = Path("visualizations/gnn_dashboard.html")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html_filled, encoding='utf-8')
    print(f"Saved dashboard to {out_path}")
    print("You may open this file in a web browser to explore the GNN data visualization.")
    
    return str(out_path)

if __name__ == "__main__":
    create_gnn_dashboard()