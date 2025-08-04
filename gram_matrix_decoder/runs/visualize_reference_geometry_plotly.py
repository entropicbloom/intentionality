"""Interactive 3D visualization of reference gram matrix geometry using Plotly."""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from experiment import build_reference_geometry
from sklearn.neighbors import NearestNeighbors


def create_interactive_visualization(model_config=None):
    """Create interactive 3D visualization using Plotly."""
    if model_config is None:
        # Use trained no-dropout model by default
        model_config = {
            'name': 'dropout',
            'ref_model_type': 'fully_connected_dropout',
            'ref_untrained': False
        }
    
    print(f"Creating interactive visualization for {model_config['name']} model...")
    
    # Build model format string
    ref_untrained_suffix = "-untrained" if model_config['ref_untrained'] else ""
    ref_model_fmt = f"{model_config['ref_model_type']}-{config.REFERENCE_DATASET_TYPE}{ref_untrained_suffix}-hidden_dim_{config.REFERENCE_HIDDEN_DIM}/seed-{{seed}}"
    
    print(f"Model format: {ref_model_fmt}")
    
    # Build reference geometry
    G_ref, C = build_reference_geometry(
        reference_seeds=config.REFERENCE_SEEDS,
        reference_model_fmt=ref_model_fmt,
        basedir=config.BASEDIR,
        all_perms=config.ALL_PERMS
    )
    
    print(f"Reference gram matrix shape: {G_ref.shape}")
    print(f"Number of classes: {C}")
    
    # Perform eigendecomposition (PCA on gram matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(G_ref)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project to 3D using top 3 eigenvectors
    coords_3d = eigenvectors[:, :3] * np.sqrt(eigenvalues[:3])
    
    # Create color palette (ensure we have enough colors)
    colors = px.colors.qualitative.Set1
    if len(colors) < C:
        colors = px.colors.qualitative.Plotly  # Fallback to larger palette
    colors = colors[:C]  # Take only what we need
    
    # Find k-nearest neighbors for edges
    k = 3
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords_3d)
    distances, indices = nbrs.kneighbors(coords_3d)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges with gradient simulation using multiple segments
    edge_traces = []
    for i in range(C):
        for j in range(1, k+1):  # Skip first neighbor (itself)
            neighbor_idx = indices[i, j]
            
            # Create gradient effect by interpolating between points
            n_segments = 10  # Number of segments for smooth gradient
            x_vals = np.linspace(coords_3d[i, 0], coords_3d[neighbor_idx, 0], n_segments)
            y_vals = np.linspace(coords_3d[i, 1], coords_3d[neighbor_idx, 1], n_segments)
            z_vals = np.linspace(coords_3d[i, 2], coords_3d[neighbor_idx, 2], n_segments)
            
            # Create color gradient by interpolating RGB values
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            def rgb_to_hex(rgb):
                return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            
            # Convert colors to RGB for interpolation
            color1_rgb = hex_to_rgb(colors[i])
            color2_rgb = hex_to_rgb(colors[neighbor_idx])
            
            # Create segments with gradually changing colors
            for seg in range(n_segments - 1):
                t = seg / (n_segments - 1)
                # Interpolate color
                interp_rgb = tuple(int(color1_rgb[k] * (1-t) + color2_rgb[k] * t) for k in range(3))
                
                fig.add_trace(go.Scatter3d(
                    x=[x_vals[seg], x_vals[seg+1], None],
                    y=[y_vals[seg], y_vals[seg+1], None],
                    z=[z_vals[seg], z_vals[seg+1], None],
                    mode='lines',
                    line=dict(
                        color=rgb_to_hex(interp_rgb),
                        width=4
                    ),
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Add scatter points (nodes)
    fig.add_trace(go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1], 
        z=coords_3d[:, 2],
        mode='markers+text',
        marker=dict(
            size=12,
            color=colors[:C],
            line=dict(color='black', width=2),
            opacity=1.0
        ),
        text=[str(i) for i in range(C)],
        textposition='middle right',
        textfont=dict(size=14, color='black'),
        name='Digit Classes',
        hovertemplate='<b>Digit %{text}</b><br>' +
                     'X: %{x:.3f}<br>' +
                     'Y: %{y:.3f}<br>' +
                     'Z: %{z:.3f}<extra></extra>'
    ))
    
    # Calculate explained variance
    total_var = np.sum(eigenvalues)
    explained_var_3d = np.sum(eigenvalues[:3]) / total_var
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Reference Gram Matrix Geometry<br><sub>Explained variance: {explained_var_3d:.1%}</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )
    
    print(f"\nExplained variance by top 3 components: {explained_var_3d:.3f}")
    print(f"Individual component ratios: {eigenvalues[:3] / total_var}")
    
    return fig, coords_3d, eigenvalues, eigenvectors


def export_for_jekyll(fig, model_config_name="dropout"):
    """Export Plotly figure as Jekyll-ready HTML include file."""
    
    # Export as HTML string
    html_str = fig.to_html(
        include_plotlyjs='cdn',  # Use CDN for smaller file size
        div_id=f"gram-matrix-{model_config_name}",
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        }
    )
    
    # Clean up the HTML for Jekyll include
    # Remove DOCTYPE and html/body tags, keep just the content
    start_marker = '<div'
    end_marker = '</script>'
    
    start_idx = html_str.find(start_marker)
    end_idx = html_str.rfind(end_marker) + len(end_marker)
    
    clean_html = html_str[start_idx:end_idx]
    
    # Save to file
    output_file = Path(__file__).parent / f'gram_matrix_3d_{model_config_name}.html'
    with open(output_file, 'w') as f:
        f.write(clean_html)
    
    print(f"\nJekyll-ready HTML saved to: {output_file}")
    print(f"\nTo use in Jekyll:")
    print(f"1. Copy {output_file.name} to your Jekyll site's _includes/ folder")
    print(f"2. In your markdown post, add: {{% include {output_file.name} %}}")
    
    return output_file


if __name__ == "__main__":
    print("Interactive Reference Gram Matrix 3D Visualization")
    print("="*60)
    
    # Create visualization
    print("\n1. Creating interactive visualization...")
    fig, coords_3d, eigenvalues, eigenvectors = create_interactive_visualization()
    
    # Export for Jekyll
    print("\n2. Exporting for Jekyll...")
    output_file = export_for_jekyll(fig, "dropout")
    
    # Show the plot
    print("\n3. Opening interactive plot...")
    fig.show()