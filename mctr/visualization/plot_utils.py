import os
import matplotlib.pyplot as plt

def save_publication_figure(fig, base_path: str, dpi: int = 300):
    """
    Saves a matplotlib figure into PDF, SVG, and PNG formats.
    """
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    # Ensure no extension in base_path
    base_path = os.path.splitext(base_path)[0]
    
    fig.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(f"{base_path}.svg", format='svg', bbox_inches='tight')
    fig.savefig(f"{base_path}.png", format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
