import os

def save_figure(fig, save_path, dpi=300):
    """
    Általános függvény a Matplotlib ábrák mentésére.

    :param fig: A Matplotlib figura, amit el szeretnénk menteni.
    :param save_path: Az elmentendő fájl elérési útja (pl. 'output/plot.png').
    :param dpi: A kép felbontása (alapértelmezett: 300 DPI).
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Létrehozza a mappát, ha nem létezik
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Kép elmentve: {save_path}")
