# text_vlm_tesla_v2 — ID-fókuszú workflow

A v2 a **Pacejka vy/r** és **ARX vy/r** identifikáció gyenge pontjait célozza: a v1 fusion + szöveg-only embedding túl sok nem-ID jellegű tartalmat (longitudinal control, pálya-narratíva) kever a vektor térbe.

**v1 érintetlen:** `prompts/`, `texts/tesla_vlm/`, `chroma_db/tesla_vlm/`, meglévő plot PNG-k (csak olvasás).

Belépési pont: [`run_tesla_vlm_redundancy_pipeline_v2.py`](../run_tesla_vlm_redundancy_pipeline_v2.py)

Belső név: **`text_vlm_tesla_v2`**

---

## v1 vs v2

| | v1 | v2 |
|--|----|----|
| Promptok | `prompts/` | `prompts_v2/` (ID-fókusz) |
| Plotok / task | 14 panel, A–D teljes | **10 panel** (longitudinal control, trajectory, wheel speeds, Fx–Fy scatter ki) |
| Összefűzés | VLM fusion → 1 bekezdés | **Nincs fusion** — 4× `technical_summary` bekezdés |
| Embedding | Ollama `qwen3-embedding` (szöveg) | **Qwen3-VL-Embedding-8B** (szöveg + 10 kép) |
| Chroma | `chroma_db/tesla_vlm` | `chroma_db/text_vlm_tesla_v2` |
| Szűrt mat | `data_filtered_mat/text_vlm_tesla_*` | `data_filtered_mat/text_vlm_tesla_v2_*` |

---

## Útvonalak

| Szerep | Path |
|--------|------|
| Promptok | `prompts_v2/` |
| Szövegek | `texts/tesla_vlm_v2/<maneuver>/` (A.json…D.json, description.txt) |
| Chroma | `chroma_db/text_vlm_tesla_v2` / collection `text_vlm_tesla_v2` |
| Cosine | `cosine_similarity_matrices/text_vlm_tesla_v2/` |
| Filtered mat | `data_filtered_mat/text_vlm_tesla_v2_{90,95,98}/` |

---

## Folyamat

```text
Results/maneuver_structured_plots_tesla/<maneuver>/  (read-only)
        → VLM A–D (prompts_v2, 10 plot összesen a taskokban)
        → description.txt = 4 bekezdés (technical_summary)
        → Qwen3-VL-Embedding-8B (text + 10 panel)
        → chroma_db/text_vlm_tesla_v2
        → cosine 90–99, mat export
```

---

## Előfeltételek

- Projekt `.venv` + `uv sync`
- Ollama + `qwen3-vl:32b` (text stage)
- `Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-8B` (embed stage)
- CUDA (8B embed)

---

## Futtatás

```bash
cd /path/to/autoencoder
source .venv/bin/activate

# Smoke (1 manőver)
python -u run_tesla_vlm_redundancy_pipeline_v2.py \
  --maneuvers allando_v_chirp_a1_v10 \
  --stages text embed --overwrite

# Teljes text (screen, ~458×4 VLM)
screen -S text_vlm_v2
python -u run_tesla_vlm_redundancy_pipeline_v2.py --stages text \
  2>&1 | tee logs/text_vlm_v2_text_$(date +%Y%m%d_%H%M%S).log

# Embed + cosine + mat
python -u run_tesla_vlm_redundancy_pipeline_v2.py \
  --stages embed cosine mat

# Minden stage egyszerre
python -u run_tesla_vlm_redundancy_pipeline_v2.py \
  --stages text embed cosine mat
```

Kept tábla / viz (cosine után):

```bash
python run_image_embedding_2b_redundancy.py --preset 8b  # csak ha 8b image DB — v2-höz:
# v2 cosine után: manoeuvres_for_removing_*_text_vlm_tesla_v2.json a matrices mappában
```

V2-specifikus kept összesítéshez a meglévő [`run_image_embedding_2b_redundancy.py`](../run_image_embedding_2b_redundancy.py) mintájára futtatható a v2 matrices dir + `text_vlm_tesla_v2` model name (vagy manuálisan a JSON-okból).

---

## Redundancia után (opcionális)

```bash
python plot_text_embed_maneuvers.py \
  --chroma-path chroma_db/text_vlm_tesla_v2 \
  --collection text_vlm_tesla_v2 \
  --out-dir Results/text_vlm_tesla_v2_viz \
  --overview-only
```

(4096-d vektorok; PCA/t-SNE csak áttekintés.)
