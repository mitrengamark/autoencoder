# Redundanciaszűrés — greedy set cover (mohó lefedőhalmaz)

Forrás a metodológia / kísérleti fejezethez, ha a dolgozat a pipeline **frekvencia-alapú** szűrése mellett a **greedy set cover** eljárást is bevezeti.

Kód: `greedy_set_cover_redundancy.py`, futtató: `run_greedy_set_cover_redundancy.py`.  
A meglévő `CosineSimilarity.remove_redundancy` (frequency) **nem** módosul; a set cover a már kiszámolt koszinuszmátrixokon fut.

---

## Miért kell második szűrő?

A frekvencia-algoritmus a \(\tau\) feletti párokból addig veszi ki a legtöbb párban szereplő manővert, amíg a **bent maradók** között nem marad ilyen pár. Ez **nem garantálja**, hogy minden *eltávolított* manőverhez maradjon legalább egy megtartott képviselő \(s_{ij}\ge\tau\) hasonlósággal.

A szemantikus lefedettség az eltávolított \(i\)-re:

\[
c_i(\tau)=\max_{j\in\mathcal{K}_{\tau,g}} s_{ij},\qquad i\in\mathcal{R}_{\tau,g}.
\]

Ha \(c_i(\tau)<\tau\), a kieső fedezetlen. A frequency a v1 mátrixokon több küszöbnél (pl. 90, 92–95%) 99,7817%-os lefedést ad (458-ból 1 fedezetlen); a v2-n 92% és 96% ugyanez. A mutató nem igazol fizikai helyettesíthetőséget.

A greedy set cover célja: **kisebb** reprezentáns-halmaz, amelyre a lefedettség **100%**.

---

## Feladat formalizálása

Csoportonként, rögzített \(\tau=t/100\):

- Minden manőver \(i\) lefedőhalmaza: \(C_i=\{i\}\cup\{j:s_{ij}\ge\tau\}\).
- Keressünk minél kisebb \(S\) részhalmazt (kept), amelyre \(\bigcup_{i\in S}C_i\) a teljes csoport.
- A nem \(S\)-beli manőverek a removed halmaz; mindegyikhez van \(i\in S\) \(s\ge\tau\)-val.

Ez klasszikus **halmazfedés** a redundancia-szomszédságokon (dominating / set cover a hasonlósági gráfon). A pontos optimum NP-nehéz; a dolgozat a mohó közelítést használja.

---

## Algoritmus (csoportonként)

1. Minden \(i\)-re előáll \(C_i\).
2. \(U\leftarrow\) a csoport összes indexe (még fedezetlen).
3. Amíg \(U\) nem üres:
   - válaszd azt az még nem kiválasztott \(i\)-t, amelyre \(|C_i\cap U|\) maximális;
   - döntetlennél: a frissen lefedett *többiekhez* mért átlagos koszinusz nagyobb;
   - további döntetlennél: lexikografikusan kisebb manővernév;
   - \(S\leftarrow S\cup\{i\}\), \(U\leftarrow U\setminus C_i\).
4. Kept = \(S\); removed = csoport \(\setminus S\).
5. Minden removed manőverhez rendelődik a leghasonlóbb kept képviselő (\(s\ge\tau\); döntetlennél a név).

A hét manővercsoport **külön** fut; küszöbönként \(t=90,\ldots,99\).

---

## Frequency vs greedy — röviden

| | Frequency (pipeline) | Greedy set cover |
|---|---|---|
| Mit optimalizál | a bent maradók között ne legyen \(\tau\) feletti pár | minden eredeti manőver legyen \(\tau\)-val lefedve |
| Kept méret | általában **nagyobb** | általában **kisebb** |
| \(c_i(\tau)\ge\tau\) minden kiesőre | **nem garantált** | **igen** (konstrukció szerint) |
| Hol fut | `run_tesla_vlm_redundancy_pipeline*.py` cosine stage | `run_greedy_set_cover_redundancy.py` a kész mátrixokon |

Ugyanaz a koszinuszmátrix, más kiválasztási szabály.

---

## Eredmények a Tesla VLM-korpuszon (458)

Forrás: `redundancy_results/greedy_set_cover/` (v1 mátrixok) és  
`redundancy_results/greedy_set_cover_text_vlm_tesla_v2/` (v2 mátrixok);  
összevont ábra: `Results/text_vlm_tesla_v2_viz/kept_vs_threshold_text_vlm_tesla_v1_v2_both_filters.png`.

### Megtartott darabszám

| Kombináció | 90 | 91 | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 + frequency | 35 | 52 | 74 | 111 | 177 | 265 | 383 | 445 | 457 | 458 |
| v1 + greedy | 16 | 18 | 32 | 55 | 107 | 199 | 346 | 443 | 457 | 458 |
| v2 + frequency | 19 | 27 | 40 | 68 | 118 | 221 | 371 | 457 | 458 | 458 |
| v2 + greedy | 7 | 10 | 15 | 23 | 51 | 141 | 329 | 457 | 458 | 458 |

### Lefedettség (`coverage_pct`)

- Frequency: v1-nél több küszöbön 99,7817%; v2-nél 92% és 96% ugyanez.
- Greedy: **100%** minden vizsgált küszöbön mindkét mátrixon.

95%-on példa: v1 frequency 265 vs greedy 199; v2 frequency 221 vs greedy 141. A greedy mindig szigorúbb (vagy egyenlő) keptben, amíg a görbék ~97–98%-on össze nem érnek.

---

## Kimenetek

Küszöbönként (példa v2):

```text
redundancy_results/greedy_set_cover_text_vlm_tesla_v2/threshold_95/
  kept.json
  removed.json
  summary.json
  manoeuvres_for_removing_95_greedy_set_cover.json
```

Összehasonlítás a frequency JSON-nal:

- `comparison_kept_counts.csv`
- `comparison_coverage_metrics.csv`
- `comparison_kept_vs_threshold.png`
- `set_differences.json`

`.mat` export (ha futtatva): `data_filtered_mat/greedy_set_cover_*` és  
`data_filtered_mat/greedy_set_cover_text_vlm_tesla_v2_*`.

---

## Futtatás

```bash
# v1 mátrixokon
python run_greedy_set_cover_redundancy.py \
  --matrices-dir cosine_similarity_matrices/text_vlm_tesla \
  --model-name text_vlm_tesla \
  --results-dir redundancy_results/greedy_set_cover

# v2 mátrixokon
python run_greedy_set_cover_redundancy.py \
  --matrices-dir cosine_similarity_matrices/text_vlm_tesla_v2 \
  --model-name text_vlm_tesla_v2 \
  --results-dir redundancy_results/greedy_set_cover_text_vlm_tesla_v2
```

Embeddinget **nem** számol újra; csak a meglévő `.npy` koszinuszmátrixokat olvassa.

---

## Dolgozatba tehető egy mondat

A mohó lefedőhalmaz-algoritmus a csoporton belüli \(\tau\)-szomszédságokat fedi le: a megtartott részhalmaz kisebb, mint a frekvencia-alapú szűrésé, ugyanakkor minden eltávolított manőverhez garantáltan tartozik legalább egy, a küszöböt elérő hasonlóságú képviselő.
