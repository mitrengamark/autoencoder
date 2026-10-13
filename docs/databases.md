# Szűrt adathalmazok — megmaradt manőverek száma

Universe: **458** manőver (`data_tesla/*.mat`).

`kept = 458 − |manoeuvres_for_removing_<thr>_*.json|`

## Megtartott darabszám küszöbönként (90–99%)


| Módszer                   | 90     | 91     | 92     | 93      | 94      | 95      | 96      | 97      | 98      | 99      |
| ------------------------- | ------ | ------ | ------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| **text_vlm_tesla**        | **35** | **52** | **74** | **111** | **177** | **265** | **383** | **445** | **457** | **458** |
| text_embed_tesla          | 10     | 11     | 11     | 13      | 15      | 16      | 21      | 27      | 47      | 110     |
| text_embed_tesla_dtw      | 11     | 13     | 17     | 17      | 21      | 32      | 49      | 95      | 219     | 377     |
| text_embed_tesla_pairwise | 12     | 14     | 17     | 21      | 31      | 61      | 152     | 342     | 399     | 445     |
| text_embed_tesla_concat   | 11     | 13     | 17     | 18      | 22      | 37      | 65      | 116     | 231     | 378     |




## Materializált `.mat` mappák (`data_filtered_mat/`)

A **`text_vlm_tesla_*`** mappák a removing-JSON kept számokkal egyeznek (`filter_manifest.json` is van).


| Adathalmaz                     | `.mat` darab |
| ------------------------------ | ------------ |
| `text_vlm_tesla_90`            | 35           |
| `text_vlm_tesla_91`            | 52           |
| `text_vlm_tesla_92`            | 74           |
| `text_vlm_tesla_93`            | 111          |
| `text_vlm_tesla_94`            | 177          |
| `text_vlm_tesla_95`            | 265          |
| `text_vlm_tesla_96`            | 383          |
| `text_vlm_tesla_98`            | 457          |
| `text_embed_tesla_90`          | 72*          |
| `text_embed_tesla_95`          | 74*          |
| `text_embed_tesla_98`          | 100*         |
| `text_embed_tesla_concat_90`   | 72*          |
| `text_embed_tesla_concat_95`   | 92*          |
| `text_embed_tesla_concat_98`   | 249*         |
| `text_embed_tesla_dtw_90`      | 72*          |
| `text_embed_tesla_dtw_95`      | 88*          |
| `text_embed_tesla_dtw_98`      | 241*         |
| `text_embed_tesla_pairwise_90` | 72*          |
| `text_embed_tesla_pairwise_95` | 110*         |
| `text_embed_tesla_pairwise_98` | 409*         |


A `text_embed_*` mat-mappák **nem egyeznek** a removing-JSON alapján számolt kepttel (nincs `filter_manifest.json`; valószínűleg régebbi / más export). A megbízható kept-szám a fenti 90–99 tábla.

Identifikációra ajánlott: `text_vlm_tesla_95` **(265)**.