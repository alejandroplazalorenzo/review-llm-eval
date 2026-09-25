# Quality gates

Version: `pysentimiento-robertuito+langdet-v1/v1+qwen3:4b/prompt-v1`

| Gate | Result | Detail |
| --- | --- | --- |
| every dated review has month precision | pass | 0 unreadable; 47 of 33038 have no date (month NULL) |
| golden: rows in store == de-duplicated source rows | pass | store 33038, source 33038 |
| golden: mean rating per hotel, SQL == Python | pass | 96 hotels, largest difference 0.0000 |
| every review in scope has layer 1 | pass | 0 missing |
| enriched rows | pass | 200 rows with version pysentimiento-robertuito+langdet-v1/v1+qwen3:4b/prompt-v1 |
| LLM rows computed on the current text | pass | 0 stale |
| `incident` takes more than one value | pass | 6 value(s); 71 row(s) contain a triggering phrase |
| `says_no_return` takes more than one value | pass | 2 value(s); 29 row(s) contain a triggering phrase |
| `legal_action` takes more than one value | pass | 1 value(s); 2 row(s) contain a triggering phrase (too few to require variation) |
| `theft` takes more than one value | pass | 2 value(s); 12 row(s) contain a triggering phrase |
| `illness` takes more than one value | pass | 2 value(s); 9 row(s) contain a triggering phrase |
| `bad_faith` takes more than one value | pass | 2 value(s); 12 row(s) contain a triggering phrase |
| `returning_guest` takes more than one value | pass | 2 value(s); 15 row(s) contain a triggering phrase |
| `with_children` takes more than one value | pass | 2 value(s); 30 row(s) contain a triggering phrase |
| `recommends` takes more than one value | pass | 2 value(s); 30 row(s) contain a triggering phrase |
| `nights` takes more than one value | pass | 2 value(s); 39 row(s) contain a triggering phrase |
| no duplicated opinions | pass | 0 row(s) with a repeated pair |
| literal quotes >= 95% | pass | 99.2% of 609 kept quotes |
| no staff name is a job title | pass | 0 of 171 |
| every staff name is in its review | pass | 0 of 171 |
| alerts on 4-5 star reviews backed by a phrase | pass | 0 of 80 4-5 star reviews flagged; 0 alert(s) without any supporting phrase (allowed 1) |

**All gates pass**
