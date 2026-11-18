# Manual review of 20 category errors (fine-tuned model, test split)

Sample: `error_sample_test.json` (seed fixed, drawn from 3,815 examples without an exact category match).
Judged from the title and candidate list only (no image or description), so the split is approximate.

| # | idx | Title (short) | Verdict | Note |
|---|---|---|---|---|
| 0 | 8758 | 2024 Renewal | model error | title uninformative |
| 1 | 4026 | Sheep Needle Gauge | model error (depth) | stopped one level above the label |
| 2 | 7811 | Battery adhesive tape | questionable label | labelled "Mobile Phone Batteries"; predicted "Adhesives & Tapes" |
| 3 | 9225 | Pressure cooker weight/pin | model error | predicted "Timers", label "Valves" |
| 4 | 4361 | Ferrari 312PB | model error (depth) | went deeper than the label, wrong leaf |
| 5 | 402 | Drop Shot Dailos | model error | shoes vs balls; not decidable from title |
| 6 | 2664 | Maternity tankini | model error (depth) | went one level deeper than the label |
| 7 | 5249 | Magnetic professions box | model error | "Pretend Professions" vs "Magnetic Dolls" |
| 8 | 4981 | Handi-Drink water dispenser | questionable label | "Travel Bowls" vs "Water Dispensers", both defensible |
| 9 | 4217 | Silicone heel pads | questionable label | two near-identical categories in different branches |
| 10 | 9529 | Turn signal | questionable label | labelled "Bicycle Computer RPM Sensors" |
| 11 | 8107 | Volleyball knee pad | questionable label | prediction "Volleyball Knee Pads" is more precise than the label |
| 12 | 8619 | Box spring bed with mattress | questionable label | "Innerspring Mattresses" vs "Box Springs" |
| 13 | 3170 | Mini pocket saxophone | questionable label | labelled "Bass Saxophones"; predicted "Saxophones" |
| 14 | 4884 | Bitters bottle | correct answer absent | label "Bottle Caps"; no candidate fits |
| 15 | 3744 | Irrigation quick coupler valve key | questionable label | "Garden Hose Valves" vs "Sprinkler Valves" |
| 16 | 6028 | NHL All-Star Hockey Game | model error (depth) | went one level deeper than the label |
| 17 | 9241 | Toilettas (toiletry bag) | questionable label | labelled "Toilet Seat Covers"; the prediction is also wrong |
| 18 | 9410 | Baby wipes | model error (depth) | went one level deeper than the label |
| 19 | 2236 | 25187-1 | model error | title uninformative |

Totals: model error 10 (5 of them depth-only: right branch, wrong level), questionable label 9, correct answer absent 1.
