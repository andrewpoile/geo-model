This repository is a companion simulation to a paper that extends the school choice model by including transport routes into preference lists.
The simulation uses real data found in "C:\Users\aejp1u19\geo-model\data."

build_prefs.py reads in the geospacial and statistical data from the data folder and merges them into dataframes. It samples points (students) from a bivariate normal distribution centred on the population centroids for each LSOA and truncated by the LSOA boundary. It uses the distance between students and schools to build their preferences and priorities, respectively.

matching.py contains the Deferred Acceptance with Transport (DAT) matching mechanism. It takes student preferences, school priorities, school capacities, and route capacities as input and outputs a stable matching.

utils.py contains all the helper functions used in other scripts.

Coding language is python.
Package manager is uv.

NEVER GUESS, NEVER INVENT DATA, NEVER SUPPRESS ERRORS. Do not deviate from instructions.
If anything breaks or doesn't make sense, DO NOT IGNORE OR SUPPRESS IT, report it.

All written code should prioritise accuracy and correctness above all else.
All written code should be as lean and human-readable as possible, no bloat.