This repository is a companion simulation to a paper that extends the school choice model by including transport routes into preference lists.
The simulation uses real data found in "C:\Users\aejp1u19\geo-model\data."

load_data.py reads in the geospacial and statistical data from the data folder and merges them into dataframes.

build_prefs.py samples points (students) from a bivariate normal distribution centred on the population centroids for each LSOA and truncated by the LSOA boundary. It uses the distance between students and schools to build their preferences and priorities, respectively. Students from disadvantaged districts rank with their own performance weight and route discount.

matching.py contains the Deferred Acceptance with Transport (DAT) matching mechanism. It takes student preferences, school priorities, school capacities, and route capacities as input and outputs a stable matching.
matching.md contains more information relevant to the matching paradigm.

dissimilarity.py draws fresh student samples over many seeds, matches each with and without routes, scores every matching with the dissimilarity index and box-plots the index per scenario.

car_displacement.py matches fresh student samples with and without routes and estimates the travel mode of every seated student from the National Travel Survey's mode shares by trip length (data/travel_data/nts), routed students taking their route, then plots the modes of both scenarios and the change between them. The mode estimate itself lives in utils.py and is scored alongside the dissimilarity index, so the sweep carries it too.

__main__.py (`python -m geo_model`) sweeps each model parameter one at a time over the same student samples, scores every setting with and without routes, and plots the index, who the matching leaves unassigned, each school's intake and the change in travel mode against each parameter, singly and as a matrix of subplots. By default the intake panels show each school's term of the dissimilarity index (a_c/A − b_c/B); `--intake share` shows the disadvantaged share of each intake instead. `--map` also maps the first sample's matchings at the default settings, with routes and without: LSOAs coloured by IDACI decile, schools by Progress 8, and a line from every seated student to their school, routed students' lines in their own colour.

utils.py contains all the helper functions used in other scripts.

Coding language is python.
Package manager is uv.
Testing tool is pytest.
Linting and formatting tool is ruff.
Type checker is pyrefly.

NEVER GUESS, NEVER INVENT DATA, NEVER SUPPRESS ERRORS. Do not deviate from instructions.
If anything breaks or doesn't make sense, DO NOT IGNORE OR SUPPRESS IT, report it.

All written code should prioritise accuracy and correctness above all else.
All written code should be as lean and human-readable as possible, no bloat.
