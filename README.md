# Spectral_Graph_Analysis

<b>Motivation:</b> The project was part of the course "Algorithmic methods of data-mining" at Aalto University, Finland as the part of Master's Programme.
More instructions about the projects can be found in the following <a href="Project_Instructions.pdf">file.</a>

## Introduction

The project aims to perform a graph partitioning algorithm with the use of Spectral Clustering on social network graphs. The goal is to minimize the number of cuts while maintaining cluster size balance.

## Data

The graphs being used for the project are available at Stanford SNAP data.

The following 5 graphs are being used for the analysis:

• ca-GrQc
• Oregon-1
• soc-Epinions1
• web-NotreDame
• roadNet-CA

## Objective function
Graph partitioning is the task of splitting the vertices of a graph into k clusters or communities, V<sub>1</sub> ,..., V<sub>k</sub>, so that v<sub>i</sub> ∩ v<sub>j</sub> = φ in such a way that, there are maximum number of edges within each cluster and very few between the clusters. This is the objective function that we have to minimize.

## Run the code
To run the code:

1. Go to the script directory.
2. Edit the "scripts/main.py" file with the parameters you would to run the script.
3. Execute: "python3 main.py"

## More details

More details about the project, algorithms, and results can be found at <a href="Project_Report.pdf">project report</a>.
