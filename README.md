# Fake news detection based on the induction of fuzzy sets
This git repository shows the work of Giovanni Lagana's final thesis of his master's degree at Università degli Studi di Milano, about the extension of the algorithm described in [1], observing how the choice of input formatting techniques [2] will affect the learning results, focusing on the problem of detecting the reliability score of news published on online newspapers.

References:

[1] Learning Membership Functions for Fuzzy Sets through Modified Support Vector Clustering,
in F. Masulli, G. Pasi e R. Yager (Eds.), Fuzzy Logic and Applications. 10th International Workshop,
WILF 2013, Genoa, Italy, November 19–22, 2013. Proceedings., Vol. 8256, Springer International
Publishing, Switzerland, Lecture Notes on Artificial Intelligence (ISBN 978-3-319-03199-6), 52–59,
2013

[2] Schütze, H., Manning, C. D., & Raghavan, P. (2008). Introduction to information retrieval 
Vol. 39, pp. 1041-4347). Cambridge: Cambridge University Press

# Datasets (updated on 29/06/2020)
Currently, two main interesting datasets about web news have been found.
1. Fake and real news dataset, 111 MB (https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv)
2. All the News 2.0, 9.2 GB (https://components.one/datasets/all-the-news-2-news-articles-dataset/)

# Note (please read this if you want to run the code)
The python code currently assumes that you have a folder "datasets" containing the 2 datasets linked up and, inside this repository which, for space reasons, has been kept out from the versioning.
