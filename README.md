# CV

## Certificats 
### DataCamp 
- <a href="https://www.datacamp.com/tracks/python-programmer" target="_blank">Python Programmer</a> (A Python Programmer uses their programming skills to wrangle data and build tools for data analysis)
  - libraries: 
    - numpy, Pandas, re
    - matplotlib (intro)
    - requests, urllib, BeautifulSoup(bs4), json, glob
    - sqlalchemy, sas7bdat, h5py
- <a href="https://www.datacamp.com/tracks/data-analyst-with-python" target="_blank">Data Analyst with Python</a> (A Data Analyst uses data visualization and manipulation techniques to uncover insights and help organizations make better decisions)
  - libraries:
    - (Python Programmer libraries)
    - EDA
    - matplotlib, seaborn
    - some Stats with numpy
- <a href="https://www.datacamp.com/tracks/data-scientist-with-python" target="_blank">Data Scientist with Python</a> (A Data Scientist combines statistical and machine learning techniques with Python programming to analyze and interpret complex data)
  - libraries:
    - (Data Analyst with Python libraries)
    - scipy
    - Bokeh
    - PostgreSQL
    - scikit-learn 
      - Transformer, Pipeline, Preprocessing
      - Classification, Regression, Ensemble, Unsupervised Learning
      - GridSearchCV, Scorer
      - Feature Extraction, Feature Selection
      - PCA, TSNE, NMF
    - Keras (DeepLearning)
- <a href="https://www.datacamp.com/courses/biomedical-image-analysis-in-python" target="_blank">Biomedical Image Analysis in Python</a>
  - The fundamentals of image analysis using NumPy, SciPy, and Matplotlib (whole-body CT scan, segment a cardiac MRI time series, determine whether Alzheimer’s disease changes brain structure)
    - scipy.ndimage, imageio
    - Image Segmentation, Image Transformation
- <a> href="https://www.datacamp.com/courses/nlp-fundamentals-in-python">Natural Language Processing Fundamentals in Python Course</a>
  - The foundation to process and parse text 
    - How to identify and separate words, how to extract topics in a text, and how to build your own fake news classifier
    - How to use basic libraries such as NLTK, alongside libraries which utilize deep learning to solve common NLP problems
  - NLTK, scikit-learn, regex, gensim, spaCy, polyglot, fuzzywuzzy, difflib
### Coursera
- Introduction  to recommender systems: non personalized and content based

### In Progress
- H20
- Spark
- Julia
- TensorFlow

## University
### Bachelor Computer Science (Licence Infomatique) 
- Python (numpy, pandas, scikit-learn), C, PHP, SQL, Bash, Java, HTML, CSS, JS, AJAX, NOSSQL
- Web, AdminSys, AI Data, AI Agent, OOP, DB, Network, DP, Supervised Learning, Reinforcement Learning, UnSupervised Learning
- ***Projects***:
  - **Supervised research (Atelier de recherche encadrée)** [/luluperet/ARE](https://github.com/luluperet/ARE) (L1,Bachelor1) 
    - (Modeling the spread of an epidemic with and without population movement) (Modélisation de la propagation d'une épidémie avec et sans mouvement de la population)
    - python 
    - simple SIR model
    - graphics
  - **Football and strategy (FOOTBALL ET STRATÉGIE)** [/luluperet/2I013](https://github.com/luluperet/2I013)(L2,Bachelor2)  
    - (Programming virtual football players, Artificial intelligence and programming of coherent collective behaviors)
    - python, OOP
    - Agent
    - Design Patterns: Strategy, Adapter, Decorator, State/Proxy, 
    - supervised Learning, reinforcement Learning, genetic algorithm
    - Decision Trees
    - Markov decision process 
    - Q-learning
  - **Web Technologies** [/luluperet/3I017](https://github.com/luluperet/3I017)
    - (Twitter)
    - JAVA, JDBC
    - Servlet - TOMCAT - Web Services
    - HTML, CSS, JS, AJAX 
    - MongoDB, MySQL
    - Map/Reduce
### Master Computer Science (ARTIFICIAL INTELLIGENCE - MACHINE LEARNING FOR DATA SCIENCE)
- C++, Java(J2EE, EJB, JMS), R, Bash, UML(starUML), Scilab, Python (numpy, pandas, scikit-learn)
- Management de projet (Agile), Droit de l'informatique et prop industrielle
- Supervised Learning, Reinforcement Learning, UnSupervised Learning
- Git, Docker
- ***Projects***:
  - **Analyse de Données**[/luluisco/ADD](https://github.com/luluperet/ADD) (ADD) 
    - Analyser de données du jeu de données “villes”, présentant par villes, différentes informations, tel que les salaires moyen de plusieurs métiers, les prix et quelques indicateurs essentiellement économiques.
    - R (FactoMineR, factoextra, corrplot)
    - EDA, PCA
  - **XPLANNER | Web**  [/luluisco/XPLANNER](https://github.com/luluperet/XPLANNER) (Internet et services Web, Applications distribuées)
    - FeedBack Planner 90 est un site web qui a pour but la création de TodoList ( de listes de choses à faire), pour chaque sessions, mois, semaines et jours. 
    Nous pensons que pour bien s’organiser il faut déterminer des sessions, qui sont simplement
trois mois consécutifs.
90 jours étant un nombre de jours, ni trop court et ni trop long, pour bien s’organiser
    - Création d'un site web (FrontEnd, BackEnd) 
    - git, docker, docker-compose et make
    - BackEnd: Spring Boot (gradle, Web Services, JPA, Security,hsqldb)
    - FrontEnt: Front-End: Framework Vue.js (The Progressive JavaScript Framework)
      - npm, webpack, scss, babel, vuex, vue-persistedstate
  -  **Analyse des Sentiments sur AMAZON** [/luluisco/BDAV](https://github.com/luluperet/BDAV) (BDAV)
     - Combinaison des N-grams(unigram, bigram,trigram) et de méthodes ensemblistes  (Naive Bayes, Voting, Stacking)
     - Scraping Sur Amazon (PHP)
     - Préprocessing: Text Mining (scikit-learn, NLTK ) 
        - Tokenization, stop words, encoding, bag of words, Regex ... (nltk)
        - Lemminization -> Treetagger
     - Models: MultinomialNB,VotingClassifier(soft, hard), Stacking
  - **Méthodes de clustering pour des données législatives et judiciaires** (R)
    - Étude comparative de différent algorithmes et critères de choix de nombre de classes (CAH, Kmeans, Skmeans, Tf-Idf, Chi-2, Elbow, Silhouette, Gap )
    - Les données sont des decisions, arrêts des juridictions administratives, et de la Cour de cassation.
      Le but était de trouver toutes les mentions aux articles des différents codes ou des conventions européennes et internationales dans ces textes.
      Nous avons du utiliser divers méthodes pour trouver ces articles qui étaient indiqué de divers façon.
    - Récuperation des données sur data.gouv.fr (JADE, CASS)
    - Preprocessing: Text Mining
      - Bash, Regex, Tokenization, etc ... (script bash et python)
    - Clustering:
      - CAH (min,max,average,ward)
      - kmeans (avec et sans TF-IDF) (avec et sans la metrique du chi-2)
      - skmeans 
      - différentes metrics : 
        - critere du Kaiser (du coude) (wss)
        - silhouette
        - gap
    - Visualisation:
      - Factoextra
