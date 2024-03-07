# Corpus Analysis

The corpus analysis phase explores the underlying characteristics behind laws from South Carolina.

### Most frequent and TF-IDF

`corpus_analysis.ipynb` find the most frequent words (along with their scores) for each year. It also finds the top bi-gram and tri-gram TF-IDF words and scores for those years.

### LDA

`LDA.ipynb` uses an unsupervised method, Latent Dirichlet Allocation (LDA), to build a topic model for the laws.
<br><br>A visualization of an early LDA model is show below,
<br><img src="../images/LDAvis.gif" width="80%" height="80%">

Some explanation for the interactive visualization:
  - In the visualization, the topics are represented by the bubbles and sorted by their size.
  - The blue bars on the right side are the total frequency of the word in the corpus. The red bars give the number of times the term was present in the selected topic.
  - If no topic is chosen, then the blue bars show the most frequent words.
  - In the graph on the left, the distance between the bubbles indicates the uniqueness. For example, the closer the bubbles are to each other, the more similar they are.    
  - <b>Hands On: </b> You can try the latest version of this visualization (which is the `LDA_vis.html` file in this directory) by clicking [this link](https://htmlpreview.github.io/?https://github.com/g-nitin/OnTheBooksUofSC/blob/main/corpus_analysis/LDA_vis.html).
    
    This link takes you to the _htmlpreview.github.io_ host for the visualization HTML file. About `htmlpreview` (from their [GitHub](https://github.com/htmlpreview/htmlpreview.github.com)): _Many GitHub repositories don't use GitHub Pages to host their HTML files. GitHub & BitBucket HTML Preview allows you to render those files without cloning or downloading whole repositories. It is a client-side solution using a CORS proxy to fetch assets._


### LDA Visualizations

  Using the outputs from the LDA model and predicted Jim Crow labels, some visualizations were made to explore the corpus in detail.

  The code for these visualizations is described in the `LDA_visualizations.ipynb` and the final plots are stored in the `imgs` subdirectory.

  Note that the `checkpoint` subdirectory stores all data, such lda models and cleaned sentences.