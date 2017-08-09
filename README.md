# Content-Based Recommender system
Algorithms for articles retrieval with different features like wordvec and tf-idf features to recommend articles with nearest neighbour models to recommend the best articles for the reader to read.

Also unsupervised Algorithms for Authorship Identification based on deep features of the article written styles by the authors. 

# Deep Features for author writing styles

## Lexical and punctuation features

    Lexical features:
        The average number of words per sentence
        Sentence length variation
        Lexical diversity, which is a measure of the richness of the author’s vocabulary
    
    Punctuation features:
        Average number of commas, semicolons and colons per sentence


## Bag of Words features

Our second feature set is Bag of Words, which represents the frequencies of different words in each chapter. This feature vector is commonly used for text classification. However, unlike text classification, we need to use in topic independent keywords (aka “function words”) since each author is writing on a variety of subjects. Our vocabulary will be the most common words across all chapters (e.g. words like ‘a’, ‘is’, ‘the’, etc.). The idea is that the authors use these common words in a distinctive, but consistent, manner.

## Syntactic features

For our final feature set, we extract syntactic features of the text. Part of speech (POS) is a classification of each token into a lexical category (e.g. noun). NLTK has a function for POS labeling, and our feature vector is comprised of frequencies for the most common POS tags
