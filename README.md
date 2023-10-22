# Exploring the Efficacy of Similarity Metrics and Learning to Rank Models in Recommending Friends and Partners 

## Author

ZSHH2

## Abstract

This paper presents an exploration of two key aspects within the context of user recommendations on the popular social networking and dating application, Bumble. The first facet of this study investigates the extent to which similarity metrics can effectively capture population variations based on user profiles. Through qualitative data analysis, the findings indicate that similarity metrics can discern meaningful patterns within the user population. The analysis demonstrates that pairs of female users exhibit higher levels of similarity compared to pairs of male users or mixed-gender pairs. Furthermore, it reveals that substantial age differences between users tend to decrease their overall similarity. Interestingly, the overall user population engaged in the Friend-finding BFF mode of Bumble exhibits a higher degree of similarity to each other compared to the population in the Dating pool where we find higher diversity. These observations are indicative of the potential of similarity metrics to capture user profile distinctions.

The second facet of this research focuses on the efficacy of similarity metrics and learning to rank models in the context of user recommendations on Bumble, specifically with regard to recommending new friends in the Friend-finding BFF mode and potential partners in the Dating mode. The learning to rank models explored in this study encompass a shallow classification-based neural network and a siamese network. The study's findings reveal that, in the task of user recommendations, similarity metrics alone can achieve high performance at this task to a degree that is hard to beat with learning to rank models.. This suggests that similarity metrics can stand alone as effective tools for generating user recommendations on Bumble.

Furthermore, the study highlights a distinction between the two recommendation tasks. While similarity metrics prove effective for suggesting friends, their performance is somewhat less robust in the context of recommending potential romantic partners. This observation aligns with existing literature, which suggests that individuals seeking friendship tend to exhibit greater similarity to one another. 

## Dependencies

To install the requirements for this project run:

```
pip install -r requitements.txt
```

## Structure

### Preprocessing the data

The folder named 'data-preprocessing' contains two files:
* clean_data.py
* create_embeddings.py

These two file need to be run at the begining of the code in the given order in order to clean the data and generate the necessary embeddings. 


### Models

There are three learning to rank models as part of this repository as follows:

**Shallow Neural Network Model**

To train and test the shallow neural network model run the following code:

```
python shallow_neural_net.py [ARG_1] [ARG_2]
```

- `[ARG_1]`: specify either 'bff' or 'dating' to specify which dataset you would like to use
- `[ARG_2]`: speciify either 'one_way' or 'matches' to specify if you would like to evaluate a one way vote or a match


**Siamese Network with Average Embedding Score**

To train and test the siamese network with average embedding model run the following code:

```
python avg_siamese_net.py [ARG_1] [ARG_2]
```

- `[ARG_1]`: specify either 'bff' or 'dating' to specify which dataset you would like to use
- `[ARG_2]`: speciify either 'one_way' or 'matches' to specify if you would like to evaluate a one way vote or a match


**Siamese Network with Set Transformere**

To train and test the siamese network with set transformer as input, first enter the 'set_transfrmer' directory and run the following code:

```
python set_transfrmer_net.py [ARG_1] [ARG_2]
```

- `[ARG_1]`: specify either 'bff' or 'dating' to specify which dataset you would like to use
- `[ARG_2]`: speciify either 'one_way' or 'matches' to specify if you would like to evaluate a one way vote or a match



