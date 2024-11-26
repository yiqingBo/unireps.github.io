---
layout: distill
title: Evaluating Representational Similarity Measures from the Lens of Functional Correspondence
description:
tags:
giscus_comments: true
date: 2024-11-20
featured: true

# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

authors:
  - name: Yiqing Bo
    url: "mailto:ybo@ucsd.edu"
    affiliations:
      name: "Department of Cognitive Science, UC San Diego"
  - name: Ansh Soni
    url: "mailto:anshsoni@sas.upenn.edu"
    affiliations:
      name: "Department of Psychology, University of Pennsylvania"
  - name: Sudhanshu Srivastava
    url: "mailto:sus021@ucsd.edu"
    affiliations:
      name: "Department of Cognitive Science, UC San Diego"
  - name: Meenakshi Khosla
    url: "mailto:mkhosla@ucsd.edu"
    affiliations:
      name: "Department of Cognitive Science, UC San Diego; Department of Computer Science, UC San Diego"


# bibliography: 2018-12-22-distill.bib
bibliography: 2024-11-20-downstream-functional-correspondence.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Equations
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Interactive Plots
  - name: Layouts
  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---


# Introduction
<d-cite key="yamins2014performance"></d-cite> 
d-cite key="gregor2015draw"></d-cite> 
Both neuroscience and artificial intelligence (AI) confront the
challenge of high-dimensional neural data, whether from neurobiological
firing rates, voxel responses, or hidden layer activations in artificial
networks. Comparing such high-dimensional neural data is critical for
both fields, as it facilitates understanding of complex systems by
revealing their underlying similarities and differences.

In neuroscience, one of the main goals is to uncover how neural activity
drives behavior and to understand neural computations at an algorithmic
level. Comparisons across species and between brain and model
representations, particularly those of deep neural networks, have been
instrumental in advancing this
understanding ([@yamins2014performance; @eickenberg2017seeing; @gucclu2015deep; @cichy2016comparison; @KriegeskorteRSA; @schrimpf2018brain; @schrimpf2020integrative; @storrs2021diverse; @kriegeskorte2008representational]).
A growing interest lies in systematically altering model
parameters---such as architecture, learning objectives, and training
data---and comparing the resulting internal representations with neural
data ([@yamins2016using; @doerig2023neuroconnectionist; @schrimpf2018brain; @schrimpf2020integrative]).

Similarly, in AI, researchers are increasingly focused on
reverse-engineering neural networks by tweaking architectural
components, training objectives, and data inputs to examine how these
modifications impact the resulting representations. However, studying
neural networks in isolation can be limiting, as interactions between
the learning algorithms and structured data shape these systems in ways
we do not yet fully understand. Comparative analysis of model
representations offers a powerful tool to probe these networks more
deeply. This endeavor is rooted in the universality hypothesis that
similar phenomena can arise across different networks. Indeed, a large
number of studies have provided empirical evidence licensing these
universal
theories ([@huh2024platonic; @kornblith2019similarity; @bansal2021revisiting; @li2015convergent; @roeder2021linear; @lenc2015understanding])
but the extent to which diverse neural networks converge to similar
representations is not well understood.

Given the growing interest in comparative analyses across neuroscience
and AI, a key question arises: what are the best tools for conducting
such analyses? Over the past decade, a wide variety of approaches have
emerged for quantifying the representational similarity across
artificial and biological neural
representations ([@gettingaligned; @klabunde2023similarity; @williams2021generalized]).
Most of these approaches can be classified as belonging to one of four
categories: representational similarity based measures, alignment-based
measures, nearest-neighbor based measures and canonical correlation
analysis-based measures ([@klabunde2023similarity]). With the wide range
of available approaches for representational comparisons, researchers
are tasked with selecting a suitable metric. The choice of a specific
metric implicitly prioritizes certain properties of the system, as
different approaches emphasize distinct invariances and are sensitive to
varying aspects of the representations. This complexity ties into
broader issues in the concept and assessment of similarity, which, as
emphasized in psychology, is highly
context-dependent ([@tversky1977features]).

What, then, are the key desiderata for network comparison metrics?
Networks may exhibit similarities in some dimensions and differences in
others, but the critical question is whether these differences are
functionally relevant or merely reflect differences in origin or
construction. This consideration leads to a central criterion for
effective metrics: behavioral differences should correspond to
differences in internal representational similarity ([@cao2022putting]).
However, identifying which measures reliably capture behaviorally
meaningful differences remains an open question.

Our study aims to address the above challenge. Here, we make the
following key contributions:

-   We conduct an extensive analysis of common representational
    comparison measures (including alignment-based, representational
    similarity matrix-based, CCA-based, and nearest-neighbor-based
    methods) and show that these measures differ in their capacity to
    distinguish between models. While some measures excel at
    distinguishing between models from different architectural families,
    others are better at separating trained from untrained models.

-   To assess which of these distinctions reflects differences in model
    behaviors, we perform complementary behavioral comparisons using a
    comprehensive set of behavioral metrics (both hard and soft
    prediction-based). We find that behavioral metrics are generally
    more consistent with each other than representational similarity
    measures.

-   Finally, we cross-compare representational and behavioral similarity
    measures, revealing that linear CKA and Procrustes distance align
    most closely with behavioral evaluations, whereas metrics like
    linear predictivity, widely used in neuroscience, show only modest
    alignment. This finding offers important guidance for metric
    selection in neuroAI, where the functional relevance of
    representational comparisons is paramount.

## Related Work

Although few studies directly compare representational similarity
measures based on their discriminative power, most efforts in this area
focus on identifying metrics that distinguish between models by their
construction. These efforts typically involve assessing measures based
on their ability to match corresponding layers across models with
varying seeds  or identical architectures with different
initializations . The closest to our work are studies by Ding et al. 
and Cloos et al. . Cloos et al.  optimized synthetic datasets to
resemble brain activity under various measures, demonstrating that
metrics like linear predictivity and CKA can yield high scores even when
task-relevant variables are not encoded. Ding et al. () examined the
sensitivity of representational similarity measures—CCA, CKA, and
Procrustes—in BERT models (NLP) and ResNet models (CIFAR-10) to factors
that either preserve functional behavior (e.g., random seed variations)
or alter it (e.g., principal component deletion). However, these studies
examine a limited set of similarity measures and primarily assess
functional similarity based on task performance alone, without
evaluating the finer-grained alignment of predictions across models.

## Metrics for Representational Comparisons

#### Notations and Definitions

Let *S* be a set of *M* fixed input stimuli. Define the kernel functions
[^1] *f* : *S* → ℝ<sup>*N*<sub>*X*</sub></sup> and
*g* : *S* → ℝ<sup>*N*<sub>*Y*</sub></sup>, where *N*<sub>*X*</sub> and
*N*<sub>*Y*</sub> are the output unit sizes of the first and second
encoders, respectively. Here, *f*(*s*<sub>*i*</sub>) and
*g*(*s*<sub>*i*</sub>) map each stimulus *s*<sub>*i*</sub> ∈ *S* to
vectors in ℝ<sup>*N*<sub>*X*</sub></sup> and
ℝ<sup>*N*<sub>*Y*</sub></sup>.

Let *X* ∈ ℝ<sup>*M* × *N*<sub>*X*</sub></sup> and
*Y* ∈ ℝ<sup>*M* × *N*<sub>*Y*</sub></sup> be the representation
matrices. For each input stimulus *s*<sub>*i*</sub>, denote the *i*th
row of *X* as *ϕ*<sub>*i*</sub> = *f*(*s*<sub>*i*</sub>) and of *Y* as
*ψ*<sub>*i*</sub> = *g*(*s*<sub>*i*</sub>), each being the activation in
response to the *i*th stimulus.

#### Representational Similarity Analysis (RSA)

  A method that quantifies the distance between *M* × *M*
Representational Dissimilarity Matrices (RDMs) of two models in response
to a common set of *M* stimuli.

RSA(*X*, *Y*) = *τ*(**J**<sub>*M*</sub> − *X*<sup>*T*</sup>*X*, **J**<sub>*M*</sub> − *Y*<sup>*T*</sup>*Y*)

with *J*<sub>*M*</sub> denoting the *M* × *M* all-ones matrix, the
representational dissimilarity matrices (RDMs) for *X* and *Y* are
*J*<sub>*M*</sub> − *X*<sup>*T*</sup>*X* and
*J*<sub>*M*</sub> − *Y*<sup>*T*</sup>*Y*, respectively.
*X*<sup>*T*</sup>*X* and *Y*<sup>*T*</sup>*Y* in ℝ<sup>*M* × *M*</sup>
represent the self-correlations of *X* and *Y*, with each matrix entry
*i*, *j* quantifying the correlation between activations for the
*i*<sup>*t**h*</sup> and *j*<sup>*t**h*</sup> stimuli. The Kendall rank
correlation coefficient *τ*(⋅) quantifies the similarity between these
RDMs.

#### Canonical Correlation Analysis (CCA)

  A popular linear-invariant similarity measure quantifying the
multivariate similarity between two sets of representations *X* and *Y*
under a shared set of *M* stimuli by identifying the bases in the unit
space of matrix *X* and *Y* such that when the two matrices are
projected on to these bases, their correlation is maximized.  
Here, the *i*<sup>*t**h*</sup> canonical correlation coefficient
*ρ*<sub>*i*</sub> (associated with the *i*<sup>*t**h*</sup> optimized
canonical weights
*w*<sub>*x*</sub><sup>*i*</sup> ∈ ℝ<sup>*N*<sub>*X*</sub></sup> and
*w*<sub>*y*</sub><sup>*i*</sup> ∈ ℝ<sup>*N*<sub>*Y*</sub></sup>) is
being calculated by:
*ρ*<sub>*i*</sub> = max<sub>*w*<sub>*x*</sub><sup>*i*</sup>, *w*<sub>*y*</sub><sup>*i*</sup></sub>corr(*X**w*<sub>*x*</sub><sup>*i*</sup>, *Y**w*<sub>*y*</sub><sup>*i*</sup>)
subject to ∀*j* \< *i*,  *X**w*<sub>*x*</sub><sup>*i*</sup> ⊥ *X**w*<sub>*x*</sub><sup>*j*</sup>  and  *Y**w*<sub>*y*</sub><sup>*i*</sup> ⊥ *Y**w*<sub>*y*</sub><sup>*j*</sup>,
with the transformed matrices *X**w*<sub>*x*</sub><sup>*i*</sup> and
*Y**w*<sub>*y*</sub><sup>*i*</sup> being called canonical variables.

To obtain a measure of similarity between neural network
representations, the mean CCA correlation coefficient *ρ̄* over the first
*N*′ components is reported, with
*N*′ = min (*N*<sub>*X*</sub>, *N*<sub>*Y*</sub>). Here,

$$\bar{\rho} = \frac{\sum\_{i=1}^{N'} \rho_i}{N'} = \frac{\left\\ Q_Y^T Q_X \right\\\_\*}{N'},$$
where ∥ ⋅ ∥<sub>\*</sub> denotes the nuclear norm. Here,
*Q*<sub>*X*</sub> = *X*(*X*<sup>*T*</sup>*X*)<sup>−1/2</sup> and
*Q*<sub>*Y*</sub> = *Y*(*Y*<sup>*T*</sup>*Y*)<sup>−1/2</sup> represent
any orthonormal bases for the columns of *X* and *Y*.

#### Linear Centered Kernel Alignment (CKA)

 

A representation-level comparison that measures how (in)dependent the
two models’ RDMs are under a shared set of *M* stimuli. This measure
possesses a weaker invariance assumption than CCA, being invariant only
to orthogonal transformations, rather than all classes of invertible
linear transformations, which implies the preservation of scalar
products and Euclidean distances between pairs of stimuli.
$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \text{HSIC}(L, L)}} 
%= \frac{\\Y^T X\\\_F^2}{\\X^T X\\\_F \\Y^T Y\\\_F}$$
with *K* and *L* be kernel matrices where
*K*<sub>*i**j*</sub> = *κ*(*ϕ*<sub>*i*</sub>, *ϕ*<sub>*j*</sub>) and
*L*<sub>*i**j*</sub> = *κ*(*ψ*<sub>*i*</sub>, *ψ*<sub>*j*</sub>). These
matrices represent the inner products of vectorized features *ϕ* and *ψ*
from two different models, respectively, computed using the kernel
function function *κ*. In the linear case, *κ* is the inner product,
implying *K* = *X**X*<sup>*T*</sup>, *L* = *Y**Y*<sup>*T*</sup>. The
Hilbert-Schmidt Independence Criterion HSIC(⋅) evaluates the
cross-covariance of the models’ internal embedding spaces, focusing on
the similarity of stimulus pairs.

#### Mutual k-nearest neighbors

  A local-biased representation-level measure that quantifies the
similarity between the representations of two models by assessing the
average overlap of their nearest neighbor sets for corresponding
features.
$$\text{MNN}(\phi_i, \psi_i) = \frac{1}{k} \|S(\phi_i) \cap S(\psi_i)\|$$
where *ϕ*<sub>*i*</sub> = *f*(*s*<sub>*i*</sub>) and
*ψ*<sub>*i*</sub> = *g*(*s*<sub>*i*</sub>) are features derived from
model representations *f* and *g* given the shared stimulus
*s*<sub>*i*</sub>. *S*(*ϕ*<sub>*i*</sub>) and *S*(*ψ*<sub>*i*</sub>) are
the set of indices of the *k*-nearest neighbors of *ϕ*<sub>*i*</sub> and
*ψ*<sub>*i*</sub> in their respective feature spaces and \|⋅\| is the
size of the intersection.

#### Linear predictivity

An asymmetric measure of alignment between the representations of two
systems, obtained using ridge regression. The numerical score is
calculated by summing Pearson’s correlations between each pair of
predicted and actual activations in the held-out set. For reporting, we
provide symmetrized scores by averaging the correlation coefficients
from both fitting directions.

#### Procrustes distance

  A rotational-invariant shape alignment distance between *X* and *Y*’s
representations after removing the components of uniform scaling and
translation and applying an optimized mapping, where the mappings from
one representation matrix to another is constrained to rotations and
reflection. Here, the Procrustes distance is given by:

*d*(*X*, *Y*) = min<sub>*T* ∈ *O*(*n*)</sub>∥*ϕ*(*X*) − *ϕ*(*Y*)*T*∥<sub>*F*</sub>
where *ϕ*(⋅) is the function that whitens the covariance of the matrix X
and Y, i.e. the columns sum to zero and
∥*ϕ*(*X*)∥<sub>*F*</sub>, ∥*ϕ*(*X*)∥<sub>*F*</sub> = 1. *O*(*n*) is the
orthogonal group.

The similarity scores reported are obtained by 1 − *d*(*X*, *Y*), such
that the comparison with a representation itself yields a score of 1,
and lower distance yields a higher score.

#### Semi-matching score

  An asymmetric correlation-based measure obtained using the average
correlation after matching every neuron in *X* to its most similar
partner in *Y*. The scores reported are the average from both fitting
directions.
$$s_{semi}(X, Y) = \frac{1}{N_x} \sum_{i=1}^{N_x} \max_{j \in \\1, \dots, N_y\\} x_i^\top y_j$$

#### Soft-matching distance

  A generalization of permutation distance  to representations with
different number of neurons. It measures alignment by relaxing the set
of permutations to “soft permutations”. Specifically, consider a
nonnegative matrix
**P** ∈ ℝ<sup>*N*<sub>*x*</sub> × *N*<sub>*y*</sub></sup> whose rows
each sum to 1/*N*<sub>*x*</sub> and whose columns each sum to
1/*N*<sub>*y*</sub>. The set of all such matrices defines a
*transportation polytope* , denoted as
T(*N*<sub>*x*</sub>, *N*<sub>*y*</sub>). Optimizing over this set of
rectangular matrices results in a “soft matching” or “soft permutation”
of neuron labels in the sense that every row and column of **P** may
have more than one non-zero element.
$$d\_{\mathrm{T}}(X, Y) = \sqrt{\min\_{P \in \mathrm{T}(N_X, N_Y)} \sum\_{i,j} P\_{ij} \\x_i - y_j\\^2}$$

## Downstream Behavioral Measures

<figure id="figs:fig1">
<img src="figs/schematic_rep_eval.png" />
<figcaption>Framework for evaluating representational similarity metrics
based on their functional correspondence. We conduct pairwise
comparisons of the representational similarities and behavioral outputs
of 19 vision models, utilizing 9 widely-used representational similarity
measures and 10 behavioral metrics across 17 distinct behavioral
datasets.</figcaption>
</figure>

For classification tasks, we incorporate various downstream measurements
at different levels of granularity to assess behavioral consistency
across systems. For a given pair of neural networks, their activations
over a shared set of stimuli are extracted. A linear readout based on a
fully connected layer is trained over a training set of activations,
where the resulting behavioral classification decisions determined by
the linear readouts on a held-out testing set are exploited in the
following ways as a comparison between the neural networks:

### Raw Softmax alignments

emphasize the consistency of numerical class-level activation strength
patterns. Compares two models’ representations by their linear-readout’s
softmax layer activation, which is a class-dimensional vector reflecting
the model’s judgement of the probabilities assigned to each label for a
given input, with scores calculated by summing the Pearson correlation
coefficient between these softmax vectors over the testing set.

### Classification Confusion Matrix alignments

emphasize the consistency of discrete inter-class (mis) classification
patterns. A similarity score is obtained by comparing the two models’
confusion matrices in the following ways:

1.  **Pearson Correlation Coefficient between the flattened confusion
    matrices given by two models, each being a vector of dimension
    *C*<sup>2</sup> over *C* classes.**

2.  **Jensen-Shannon (JS) Distance  introduced as a behavioral alignment
    measure by  is functionally similar to a symmetrized and smoother
    version of the Kullback-Leibler (KL) divergence. For class-wise JS
    distance, let
    *p̂* = ⟨*p*<sub>1</sub>, *p*<sub>2</sub>, …, *p*<sub>*C*</sub>⟩ and
    *q̂* = ⟨*q*<sub>1</sub>, *q*<sub>2</sub>, …, *q*<sub>*C*</sub>⟩ be
    error probability vectors over C classes, with
    $$p_i = \frac{e_i}{\sum\_{i=1}^C e_i}, \forall i \in \\1,2,...,C\\$$
    where *e*<sub>*i*</sub> represents error counts per class. The JS
    divergence is defined as:
    $$JSD(p, q) = \sqrt{\frac{D(p \|\| m) + D(q \|\| m)}{2}},$$
    $$\text{with } D(p \|\| m) = \sum\_{i=1}^C p_i \log\left(\frac{p_i}{m_i}\right)\text{ and  }  m_i = \frac{p_i + q_i}{2}$$
    **

    A finer inter-class dissimilarity measure derived from the complete
    misclassification patterns shown in the non-diagonal elements of the
    confusion matrix results in two *C* \* (*C* − 1) dimensional
    flattened vectors *p̂* and *q̂*, where each component is proportional
    to the counts of misclassifications from class *i* to class *j*, is
    calculated as
    $$\frac{e\_{ij}}{\sum\_{i=1}^C \sum\_{j=1, j \neq i}^C e\_{ij}}, \quad \forall i, j \in \\1, 2, \ldots, C\\$$
    .

    The resulting distances from both method range from \[0, 1\], where
    we simply report a similarity measure given by
    1 − *JSD*(*p*, *q*).

### Classification Binary Correctness alignments

emphasize consistency in per-stimulus prediction correctness. The error
patterns for each model are encoded as vectors of binary values, where
each entry corresponds to the correctness of a stimulus’s prediction. We
incorporate the following measures to compare alignment between the
binary vectors:  

1.  **Pearson Correlation Coefficient** between the two binary vectors of
    dimension *M* over *M* shared testing stimuli, reflecting the
    prediction correctness of two models (1 = correct, 0 = incorrect).

2.  **Cohen’s *κ* Score** Consider two systems tested independently on
    identical trials, each correctly classifying with a probability
    *p*<sub>*c**o**r**r**e**c**t*</sub>, leading to i.i.d. samples from
    a binomial distribution.
    $$\kappa\_{xy} = \frac{c\_{obs,xy} - c\_{exp,xy}}{1 - c\_{exp,xy}},$$
    

    with *c*<sub>*e**x**p*, *x**y*</sub> = *p*<sub>*i*</sub>*p*<sub>*j*</sub> + (1 − *p*<sub>*i*</sub>)(1 − *p*<sub>*j*</sub>) , *c*<sub>*o**b**s*, *x**y*</sub> = #of agreements/*M*

    where *c*<sub>*e**x**p*, *x**y*</sub> represents the expected
    probability of agreement between model *x* and *y*, calculated from
    the accuracies *p*<sub>*x*</sub> and *p*<sub>*y*</sub> of two
    independent binomial observers, and *c*<sub>*o**b**s*, *x**y*</sub>
    denotes the observed probability of agreement. Cohen’s *κ* assesses
    the consistency of error overlap, providing a measure of
    classification agreement without distinguishing error types.

3.  **Jaccard Similarity Coefficient** is defined as:
    $$J(x,y) = \frac{\sum_{i=1}^n x_i y_i}{\sum_{i=1}^n (x_i + y_i - x_i y_i)}$$
    where each *x*<sub>*i*</sub>, *y*<sub>*i*</sub> ∈ {0, 1} represents
    the correctness (1) or incorrectness (0) of the *i*th sample
    prediction from the two models, respectively. The numerator
    "\|Intersections\|" counts samples where both models predict
    correctly, normalized by "\|Unions\|", which counts samples where
    either model predicts correctly.

4.  **Hamming Distance** counts the number of discrepancies in the
    correctness of predictions:
    *d*(*x*, *y*) = \|{*i* : *x*<sub>*i*</sub> ≠ *y*<sub>*i*</sub>, *i* = 1, …, *n*}\|.
    

5.  **Agreement Score is the normalized difference between counts of
    agreement and disagreement in the prediction correctness made by the
    two models:
    $$s(x, y) = \frac{(n\_{11} + n\_{00}) - (n\_{10} + n\_{01})}{n\_{11} + n\_{00} + n\_{10} + n\_{01}}$$
    with *n*<sub>*i**j*</sub>, where *i*, *j* ∈ 0, 1, counts predictions
    where model *x* predicts *i* (correct/incorrect) and model *y*
    predicts *j* over shared stimuli.**

## Downstream Behavioral Datasets

We analyze the behavior of all models across a series of downstream
tasks, including in-distribution and several out-of-distribution image
types, such as silhouettes, stylized images, and natural images
distorted by various noise types (see
Appendix <a href="#subsec:datasets" data-reference-type="ref"
data-reference="subsec:datasets">4.1</a> for details). In total, these
comparisons span 17 behavioral datasets.

## Selection of Neural Network Architectures and Layers

We incorporated a comprehensive list of popular deep learning models
pretrained over the 1000-class classification tasks over the ImageNet-1k
dataset . The selection spans a diverse set of architectures, including
conventional convolutional neural networks (CNNs) and transformers.
These models were trained using various objective functions, both
supervised and self-supervised. Specifically, our lineup includes
AlexNet , ResNet  , VGG16  , Inception  , ResNeXt  , MoCo , ResNet
Robust , and several variants of Vision Transformers (ViTs)   such as
Vit-b16 and ViT-ResNet (vit on ResNet architecture), and Swin
transformer . For representational analysis, we mainly focused on the
penultimate layer of each model, where we averaged the outputs across
channels or patches, as applicable per architecture. For transformer
models, we’ve included outputs from the final GELU activation layers in
addition to their penultimate layer.

We included randomized versions of AlexNet, ResNet, ViT, and Swin to
study their behavior under random initialization before training.

# Results

<figure id="figs:fig2a">
<img src="figs/model_sim_viz.png" />
<figcaption>Model-by-model similarity matrices from different measures
on the Cue Conflict task. <strong>Left</strong>: The Procrustes measure
clearly distinguishes between trained and untrained models.
<strong>Middle</strong>: Linear Predictivity reveals no noticeable
separation between trained and untrained models or across different
architectures. <strong>Right</strong>: Soft-matching more effectively
differentiates between architectural families (CNN vs. transformers)
compared to other representational metrics.</figcaption>
</figure>

## Different Representational Similarity Measures have Distinct Capacities for Model Separation

To characterize how different representational similarity measures
discriminate models, we first visualize the model-by-model similarity
matrices for each measure. We observed that while some measures like the
soft-matching distance were effective at differentiating architectural
families (Fig. <a href="#figs:fig2a" data-reference-type="ref"
data-reference="figs:fig2a">2</a>, right), others like the Procrustes
distance were more sensitive to the effects of training (Fig.
<a href="#figs:fig2a" data-reference-type="ref"
data-reference="figs:fig2a">2</a>, left), clearly separating trained
from untrained models. Other measures, like linear predictivity, which
allow greater flexibility in aligning the two representations, showed
limited ability in distinguishing between models trained with different
architectures or trained from untrained models (see
Appendix <a href="#subsec:rsms" data-reference-type="ref"
data-reference="subsec:rsms">4.4</a> for additional similarity
matrices). To quantify these distinctions, we computed *d*′ scores
(Appendix <a href="#subsec:dprime" data-reference-type="ref"
data-reference="subsec:dprime">4.2</a>) to assess each measure’s ability
to differentiate two categories of models: (a) those from different
architectural families, and (b) those with varying levels of training
(trained vs. untrained). Significant differences in *d*′ scores emerged
across measures (Fig. <a href="#figs:fig2b" data-reference-type="ref"
data-reference="figs:fig2b">3</a>). For instance, Procrustes achieved
*d*′ scores with a mean of 3.70 when separating trained from untrained
models across all datasets, while commonly used measures like CCA and
linear predictivity produced much lower scores with means of 0.53 and
0.87, respectively. Similarly, some measures were better at
discriminating architectural differences, with the soft-matching
distance demonstrating the highest discriminability (mean of *d*′ scores
= 1.6). Previous studies have also demonstrated that different measures
vary in their effectiveness at establishing layer-wise correspondence
across networks with the same architecture . Considering these
differences in how measures distinguish between models, a key question
emerges: Which distinctions should we prioritize?

<figure id="figs:fig2b">
<img src="figs/dprimes" style="width:80.0%" />
<figcaption>Discriminative ability (d’ scores) of (top) representational
and (bottom) behavioral similarity measures in distinguishing between
trained vs. untrained models (left) and architectures
(right).</figcaption>
</figure>

## Behavioral Metrics Primarily Reflect Learning Differences Over Architectural Variations

To address the question of which separation should be prioritized, we
return to our central premise: measures that emphasize functional
distinctions should be favored. Therefore, we next evaluated how
different behavioral measures (as previously described) distinguish
between models. Our results show that behavioral metrics effectively and
consistently separate trained from untrained networks, with even the
weakest metric (Confusion Matrix (JSD)) achieving a mean *d*′ of 1.82.
However, most behavioral measures struggle to differentiate between
architectural families (e.g., CNNs vs. Transformers), with the
best-performing metric (Confusion Matrix (Inter-class JSD)) achieving an
average *d*′ of 0.65 across all behavioral datasets (see
Appendix <a href="#subsec:behavmatrices" data-reference-type="ref"
data-reference="subsec:behavmatrices">4.5</a> for all similarity
matrices). This suggests that differences in these architectural motifs
have minimal impact on model behavior.

## Behavioral Metrics Show Greater Consistency Than Neural Representational Similarity Measures

We next examined the consistency across different representational
similarity measures and across different behavioral measures by
computing correlations between the model-by-model similarity matrices
generated by each measure. As shown in Fig.
<a href="#figs:fig3" data-reference-type="ref"
data-reference="figs:fig3">4</a> (Top),we find that behavioral metrics
(mean r: 0.85 ± 0.01) are more correlated on average than
representational metrics (mean r: 0.75 ± 0.007), with a significant
difference (*z* = −7.10, *p* = 5 × 10<sup>−8</sup> \< 0.0001).

To further understand the relationships between different
representational similarity measures, we analyzed the MDS plot (Fig.
<a href="#figs:fig3" data-reference-type="ref"
data-reference="figs:fig3">4</a> (Bottom)). This visualization revealed
distinct clusters of measures based on their theoretical properties.
Measures that rely on inner product kernels (stimulus-by-stimulus
dissimilarities) tend to group together, indicating they capture similar
aspects of representational structure. On the other hand, measures that
use explicit, direct mappings between individual neurons—such as Linear
Predictivity and Semi-Matching—form a separate cluster. Notably,
Procrustes Distance and CCA also involve alignment, similar to Linear
Predictivity and Semi-Matching; however, this alignment is achieved
collectively across all units or neurons rather than through
independently determined mappings for each neuron. Procrustes aligns the
entire configuration of points, while CCA projects the two
representations onto common subspaces to maximize correlation, further
distinguishing them from other representational similarity approaches.

How behavioral metrics distinguish models is crucial, as most
comparative analyses of representations in neuroscience and AI revolve
around understanding computations and how those computations relate to
behavior; behaviorally grounded comparisons of model representations are
key to this endeavor. We find that behavioral metrics distinguish
between models in a consistent manner across different datasets,
reinforcing the robustness of the model relationships they uncover
(Appendix  <a href="#subsec:consistency" data-reference-type="ref"
data-reference="subsec:consistency">4.3</a>). The consistency of the
behavioral metrics -across datasets and with each other- fulfills
another scientific desiderata of replicability. Therefore, the model
relationships identified by behavioral metrics are not only important
but also reliable. It becomes crucial, then, to determine which
representational similarity measures align with these robust behavioral
relationships between models.

<figure id="figs:fig3">
<img src="figs/mds_metric_corr.png" />
<figcaption><strong>Consistency Between Similarity Metrics.</strong> (A)
and (C) display the correlation matrix averaged across all behavioral
datasets and the 2D-projected multidimensional scaling (MDS) plot (using
1 minus the correlation matrix as the distance matrix) for behavioral
measures. (B) and (D) illustrate the average correlation matrix and the
MDS plot for representational similarity measures.</figcaption>
</figure>

## Which representational similarity measures show the strongest correspondence with behavioral measures?

Seeing that we want to prioritize the model relationships uncovered by
behavioral metrics, we move on to investigate which –if any–
representational similarity metrics reveal the same underlying
relationships between models. To rigorously assess this, we computed
correlations between the model-by-model similarity matrices of each
representational metric with the model-by-model behavioral similarity
matrix averaged across all behavioral metrics, separately for many
datasets (Fig <a href="#figs:fig4" data-reference-type="ref"
data-reference="figs:fig4">5</a>). We found that three metrics stood out
in their alignment with behavioral metrics - RSA (mean r: 0.52), Linear
CKA (mean r: 0.64), and Procrustes (mean r: 0.70). Going back to our
original analysis, these metrics are also able to more strongly
differentiate trained and untrained models (Fig
<a href="#figs:fig1" data-reference-type="ref"
data-reference="figs:fig1">1</a> Top d’ measures). All these
representational metrics emphasize alignment in either the overall
geometry or shape of representations. Alternate measures like linear
predictivity and CCA, which are commonly employed in representational
comparisons in neuroscience and AI, showed significantly weaker
alignment with mean correlation scores of 0.26 and 0.19 respectively.
Given the opacity of neural representations, selecting appropriate
representational similarity metrics can be challenging; these findings
offer crucial guidance for metrics that support behaviorally grounded
comparisons.

<figure id="figs:fig4">
<img src="figs/metric_gt_corr_figure.png" />
<figcaption><strong>Granular Comparison of Representational Similarity
Measures with Behavioral Measures</strong>: (A) Average correlation
between representational and behavioral metrics across datasets. (B)
Distribution of correlation scores for each representational similarity
measure with behavioral measures; each point represents the averaged
score for a dataset across all behavioral measures.</figcaption>
</figure>

# Discussion

In this study, we compared 8 neural representational similarity metrics
and 9 behavioral measures across 17 datasets.Based on the premise that
behavioral differences should be mirrored in the representational
structure of neural networks, we examined practical distinctions in
their alignment with behavior. Metrics like RSA, CKA, and Procrustes
distance, which preserve the overall geometry of neural representations,
tend to align closely with behavioral measures. In contrast, methods
like linear predictivity, which align dimensions without preserving
global geometry, show weaker alignment. This divergence likely arises
because linear predictivity has the capacity of mapping complex,
distributed geometric structures to simpler, compressed ones while
maintaining prediction accuracy. For instance, trained networks were
observed to predict untrained network activation patterns well, yielding
high symmetrized scores.

Moreover, while different behavioral measures generally show
consistency, neural representational similarity metrics do not,
underscoring the need for a deeper understanding of how these
representational metrics discriminate between models in practical
applications. Our analysis sets a new standard for representational
similarity measures in neuroscience and AI, using downstream behavioral
robustness as a guide for selecting the most suitable metric. This
framework is especially crucial in model-brain comparisons, where
representational analyses are frequently applied to assess if artificial
neural networks and biological systems are serving comparable functional
roles in terms of perceptual and cognitive processes.

Our framework for representational metric selection, though robust,
makes some key assumptions. It assumes a specific mechanism for how
behavior is ‘reading out’ from neural representations, and different
readout mechanisms could reveal qualitatively different relationships
between models. For example, applying biologically-inspired constraints,
such as sparsity, could reveal divergent relationships, especially if
some models encode behaviorally relevant information in a sparse manner
that others do not. In such cases, the precise representation structure
at the unit-level becomes critical. Additionally, we defined "behavior"
within the scope of object classification across multiple
out-of-distribution (OOD) image datasets. Extending evaluations to
include fine-grained visual discrimination or broader tasks beyond
categorization would better capture the full range of visual processing.
Lastly, a stronger theoretical framework explaining why certain
similarity measures align more closely with behavior than others is
currently lacking in our work, but this remains an exciting direction
for future research.

# Appendix

## Downstream Behavioral Datasets

All datasets, directly drawn from , share the coarser 16 labels from
ImageNet. These consist of a subset of the ImageNet1k validation set
sampled from the following categories: Airplane, Bear, Bicycle, Bird,
Boat, Bottle, Car, Cat, Chair, Clock, Dog, Elephant, Keyboard, Knife,
Oven, Truck.

-   **Colour**: Served as a baseline in-distribution dataset, with half
    of the images randomly converted to greyscale and the rest kept in
    original color. Includes a total of 1280 images (80 images per
    label).

-   **Stylized ImageNet (SIN)**: Textures from one class are applied to
    shapes from another while maintaining object shapes. Shape labels
    are used as "true labels" for confusion matrix and correctness
    analyses. Includes a total of 800 images

-   **Sketch**: Contains cartoon-styled sketches of objects from each
    class, totaling 800 images.

-   **Edges**: Created from the original dataset using the Canny edge
    extractor for edge-based representations. Includes a total of 160
    images

-   **Silhouette**: Black objects on a white background, generated from
    the original dataset. Includes a total of 160 images

-   **Cue Conflict**: Images with texture conflicting with shape
    category, generated using iterative style transfer   between Texture
    dataset images (style) and Original dataset images (content).
    Includes a total of 1280 images.

-   **Contrast**: Variants of images adjusted for contrast levels.
    Includes a total of 1280 images.

-   **High-Pass/Low-Pass**: Images filtered to emphasize either
    high-frequency or low-frequency components using Gaussian filters.
    Includes a total of 1280 images per dataset.

-   **Phase-Scrambling**: Images had phase noise added to frequencies,
    creating different levels of distortion from 0 to 180 degrees.
    Includes a total of 1120 images.

-   **Power-Equalisation**: Images were processed to equalize the power
    spectra across the dataset by setting all amplitude spectra to their
    mean value. Includes a total of 1120 images.

-   **False-Colour**: Images had colors inverted to their opponent
    colors while keeping luminance constant using the DKL color space.
    Includes a total of 1120 images.

-   **Rotation**: Images are rotated by 0, 90, 180, or 270 degrees to
    test rotational invariant robustness. Includes a total of 1120
    images.

-   **Eidolon I, II, III**: Images distorted using the Eidolon toolbox,
    varying coherence and reach parameters to manipulate local and
    global image structures. Each filtering intensity level contains
    1280 images.

-   **Uniform Noise**: White uniform noise added to images with a
    varying range to assess robustness; pixel values exceeding bounds
    were clipped. Includes a total of 1280 images.

## Inter vs Intra Group Statistic Measures using *d*′ Scores

To quantify a comparative metric’s ability to reflect the expected
proximity between similarly trained models, compared to their
dissimilarity with the untrained models, involves speculating the group
statistics from the resulting similarity matrix. We employ the *d*′
score defined as:
$$d' = \frac{\mu(A) - \mu(B)}{\sqrt{\frac{\sigma^2_A + \sigma^2_B}{2}}}$$
where **A** represents the set of similarity scores from **intra-group**
comparisons, specifically the similarity scores between every pair of
trained models. **B** represents the set of similarity scores from
**inter-group** comparisons, specifically the similarity scores between
each pair of trained and untrained models. Equivalent to the set of
entries located at the intersection of trained model rows and untrained
model columns in the model-by-model similarity matrix of the metrics.

A similarity metric with *d*′ ≥ 0 of greater magnitude indicates a
greater ability to separate trained models from untrained ones. A metric
with *d*′ = 0 or *d*′ \< 0 indicates that there were no discernible
difference in average similarity scores computed in "trained model
pairs" and "trained vs. untrained model pairs", or that trained vs.
untrained models exhibit even higher similarity than that among trained
models.

Similarly, when examining architectural differences, **A** represents
intra-group comparisons within Convolutional models, while **B**
captures inter-group comparisons between Convolutional models and
Transformers.

## Dataset Consistency

To assess consistency across behavioral datasets, we used an *M* × *M*
correlation matrix, where *M* is the number of datasets. Each entry
*i*, *j* represents the correlation between datasets *i* and *j*,
derived from their downstream similarity matrices. Averaging these
scores across all behavioral measures revealed high correlations,
indicating consistent uniformity across most datasets.

<figure id="figs:dataset-consis">
<p><img src="figs/dataset-consistency.png" style="width:80.0%"
alt="image" /> <span id="figs:dataset-consis"
data-label="figs:dataset-consis"></span></p>
</figure>

## Representation Similarity Matrices

We include the Model-by-Model Similarity Matrix given by the 8 distinct
representation measures. The scores provided are averaged across 17
datasets. For mutual k-NN, different neighborhood sizes (*k*) are
included. Note that the "1 − Procrustes" score can range from (−∞, 1\],
whereas all other metrics yield scores within the range \[0, 1\].

<figure id="figs:rep-all">
<p><embed src="figs/all-rep-sims.pdf" /> <span id="figs:rep-all"
data-label="figs:rep-all"></span></p>
</figure>

<figure id="figs:rep-all">
<p><embed src="figs/all-rep-sims-NN.pdf" /> <span id="figs:rep-all"
data-label="figs:rep-all"></span></p>
</figure>

## Behavioral Similarity Matrices

Similarly, we include the Model-by-Model Similarity Matrix given by the
9 distinct behavioral measures. The scores are averaged across 17
datasets. For the measures "1 − Hamming Distance" and "Agreement
Scores", the alignment value can all range from (−∞, 1\], whereas all
other measures yield scores within the range \[0, 1\].

<figure id="figs:rep-all">
<p><embed src="figs/all-behavioral-sims-bin.pdf" /> <span
id="figs:rep-all" data-label="figs:rep-all"></span></p>
</figure>

<figure id="figs:rep-all">
<p><embed src="figs/all-behavioral-sims.pdf" /> <span id="figs:rep-all"
data-label="figs:rep-all"></span></p>
</figure>

[^1]: The term "encoder/kernel function: refers to the function that
    represents the mapping from an input to the output of a specific
    layer’s activation in a neural network

## Equations

This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) that brought a significant improvement to the loading and rendering speed, which is now [on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php).

---

## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

---

## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>

---

## Code Blocks

Syntax highlighting is provided within `<d-code>` tags.
An example of inline code snippets: `<d-code language="html">let x = 10;</d-code>`.
For larger blocks of code, add a `block` attribute:

<d-code block language="javascript">
  var x = 25;
  function(x) {
    return x * x;
  }
</d-code>

**❗️ Note:** `<d-code>` blocks do not look good in the dark mode. ❗️

You can always use the default code-highlight using the `highlight` liquid tag:

{% highlight javascript %}
var x = 25;
function(x) {
return x \* x;
}
{% endhighlight %}

---

## Interactive Plots

You can add interative plots using plotly + iframes :framed_picture:

<div class="l-page">
  <iframe src="{{ '/assets/plotly/demo.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

The plot must be generated separately and saved into an HTML file.
To generate the plot that you see above, you can use the following code snippet:

{% highlight python %}
import pandas as pd
import plotly.express as px
df = pd.read_csv(
'https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv'
)
fig = px.density_mapbox(
df,
lat='Latitude',
lon='Longitude',
z='Magnitude',
radius=10,
center=dict(lat=0, lon=180),
zoom=0,
mapbox_style="stamen-terrain",
)
fig.show()
fig.write_html('assets/plotly/demo.html')
{% endhighlight %}

---

## Details boxes

Details boxes are collapsible boxes which hide additional information from the user. They can be added with the `details` liquid tag:

{% details Click here to know more %}
Additional details, where math $$ 2x - 1 $$ and `code` is rendered correctly.
{% enddetails %}

---

## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body` sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

---

## Other Typography?

Emphasis, aka italics, with _asterisks_ (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or **underscores**.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
   ⋅⋅\* Unordered sub-list.
3. Actual numbers don't matter, just that it's a number
   ⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

- Unordered list can use asterisks

* Or minuses

- Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links.
http://www.example.com or <http://www.example.com> and sometimes
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style:
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style:
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```

```python
s = "Python syntax highlighting"
print s
```

```
No language indicated, so no syntax highlighting.
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        |      Are      |  Cool |
| ------------- | :-----------: | ----: |
| col 3 is      | right-aligned | $1600 |
| col 2 is      |   centered    |   $12 |
| zebra stripes |   are neat    |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the
raw Markdown line up prettily. You can also use inline Markdown.

| Markdown | Less      | Pretty     |
| -------- | --------- | ---------- |
| _Still_  | `renders` | **nicely** |
| 1        | 2         | 3          |

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can _put_ **Markdown** into a blockquote.

Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a _separate paragraph_.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the _same paragraph_.
