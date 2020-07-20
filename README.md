# OsGenePrio

- We first built the **OsGenePrio** dataset from several open-access databases and used it as a benchmark for the evaluation. 
- Next, we used the Translating Embedding (TransE) embedding model to vectorize gene information. Then we combined the use of Convolutional Neural Network (CNN) with Knowledge Base (KB) to accurate information of genes vectors.
- More specifically, we compared the use of Convolutional Knowledge Base model (ConvKB) and Capsule Network-based Embedding Model (CapsE) to optimized genes vectors.
- Finally, we evaluated results following two steps. By using link prediction performance to evaluate the vectorized model, then using unsupervised learning techniques to check the accuracy of candidate genes. In conclusion, K-means gave similar results compare to biologists predictions while K-Nearest Neighbor recommended genes that have similar attributes.

##Installation
* Clone this repo
```
cd
git clone https://github.com/hongquando/RiceNLP.git
```

#Setup for testing model
* To use result of ConvKB model, download [ConvKB](https://drive.google.com/file/d/1PrvejuEUC1iFPyBVH7TAcxlEUCljmlo8/view?usp=sharing)
  Place convert weights into ```/weights/```. 

* To use result of CapsE model, download [CapsE](https://drive.google.com/file/d/1xUJi2oxSf8som4nBRknJjAUfxVUss_jh/view?usp=sharing)
  Place convert weights into ```/weights/```. 

* Downloads data of genes, download [database](https://drive.google.com/file/d/1-iJDvrUh83ewYzFlDMfwC21hw3Y2QM80/view?usp=sharing)
  Place convert weights into ```/support/```. 
  We have the result:
  - ```/support/iric_dick.pkl```
  - ```/support/pyrice.pkl```
  - ```/support/uniprot.pkl```
   
##Training with OsGenePrio
```
# For ConvKB model

python TrainConvKB.py

# For CapsE model

python TrainCapsE.py
```

##Testing with OsGenePrio
* Results on the validation set with 2 models:
 
    |   Evaluation  | ConvKB          | CapsE  |
    | ------------- |:-------------:| -----:|
    | MR      | 62854 | **56665** |
    | MRR      | 0.686     |   **0.687** |
    | Hits@10  |  0.904     |    **0.919**


* With ConvKB model:
    - Using [KMeans_convkb.ipynb]() to see result of K-means algorithm
    - Using [KNN_convkb.ipynb]() to see result of K-Nearest Neighbor algorithm
* With CapsE model:
    - Using [KMeans_capse.ipynb]() to see result of K-means algorithm
    - Using [KNN_capse.ipynb]() to see result of K-Nearest Neighbor algorithm
    
#Citation
```
@inproceedings{Nguyen2018,
  author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dat Quoc Nguyen and Dinh Phung},
  title={{A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network}},
  booktitle={Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  pages={327--333},
  year={2018}
}

@inproceedings{nguyen-etal-2019-capsule,
    title = "A Capsule Network-based Embedding Model for Knowledge Graph Completion and Search Personalization",
    author = "Nguyen, Dai Quoc  and
      Vu, Thanh  and
      Nguyen, Tu Dinh  and
      Nguyen, Dat Quoc  and
      Phung, Dinh",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1226",
    doi = "10.18653/v1/N19-1226",
    pages = "2180--2189",
}
```
