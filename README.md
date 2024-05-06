# OAPapers
This is reference repository for paper _OARelatedWork: A Large-Scale Dataset of Related Work Sections with Full-texts from Open Access Sources_.  
It contains the code used to create the OARelatedWork dataset and the OAPapers corpus. Read the paper for more information about the dataset and the corpus.

If you just want a loader for the dataset, you can use the [https://github.com/KNOT-FIT-BUT/OAPapersLoader](https://github.com/KNOT-FIT-BUT/OAPapersLoader).

We also provide handy data viewer at [https://github.com/KNOT-FIT-BUT/OAPapersViewer](https://github.com/KNOT-FIT-BUT/OAPapersViewer).

# Install
You must install the faiss (faiss~=1.7.1):

    conda install -c pytorch faiss-gpu

or

    conda install -c conda-forge faiss-gpu

Install ScispaCy model (https://github.com/allenai/scispacy), which is used for sentence segmentation:

    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz

Voluntary install spacy for gpu. If you know you cuda (spacy[cuda102], spacy[cuda112], spacy[cuda113], ...) specify it
or try just cuda. Example of installation command:

    pip install -U spacy[cuda113]

Use the standard requirements.txt for installation of other packages:

    pip install -r requirements.txt

Also, as there are Cython extensions, you need to build them:

    python setup.py build_ext --inplace

or use:

    ./build_cython.sh

## OAPapers Corpus
The corpus is available at [https://athena2.fit.vutbr.cz/oapapers/](https://athena2.fit.vutbr.cz/oapapers/).

You can load it using:

```python
from oapapers.datasets import OADataset

with OADataset("oapapers.jsonl", "oapapers.jsonl.index") as dataset:
    print("Document:", dataset[0].title)


```
As this is rather heavyweight package with lots of dependencies, you can also use the repository containing only dataset loaders: [https://github.com/KNOT-FIT-BUT/OAPapersLoader](https://github.com/KNOT-FIT-BUT/OAPapersLoader).

## OARelatedWork dataset
There is existing Hugging Face dataset for OARelatedWork dataset. You can find it at [https://huggingface.co/datasets/BUT-FIT/OARelatedWork](https://huggingface.co/datasets/BUT-FIT/OARelatedWork).

You can also load it directly using this package:

```python
from oapapers.datasets import OARelatedWork, OADataset

with OARelatedWork("train.jsonl", "train.jsonl.index") as dataset, \
            OADataset("references.jsonl", "references.jsonl.index") as references:
    d = dataset[0]
    print("Document:", d.title)
    print("Cited paper:", references.get_by_id(d.citations[0]).title)


```
The OARelatedWork will load the target papers with related work sections and the OADataset will load dataset of all references
that can be used for loading cited papers.

As this is rather heavyweight package with lots of dependencies, you can also use the repository containing only dataset loaders: [https://github.com/KNOT-FIT-BUT/OAPapersLoader](https://github.com/KNOT-FIT-BUT/OAPapersLoader).


### Fields

* **id** - id from our corpus
* **s2orc_id** - SemanticScholar id
* **mag_id** - Microsoft Academic Graph id
* **DOI** - Might be DOI for another version of document than the one used for processing.
* **title** - title of publication
* **authors** - authors of publication
* **year** - year of publication
* **fields_of_study** - list of fields of study
* **citations** - list of paper ids cited in the related work section
* **hierarchy** - Document body without related work section, abstract is the first section in the hierarchy.
* **related_work** - The target related work section.
* **related_work_orig_path** - Path to the original related work section in the document.
* **bibliography** - document bibliography
* **non_plaintext_content** - tables and figures

# Sources
The corpus (and derived dataset) contains open access papers obtained from **CORE** and **SemanticScholar** corpora. These corpora contain third party content and materials, such as open access works from publicly available sources. In addition to the licenses of those organizations (ODC-By, CC BY-NC), any underlying Third Party Content may be subject to separate license terms by the respective third party owner. We made the best effort to provide identifiers (title, authors, year, DOI, or SemanticScholar ID) of collected papers to allow the user of this corpus (and derived dataset) to check the license.


# Cite

If you use the dataset or the corpus, please cite the following paper:

```bibtex
@article{docekal2024oarelatedwork,
      title={OARelatedWork: A Large-Scale Dataset of Related Work Sections with Full-texts from Open Access Sources}, 
      author={Martin Docekal and Martin Fajcik and Pavel Smrz},
      year={2024},
      eprint={2405.01930},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```