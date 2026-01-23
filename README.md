# artsy

The overall goal of the project is to gain experience with the frameworks introduced in this course as well as other open-ended frameworks. We will demonstrate what we have learnt by conducting an experiment where we classify images into different art styles such as Impressionism, Realism, etc.
Beyond using PyTorch as our basic framework, we will use PyTorch Lightning to efficiently train and evaluate our model. Furthermore, we will employ Docker for resource management, Weights And Biases (Wandb) for experiment management and hyperparameter tuning, and VS Code for software engineering.

We are using the “WikiArt” dataset from Huggingface which has been gathered from WikiArt.org, an open-source encyclopedia of visual art. It contains images of paintings, the name of the artist, a genre label as well as a style label. There are 27 style classes dominated by Impressionism, Realism and Post-impressionism. A style is defined by its "visual elements, techniques and methods" and "usually corresponds with an art movement".

In total there are 81444 color images of varying shapes, painted by different 129 artists. The dataset is 33.7 GB large, so our idea is to reshape the image to 128x128. That way the dataset decreases in size and the images become more comparable.  Beyond this, the images should be scaled before being saved as .pt files.
This can either be a random selection of data, or by choosing a set number of images from each style. Since the dataset also seems unbalanced with regards to the styles, we will try to create a more balanced subset. We expect to train a classic CNN to classify the images into the styes. We will try to develop a tool which can be used to label images into one of these styles.

Others have also worked on the dataset – for example creating images based on artistic styles, and classification tasks – indicating that it is realistic to train a model on it, and depending on the task, achieving high accuracy. As a baseline we will use a single-layer MLP.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
