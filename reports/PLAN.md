## Current tasks
* [ ] Test that whole group can access data in cloud (M21)

- S
* [ ] Construct evaluate docker file (M10)
* [ ] Build the docker file locally and make sure it work as intended (M10)

- N
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (Wandb) (M14)

- B
* [ ] Construct train docker file (M10)
* [ ] Construct api docker file
* [ ] Build the docker files locally and make sure they work as intended (M10)

## General
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7) (Ruff)
* [ ] Remember to keep your `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [ ] Add command line interfaces and project commands to your code where it makes sense (Typer) (M9)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7) (Adding types to functions)
* [ ] Use profiling to optimize your code (M12)

## Overall project remaining checklist points

### Week 1
- Core

### Week 2
- Core
* [ ] Unit test of train
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)

- Optional
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3
- Core
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)

- Optional
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra
- Optional
* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub
