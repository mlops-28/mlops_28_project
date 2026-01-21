## Current tasks
* [ ] Test that whole group can access data in cloud (M21)

- S
* [ ] Create evaluate.py (using Pytorch lightning) and visualize.py
* [ ] Use Hydra to load the configurations and manage your hyperparameters (missing model) (M11)
* [ ] Use logging to log important events in your code (Hydra + logging or WandB) (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (Wandb) (M14)

-N
* [ ] Dependabot
      
- B
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)

## General
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7) (Ruff)
* [ ] Remember to keep your `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [ ]  Add command line interfaces and project commands to your code where it makes sense (Typer) (M9)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7) (Adding types to functions)
      
## Overall project remaining checklist points

### Week 1
* [ ] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [ ] Use profiling to optimize your code (M12)

### Week 2
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3
* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra
* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub
