# Causal ML for Real-World Decision Systems

This project implements a causal machine learning framework based on **Judea Pearl’s Structural Causal Models (SCM)**. The goal is to transition from the first layer of the causal hierarchy (**Association/Prediction**) to the second layer (**Intervention**) to understand how a system responds to external changes.

---

## 📋 Overview

Standard machine learning models typically operate at the level of "What do I believe about the data?" (Layer 1). 
This framework enables the system to answer "What if the system had an intervention?" (Layer 2) by modeling the underlying data-generating process.

In this project, we apply this framework to a **climate dataset for rainfall prediction**. 
We believe that by utilizing causality, we can model a more sound physical process. 
This approach aims to remain robust and effective even when a **distribution shift** in the variables occurs, where standard correlational models often fail.

### Core Methodology
* **Structural Model:** We represent the system using a **Directed Acyclic Graph (DAG)**.
* **Hidden Confounders:** The model explicitly accounts for unobserved variables that influence both the causes and the effects.
* **CausalVAE:** We utilize a Causal Variational Autoencoder to estimate these latent (hidden) confounder variables.
* **Interventional Distribution:** Once trained, the model allows us to approximate the response interventional distribution, enabling robust decision-making in real-world scenarios.

---

## 🛠️ Project Structure

The workflow is divided into data acquisition, model training, and causal inference:

| Step | Script | Description |
| :--- | :--- | :--- |
| **1. Preprocessing** | `download_precipitation.py` | Downloads and prepares the raw climate data for processing. |
| **2. Training** | `trainer_model.py` | Trains the CausalVAE to identify latent factors and structural relationships. |
| **3. Evaluation** | `prediction.py` | Validates the model's predictive performance on held-out data. |
| **4. Analysis** | `counterfactual.py` | Performs counterfactual and interventional analysis to simulate "What-if" scenarios. |

---

## 🚀 How to Use

To replicate the results or apply the model to your data, run the scripts in the following order:

1.  **Prepare the data:**
    ```bash
    python download_precipitation.py
    ```

2.  **Train the CausalVAE model:**
    ```bash
    python trainer_mode.py 

3.  **Prediction with the CausalVAE model:**
    ```bash
    python prediction.py

4.  **Counterfactual estimation with the CausalVAE model:**
    ```bash
    python counterfactual.py 
