This project implements **Federated Learning (FL)** using a **Multi-Layer Perceptron (MLP)** for HuGaDB.

---

## **How the Federated Learning Pipeline Works**

### **Each Federated Learning Round Includes 3 Steps:**

#### **1. Local Training**

* The server sends the current global model to every client.
* Each client trains the model on its own local dataset for a few epochs.

#### **2. Model Aggregation (FedAvg)**

* The server averages all client models using **weighted averaging**, where weights depend on the number of samples each client holds.
* This produces a new **global model**, combining weights from all clients.

#### **3. Global Evaluation**

* The updated global model is evaluated on:

  * A **validation set** (for tracking improvement & early stopping)
  * A **test set** (for monitoring accuracy each round)

---

