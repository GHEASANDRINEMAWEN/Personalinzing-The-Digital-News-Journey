# Personalizing the Digital News Journey  
### Click-Through Prediction Using Fused Content Embeddings and Behaviour-Aware Modelling

This project builds a hybrid news recommendation pipeline combining **fused article embeddings**, **behaviour-derived user vectors**, and **machine-learned ranking models** (LightGBM and DeepFM).  
The goal is to predict which article a user is most likely to click within each impression.

Dataset: Ekstra Bladet RecSys Challenge (News CTR Prediction)

---

##  Dataset Overview

Four datasets were used and can be gotten from https://www.recsyschallenge.com/2024/#about:

| Dataset | Description |
|--------|-------------|
| **Articles** | Contains title, subtitle, and full article text. |
| **Contrastive Vectors** | Precomputed multimodal embeddings capturing both image and text signals. |
| **User History** | Past engagements with read_time, scroll_depth, and article IDs. |
| **Behaviors** | Impression-level data containing candidate articles and the clicked label. |

---

##  Methodology Overview

### 1. Item Embeddings
Each article was represented using a **1536-dimensional fused vector**:

- **Text embedding:**  
  Title + subtitle + body text encoded using a transformer model.

- **Contrastive image–text embedding:**  
  Provided by the challenge dataset.

- **Fusion:**

- 
This produced a rich content representation capturing both semantic and visual cues.

---

### 2. User Embeddings
User preferences were derived from their reading history.

For each article \( j \), we computed an engagement weight:

\[
w_j = 0.7 \cdot \mathrm{read\_time\_norm}_j + 0.3 \cdot \mathrm{scroll\_norm}_j
\]

The final user vector is a weighted average of the article embeddings:

\[
U = \frac{\sum_j w_j A_j}{\sum_j w_j}
\]

This captures what the user reads and **how strongly they engaged**.

---

### 3. Behavioural Feature Engineering
The behaviors dataset was **exploded** so each candidate article becomes one training row.  
For each row we added:

- Positional features (top-1, top-3, last, normalized position)  
- Time-of-day / weekend indicators  
- Device type, SSO status, subscription status  
- Engagement features (scroll, read_time flags)  
- User–item similarity (cosine similarity)  

Each row becomes:


---

### 4. Final Training Table
After merging **item embeddings**, **user embeddings**, and **behavioural features**,  
the final dataset is a large tabular matrix where:

- each row = *one article shown to one user*  
- columns = *content + behavioural + contextual features*  
- label = *clicked or not clicked*

---

##  Models Used

### **LightGBM**
Selected because it:

- handles mixed tabular data efficiently  
- captures non-linear relationships  
- trains fast and scales well  
- performs strongly on CTR problems  

**Setup:**

- 1,000 trees  
- early stopping on validation AUC  
- same feature space as DeepFM  

---

### **DeepFM**
Used to test whether a deep model can capture richer feature interactions.

- FM component learns low-order interactions  
- Deep neural network learns high-order patterns  
- Designed specifically for CTR prediction  

**Setup:**

- up to 50 epochs  
- early stopping on val-AUC  
- Adam optimizer, batch size 512  

---

##  Evaluation Metrics

We evaluated using industry-standard ranking and classification metrics:

- **AUC** — Measures ranking ability between clicked vs. non-clicked.  
- **Log Loss** — Penalizes incorrect predicted probabilities.  
- **Average Precision (AP)** — Precision–recall performance across thresholds.  
- **NDCG@10** — Quality of ranking within impressions.  
- **MRR@10** — Rank of the first clicked article.  
- **Recall@10** — How many true clicks appear in the top-10 predictions.  

---

##  Key Results

LightGBM achieved stronger performance:

| Metric | LightGBM | DeepFM |
|--------|----------|--------|
| **AUC** | 0.7796 | 0.7312 |
| **NDCG@10** | 0.5846 | 0.5034 |
| **MRR@10** | 0.4740 | 0.3840 |
| **AP** | 0.2447 | 0.2030 |

**Conclusion:**  
LightGBM provided the best overall ranking quality and was chosen as the final model.

---

##  Real User Test 

We tested the model on a real user in the validation set.

- The model correctly ranked **9 true clicks above non-clicks**.  
- It mis-ranked **6 click events**.  
- Roughly **60% agreement** between predicted and actual behaviour.

This illustrates that the model captures meaningful user preferences.

---

## How to Run This Project

```bash
# clone repo
git clone https://github.com/yourname/news-recsys.git
cd news-recsys

# install dependencies
pip install -r requirements.txt

# run full pipeline
python build_embeddings.py
python build_user_embeddings.py
python build_interactions.py
python train_lightgbm.py
python train_deepfm.py
python evaluate.py

  The two 768-D vectors were concatenated:

