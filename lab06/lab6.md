# Lab 5 – Unsupervised Learning

**Group:** Group 10

**Teammates:**

| Student ID | Student Name            |
| ---------- | ----------------------- |
| 61\*\*\_4  | Jephthah Kimanthi David |
| 61\_\*\*2  | Gideon Bamuleseyo       |

### Question 1:

What is the purpose of the neighborhood function in the SOM? How does it change the learning?

**Answer:**
The neighborhood function in a Self-Organizing Map (SOM) determines how strongly the neurons around the winning neuron are updated when learning from an input. The winning neuron receives the strongest update, while nearby neurons receive smaller updates.

This helps the SOM organize similar data points close together and preserves the relationships between neighboring neurons. As training progresses, the neighborhood usually becomes smaller, allowing the SOM to move from broad organization to more precise adjustments.

### Question 2:

A simplistic intruder detection system for a computer network consists of an attempt to categorize users according to (i) the time of day they log in, (ii) the length of time they log in for, (iii) the types of programs they run while logged in, (iv) the number of programs they run while logged in. Suggest how you would train a SOM classifier to perform the categorization. What preprocessing of the data would you do, how much data would you need, and how large would you make the SOM? Do you think that such a system would work for intruder detection?

**Answer:**
I would first collect normal user activity data over a period of time and represent each login session as a feature vector containing the four variables: time of login, session length, types of programs used, and number of programs run.

Before training the SOM, I would preprocess the data by **normalizing the numerical features** so that variables with larger values do not have more influence than the others. Program types could be converted into numerical or binary features indicating which programs were used.

I would use **several weeks or months of login data** to capture normal variations in user behavior. The SOM could start with a relatively small grid, such as **10 × 10 neurons**, and be adjusted depending on the amount of data and number of user behavior patterns.

The SOM would be trained primarily on **normal user activity**. After training, new login sessions could be mapped onto the SOM. Sessions that are very different from the learned patterns could be flagged as possible intrusions.

I think this system **could help with intruder detection**, but it would not be completely reliable. Users' behavior can naturally change, and an intruder could sometimes behave similarly to a legitimate user. Therefore, the SOM would be better used as **one component of an intrusion detection system rather than the only method**.

### Question 3:

A bank wants to detect fraudulent credit card transactions. They have data for lots and lots of transactions (each transaction is an amount of money, a shop, and the time and date) and some information about when credit cards were stolen, and the transactions that were performed on the stolen card. Describe how you could use a competitive learning method to cluster people's transactions together to identify patterns, so that stolen cards can be detected as changes in pattern. How well do you think this would work? There is much more data of transactions when cards are not stolen, compared to stolen transactions. How does this affect the learning, and what can you do about it?

**Answer:**
I would use a competitive learning method such as a **Self-Organizing Map (SOM)** to cluster transactions based on features such as the transaction amount, shop, and time/date. The SOM would learn the normal patterns of transactions for each card or group of similar cards.

Most transactions would be normal, so the SOM would primarily learn what **normal spending behavior** looks like. When a card is stolen, its transactions may suddenly differ from its usual pattern—for example, unusual amounts, unfamiliar shops, or transactions at unusual times. A large change from the learned pattern could then be flagged as possible fraud.

I think this could work **reasonably well for detecting unusual behavior**, but it would not detect every fraudulent transaction. Some fraudulent transactions may look similar to normal transactions, while legitimate transactions can sometimes look unusual.

The fact that there are **far more normal transactions than stolen/fraudulent transactions** creates an imbalance in the training data. The SOM may become very good at representing normal behavior but may not learn enough about fraudulent patterns.

To address this, I could **give more weight to fraudulent transactions, oversample the stolen-card transactions, or undersample some of the normal transactions**. I could also train primarily on normal transactions and treat transactions that are significantly different from the learned normal patterns as potential fraud. This would allow the system to focus on detecting anomalies rather than requiring large amounts of fraudulent data.

### Question 4:

Consider the following data set consisting of the scores of two variables on each of seven individuals (subjects). This data set is to be grouped into two clusters using k-Means clustering algorithm

### Given data

| Subject | A   | B   |
| ------- | --- | --- |
| 1       | 1.0 | 1.0 |
| 2       | 1.5 | 2.0 |
| 3       | 3.0 | 4.0 |
| 4       | 5.0 | 7.0 |
| 5       | 3.5 | 5.0 |
| 6       | 4.5 | 5.0 |
| 7       | 3.5 | 4.5 |

Initialization - As a first step in finding a sensible initial partition, let the A & B values of the two individuals furthest apart (using the Euclidean distance measure), define the initial cluster means, giving:

|              | Individual | Mean Vector (centroid) |
| ------------ | ---------- | ---------------------- |
| **Group 1:** | 1          | **(1.0, 1.0)**         |
| **Group 2:** | 4          | **(5.0, 7.0)**         |

---

Answer:

## Iteration 1 — Assign each subject to the closest centroid

We calculate Euclidean distance:

[
d=\sqrt{(A-A_c)^2+(B-B_c)^2}
]

For example, Subject 2:

Distance to Group 1:

[
\sqrt{(1.5-1)^2+(2-1)^2}
=\sqrt{1.25}
\approx1.118
]

Distance to Group 2:

[
\sqrt{(1.5-5)^2+(2-7)^2}
=\sqrt{37.25}
\approx6.103
]

So Subject 2 belongs to **Group 1**.

Doing this for all subjects:

| Subject | Distance to G1 | Distance to G2 | Cluster |
| ------- | -------------- | -------------- | ------- |
| 1       | 0.000          | 7.211          | **G1**  |
| 2       | 1.118          | 6.103          | **G1**  |
| 3       | 3.606          | 3.606          | **G1**  |
| 4       | 7.211          | 0.000          | **G2**  |
| 5       | 4.717          | 2.500          | **G2**  |
| 6       | 5.315          | 2.062          | **G2**  |
| 7       | 4.301          | 2.915          | **G2**  |

> Subject 3 is exactly the same distance from both centroids. Using the usual tie-breaking convention, we assign it to **Group 1**.

### New centroids

**Group 1:** Subjects 1, 2, 3

[
A=\frac{1+1.5+3}{3}=1.833
]

[
B=\frac{1+2+4}{3}=2.333
]

So:

[
\boxed{G1=(1.833,2.333)}
]

**Group 2:** Subjects 4, 5, 6, 7

[
A=\frac{5+3.5+4.5+3.5}{4}=4.125
]

[
B=\frac{7+5+5+4.5}{4}=5.375
]

So:

[
\boxed{G2=(4.125,5.375)}
]

---

# Iteration 2

Now use:

- G1 = **(1.833, 2.333)**
- G2 = **(4.125, 5.375)**

After calculating the distances again:

| Subject | Distance to G1 | Distance to G2 | Cluster |
| ------- | -------------- | -------------- | ------- |
| 1       | 1.572          | 5.376          | **G1**  |
| 2       | 0.471          | 4.276          | **G1**  |
| 3       | 2.034          | 1.777          | **G2**  |
| 4       | 5.640          | 1.846          | **G2**  |
| 5       | 3.145          | 0.729          | **G2**  |
| 6       | 3.771          | 0.530          | **G2**  |
| 7       | 2.734          | 1.075          | **G2**  |

Now Subject 3 moves from **G1 → G2**.

So we have:

- **G1:** Subjects 1, 2
- **G2:** Subjects 3, 4, 5, 6, 7

### Calculate new centroids

For G1:

[
A=\frac{1+1.5}{2}=1.25
]

[
B=\frac{1+2}{2}=1.5
]

[
\boxed{G1=(1.25,1.50)}
]

For G2:

[
A=\frac{3+5+3.5+4.5+3.5}{5}=3.9
]

[
B=\frac{4+7+5+5+4.5}{5}=5.1
]

[
\boxed{G2=(3.90,5.10)}
]

---

# Iteration 3

Now use:

- G1 = **(1.25, 1.50)**
- G2 = **(3.90, 5.10)**

The assignments are:

| Subject | Distance to G1 | Distance to G2 | Cluster |
| ------- | -------------- | -------------- | ------- |
| 1       | 0.559          | 5.022          | **G1**  |
| 2       | 0.559          | 3.920          | **G1**  |
| 3       | 3.052          | 1.421          | **G2**  |
| 4       | 6.657          | 2.195          | **G2**  |
| 5       | 4.161          | 0.412          | **G2**  |
| 6       | 4.776          | 0.608          | **G2**  |
| 7       | 3.750          | 0.721          | **G2**  |

**No subjects change clusters.**

Therefore, the centroids remain:

[
\boxed{G1=(1.25,1.50)}
]

[
\boxed{G2=(3.90,5.10)}
]

Since the assignments and centroids no longer change, **k-Means has converged**. We don't need all 5 iterations.

---

### Final Answer

### **Group 1**

Subjects:

[
\boxed{1,\ 2}
]

Final centroid:

[
\boxed{(1.25,\ 1.50)}
]

### **Group 2**

Subjects:

[
\boxed{3,\ 4,\ 5,\ 6,\ 7}
]

Final centroid:

[
\boxed{(3.90,\ 5.10)}
]

### Final clustering

| Group       | Subjects          | Final Centroid   |
| ----------- | ----------------- | ---------------- |
| **Group 1** | **1, 2**          | **(1.25, 1.50)** |
| **Group 2** | **3, 4, 5, 6, 7** | **(3.90, 5.10)** |

**The algorithm converges after 3 iterations.**
