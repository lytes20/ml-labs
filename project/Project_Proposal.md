# CS 582 — Machine Learning: Group Project Proposal

**Project Title:** Diabetes Risk Prediction: Deep Learning vs. Classical ML with a Bayesian Uncertainty-Aware Triage Agent

**Group:** Group 10

**Teammates:**
| Student ID | Student Name |
| :--- | :---: |
| 61***4 | Jephthah Kimanthi David |
| 61***2 | Gideon Bamuleseyo |

**Dataset and Tools/Platforms:**
Diabetes Health Indicators Dataset (Kaggle, cleaned/balanced from the CDC's 2015 Behavioral Risk Factor Surveillance System survey by Alex Teboul: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). 253,680 records, 21 lifestyle/health features (mostly categorical, BMI numeric), binary label (diabetic/prediabetic vs. none).

**Project Idea:**
Diabetes screening at population scale requires balancing two competing needs: catching at-risk individuals early and not overwhelming clinicians with false alarms. Most published work on this problem (and most Kaggle notebooks) frames it as a single-model accuracy contest between classical machine learning and deep learning. However, recent comparative work (Ayoade et al., 2025) shows that on tabular survey data like BRFSS, deep learning models do not reliably outperform classical ones — model performance is tightly coupled to dataset scale, feature mix, and label structure, and a feed-forward DNN achieved only a modest F1 of 0.45 on the binary BRFSS subset. This motivates a more nuanced question: rather than picking one "best" model, can we combine models to know *when to trust* a prediction, not just *what* the prediction is?

Our project will train and compare classical models (Logistic Regression, Random Forest, SVM) against a feed-forward deep neural network on the Diabetes Health Indicators Dataset, replicating the kind of comparison in Ayoade et al. (2025) as a baseline. We will then layer a Bayesian Network on the best-performing model's outputs to estimate per-case prediction uncertainty, inspired by the Bayesian uncertainty-quantification approach used for diabetic retinopathy imaging in Akram et al. (2025), adapted here to tabular survey data instead of images. This powers a simple triage agent: confidently-negative cases are auto-cleared, confidently-positive cases are auto-flagged for follow-up, and genuinely uncertain cases are escalated for human review. We will evaluate not just raw accuracy, but the practical value of this triage split — e.g., what fraction of cases can be safely automated while keeping the false-negative rate low among escalated/auto-cleared cases.

**Platform/Tools/Programming Language:**
Python 3.x; scikit-learn (Logistic Regression, Random Forest, SVM, preprocessing); PyTorch (feed-forward DNN); pgmpy (Bayesian Network for uncertainty estimation); pandas/NumPy for data handling; Jupyter/Google Colab for experimentation; Git/GitHub for version control.

**Papers to Read:**
1. Sharma, T. & Shah, M. (2021). *A comprehensive review of machine learning techniques on diabetes detection.* Visual Computing for Industry, Biomedicine, and Art, 4:30. https://doi.org/10.1186/s42492-021-00097-7
2. Ayoade, O.B., Shahrestani, S., & Ruan, C. (2025). *Machine Learning and Deep Learning Approaches for Predicting Diabetes Progression: A Comparative Analysis.* Electronics, 14(13), 2583. https://doi.org/10.3390/electronics14132583
3. Akram, M., Adnan, M., Ali, S.F., Ahmad, J., Yousef, A., Alshalali, T.A.N., & Shaikh, Z.A. (2025). *Uncertainty-aware diabetic retinopathy detection using deep learning enhanced by Bayesian approaches.* Scientific Reports. https://doi.org/10.1038/s41598-024-84478-x


**15th Week Milestone:**
A complete end-to-end pipeline with: (1) trained classical (Logistic Regression, Random Forest, SVM) and deep learning (feed-forward DNN) models with a comparative results table (accuracy, F1, AUC) benchmarked against Ayoade et al. (2025)'s published numbers; (2) a Bayesian Network uncertainty layer integrated on top of the best model; (3) a working triage agent that classifies each test case into auto-clear / auto-flag / escalate tiers, with a breakdown of what fraction falls into each tier and the false-negative rate within each; (4) a short analysis of the accuracy-vs-automation tradeoff, ready to present.
