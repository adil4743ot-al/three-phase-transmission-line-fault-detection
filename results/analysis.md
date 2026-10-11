# Task 6 — Analysis of Results

## 6.1 Why are Logistic Regression Set R and Linear SVM results similar?

Both models use the same six original features and learn a linear decision boundary. Logistic Regression Set R achieves approximately 73.08% validation accuracy, while the Linear SVM achieves approximately 72.58%. Their similar performance suggests that the original features are not sufficiently separable using a simple linear boundary.

## 6.2 What do the squared-feature weights in Set Q indicate?

The learned weights for the squared current features are positive, while the squared voltage-feature weights are negative. This suggests that larger squared current values contribute positively to the model's fault prediction, whereas larger squared voltage values contribute negatively, keeping the other features fixed. These weights describe the model's learned relationships and do not, by themselves, prove physical causation.

## 6.3 How do the support-vector counts compare?

The selected Linear SVM with C = 0.1 has 5,849 support vectors, while the selected RBF SVM with C = 100 and gamma = 1 has 266 support vectors. The linear model relies on many training samples to define its margin. The RBF model uses a nonlinear boundary and has fewer support vectors in the selected configuration. A lower support-vector count alone does not guarantee a better model; validation and test performance must also be considered.

## 6.4 What happens when C increases?

The Linear SVM validation accuracy remains approximately 72.58% across the tested C values. RBF SVM performance improves as C increases in the tested configurations. The best validation accuracy is approximately 99.67%, achieved with C = 100 and gamma = 1. The number of support vectors does not always decrease monotonically for every RBF configuration, so both accuracy and support-vector counts should be examined.

## 6.5 Can a random split give an optimistic estimate of real-world performance?

Yes. If nearby time samples are strongly correlated, a random split can place similar observations in both training and test sets. This may make test performance look better than performance on a genuinely new time period. A time-based evaluation using a later, independent period would better assess how the model may perform on future transmission-line data.

## 6.6 Which model would you choose for fault detection?

The RBF SVM is the strongest candidate among the four tested models. Its test accuracy is approximately 99.58%, with 6 missed faults and 4 false alarms. Logistic Regression Set Q achieves approximately 97.88% test accuracy, with 45 missed faults and 6 false alarms. Since missed faults can be particularly serious in transmission-line protection, the RBF SVM is preferable based on these results. Before deployment, it should also be tested on independent operating periods and evaluated for reliability and response time.
