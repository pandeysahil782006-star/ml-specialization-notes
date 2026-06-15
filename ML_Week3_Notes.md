Machine Learning Specialization **Course 1: Supervised & Unsupervised Learning** 

Week 3 — Classification, Logistic Regression & Overfitting 

Andrew Ng  |  Coursera  |  by Sahil Pandey, DTU 

## **Section 1: Classification & Why We Need It** 

## **1.1 The Problem with Using Linear Regression for Classification** 

## **Why This Section Matters** 

So far (Week 1-2) we predicted CONTINUOUS numbers (price, SGPA) using Linear Regression. But many real problems need a YES/NO answer: 

- Is this email spam? (yes/no) 

- Is this tumor malignant? (yes/no) 

- Will this transaction be fraud? (yes/no) 

These are CLASSIFICATION problems. y can only take a small set of values (e.g., 0 or 1), not any number on a continuous line. 

## **1.2 Binary Classification** 

- y can only be 0 or 1 (two possible classes/categories) 

- y = 0 is called the 'negative class' (e.g., not spam, benign) 

- y = 1 is called the 'positive class' (e.g., spam, malignant) 

- 'Positive/Negative' doesn't mean good/bad — just labels for presence/absence of something 

## **1.3 Why Linear Regression FAILS Here** 

## **The Core Problem** 

If you fit a straight line f(x)=wx+b to 0/1 labeled data and use a threshold (e.g., predict 1 if f(x) >= 0.5), it sort of works initially. 

BUT: if you add ONE new training example far to the right (an outlier), 

the best-fit LINE SHIFTS significantly. This shifts your threshold and can flip predictions for examples that were correct before! 

CONCLUSION: Linear regression is NOT a good algorithm for classification. We need a model whose output is BOUNDED (e.g., always between 0 and 1) -> this is exactly what LOGISTIC REGRESSION provides. 

## **Section 2: Logistic Regression & the Sigmoid Function** 

## **2.1 The Sigmoid (Logistic) Function** 

## **Why and Where We Need It** 

We need a function that takes ANY real number as input and squashes it to output a value strictly between 0 and 1 - perfect for representing a PROBABILITY. The Sigmoid function does exactly this. 

**g(z)** `g(z) = 1 / (1 + e^(-z))` 

## **Key Properties of Sigmoid** 

|**Input z**|**Output g(z)**|
|---|---|
|z = 0|g(z) = 0.5  (exact middle)|
|z -> +infinity (large<br>positive)|g(z) -> 1|
|z -> -infinity (large<br>negative)|g(z) -> 0|
|Any z|0 < g(z) < 1  (always between 0 and 1)|



Shape: an 'S-curve' - smooth, continuous, and always bounded between 0 and 1. 

## **2.2 The Logistic Regression Model** 

We take the linear regression output (z = w.x + b) and pass it through sigmoid: 

|**Step 1**|`z = w . x + b   (same as linear regression)`|
|---|---|
|||
|**Step 2**|`f(x) = g(z) = 1 / (1 + e^-(w.x + b))`|



## **2.3 Interpretation: Probability** 

## **THE MOST IMPORTANT INTERPRETATION** 

f(x) is interpreted as the PROBABILITY that y = 1, given input x. 

Example: If f(x) = 0.7 for a tumor classification problem, 

this means: 'There is a 70% chance that y = 1 (malignant)' and therefore a 30% chance that y = 0 (benign). 

Mathematical notation: f(x) = P(y=1 | x) (probability that y=1, given the input features x) 

## **2.4 Example Calculation** 

## **Tumor Size -> Malignancy Probability** 

Suppose model learned: w = 2, b = -8, feature x = tumor size in cm For x = 4.5 cm: 

z = w*x + b = 2*4.5 + (-8) = 9 - 8 = 1 f(x) = g(1) = 1 / (1 + e^-1) = 1 / (1 + 0.368) = 0.731 

Interpretation: 73.1% chance this tumor is malignant (y=1) 

## **Section 3: Decision Boundary** 

## **3.1 From Probability to a Class Prediction** 

## **Why We Need This** 

Logistic regression outputs a PROBABILITY (a number between 0 and 1). But often we need to make a final YES/NO decision 

(e.g., 'flag this email as spam' or 'don't flag'). 

We use a THRESHOLD (commonly 0.5) to convert probability -> class: 

|**Condition**|**Predicted Class**|
|---|---|
|f(x) >= 0.5|Predict y = 1|
|f(x) < 0.5|Predict y = 0|



## **3.2 When is f(x) >= 0.5?** 

**Translating the Sigmoid Condition to z** 

Recall: g(z) >= 0.5 happens exactly when z >= 0 (since g(0) = 0.5, and g is increasing) 

So: f(x) >= 0.5  <=>  z >= 0  <=>  w.x + b >= 0 

Therefore: Predict y=1  if  w.x + b >= 0 Predict y=0  if  w.x + b < 0 

## **3.3 What is the Decision Boundary?** 

## **Definition** 

The DECISION BOUNDARY is the line (or curve) where: 

w.x + b = 0 

This is the boundary that SEPARATES the region where we predict y=1 from the region where we predict y=0. On this exact line, f(x) = 0.5 (model is 'undecided'). 

## **3.4 Example: Linear Decision Boundary** 

## **Two Features (x1, x2)** 

Suppose: w1=1, w2=1, b=-3, so z = x1 + x2 - 3 

Decision boundary: x1 + x2 - 3 = 0  ->  x1 + x2 = 3 This is a STRAIGHT LINE on the x1-x2 plane. 

- If x1+x2 > 3  ->  z > 0  ->  predict y=1 - If x1+x2 < 3  ->  z < 0  ->  predict y=0 

## **3.5 Non-Linear Decision Boundaries** 

## **Using Polynomial Features (from Week 2!)** 

If we add polynomial terms (e.g., x1^2, x2^2) as features, the decision boundary can become a CURVE (e.g., a circle or ellipse), allowing logistic regression to separate data that isn't linearly separable. 

Example: z = x1^2 + x2^2 - 1 

Decision boundary: x1^2 + x2^2 = 1  -> this is a CIRCLE of radius 1! 

- Inside circle (x1^2+x2^2 < 1)  -> predict y=0 

- Outside circle (x1^2+x2^2 > 1) -> predict y=1 

Even more complex polynomial terms -> even more complex boundary shapes. 

**Section 4: Cost Function for Logistic Regression** 

## **4.1 Why Can't We Use the Squared Error Cost Here?** 

## **THE BIG PROBLEM** 

If we plug the logistic regression f(x) = sigmoid(w.x+b) into the Mean Squared Error cost function from Week 1 (J = (1/2m)*sum[(f-y)^2]), 

the resulting J(w,b) becomes NON-CONVEX - it has many wiggly bumps and LOCAL MINIMA. 

This means gradient descent can get STUCK in a local minimum instead of finding the GLOBAL minimum (best possible w,b). 

SOLUTION: Use a DIFFERENT cost function called LOGISTIC LOSS, which IS convex for logistic regression -> guarantees gradient descent converges to the global minimum. 

## **4.2 The Logistic Loss Function** 

The LOSS for a SINGLE training example (i) is defined piecewise based on the true label y: 

|**If y=1**|`Loss = -log(f(x))`|
|---|---|
|||
|**If y=0**|`Loss = -log(1 - f(x))`|



## **Intuition: Why -log()?** 

|**True y**|**Model predicts...**|**Loss behavior**|
|---|---|---|
|y=1|f(x) close to 1 (correct!)|-log(f(x)) -> 0 (LOW loss, good)|
|y=1|f(x) close to 0 (very wrong!)|-log(f(x)) -> infinity (HUGE loss,<br>punished hard)|
|y=0|f(x) close to 0 (correct!)|-log(1-f(x)) -> 0 (LOW loss, good)|
|y=0|f(x) close to 1 (very wrong!)|-log(1-f(x)) -> infinity (HUGE loss,<br>punished hard)|



## **Key Takeaway** 

This loss function HEAVILY PENALIZES confident WRONG predictions and gives almost ZERO loss for confident CORRECT predictions. This shape is what makes the overall cost function J(w,b) CONVEX. 

## **4.3 Simplified Cost Function (Single Combined Formula)** 

Since y is always 0 or 1, both cases above can be written as ONE formula: 

**Loss(f(x),y)** `-y*log(f(x)) - (1-y)*log(1-f(x))` 

## **How This Works (Trick)** 

If y=1: the second term (1-y)=0 vanishes -> Loss = -log(f(x))   [matches case 1] If y=0: the first term y=0 vanishes -> Loss = -log(1-f(x))      [matches case 2] 

One formula covers BOTH cases automatically! 

## **4.4 Overall Cost Function J(w,b)** 

Average the loss over all m training examples: 

**J(w,b)** `J(w,b) = -(1/m) * sum[ y^(i)*log(f(x^(i))) + (1y^(i))*log(1-f(x^(i))) ]` 

This is called the BINARY CROSS-ENTROPY loss (a term you'll see often in deep learning too). 

## **Section 5: Gradient Descent for Logistic Regression** 

## **5.1 The Surprising Result** 

## **Why This Section Matters** 

After taking derivatives of the logistic cost function J(w,b) with respect to w and b, we get update rules that look IDENTICAL in form to linear regression's gradient descent! The only difference is what f(x) means. 

## **5.2 Update Rules** 

Repeat until convergence (simultaneous update): 

**Update wj** `wj = wj - alpha * (1/m) * sum[(f(x^(i))-y^(i)) * x^(i)_j]` **Update b** `b = b - alpha * (1/m) * sum[(f(x^(i)) - y^(i))]` 

## **5.3 The CRUCIAL Difference from Linear Regression** 

||**Linear Regression**|**Logistic Regression**|
|---|---|---|
|f(x) =|w.x + b|sigmoid(w.x + b)|
|Update formula looks|same|same (but f(x) differs!)|
|Cost function|Mean Squared Error|Logistic Loss (cross-entropy)|



## **Don't Get Confused!** 

Even though the UPDATE EQUATIONS look the same, they are completely DIFFERENT algorithms because f(x) is defined differently (linear vs sigmoid-wrapped). The math just happens to simplify nicely. 

## **5.4 Everything from Week 2 Still Applies!** 

- Vectorization (np.dot) - still used to speed up computation 

- Feature scaling - still important for faster convergence 

- Learning curve - still used to check convergence (plot J vs iterations) 

- Choosing alpha - still done by trying 0.001, 0.003, 0.01... etc. 

- scikit-learn - LogisticRegression class works similarly to LinearRegression 

```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)        # gives 0 or 1
probabilities = model.predict_proba(X_test) # gives probability values
```

**Section 6: The Problem of Overfitting** 

## **6.1 Why This Matters - The Real-World Problem** 

## **Setting the Scene** 

A model can achieve PERFECT (or near-perfect) accuracy on the TRAINING data, but perform TERRIBLY on NEW, unseen data. This is one of the most common and important problems in ALL of machine learning - not just logistic regression. 

## **6.2 Three Scenarios: Underfit, Just Right, Overfit** 

|**Term**|**Also Called**|**What's Happening**|
|---|---|---|
|Underfitting|High Bias|Model too simple - doesn't even fit<br>training data well. e.g., fitting a<br>straight line to clearly curved data.|
|Generalizes Well|'Just Right'|Model fits training data reasonably<br>AND performs well on new data. This<br>is the GOAL.|
|Overfitting|High Variance|Model fits training data TOO well<br>(even noise/outliers), but fails badly<br>on new data.|



## **6.3 Bias - An Intuitive Definition** 

## **High Bias (Underfitting)** 

The model has a strong 'preconception' (bias) that the data follows a simple pattern (e.g., a straight line), even when the actual data is more complex. It ignores useful information in the training data. 

Example: trying to fit y = wx + b to data that's clearly a parabola. No matter how you adjust w and b, the line can never capture the curve. 

## **6.4 Variance - An Intuitive Definition** 

## **High Variance (Overfitting)** 

The model is EXTREMELY sensitive to the specific training data used - even small changes/noise in training data lead to wildly different 

models. The model essentially 'memorizes' training data instead of learning general patterns. 

Example: a degree-10 polynomial that wiggles through every single training point perfectly, but oscillates wildly between points - predictions for new x values are nonsensical. 

## **6.5 Overfitting in Classification** 

## **Same Concept, Different Picture** 

- Underfit decision boundary: too simple (e.g., a straight line) that doesn't separate classes well even in training data. 

- Overfit decision boundary: an extremely WIGGLY, complex curve that perfectly separates every single training point (including noise/ outliers), but would misclassify new points badly. 

## **Section 7: Addressing Overfitting** 

## **7.1 Three Main Solutions** 

## **Solution 1: Collect More Training Data** 

## **Why It Helps** 

With more data, even a complex/high-degree model is FORCED to find a smoother fit that generalizes, because it can't 'memorize' a much larger dataset point-by-point. 

LIMITATION: More data isn't always available or affordable. (More relevant for big projects, not always feasible for college projects.) 

## **Solution 2: Feature Selection (Use Fewer Features)** 

## **Why It Helps** 

If you have too many features (especially with small training sets), the model has too much 'freedom' and tends to overfit. Choosing only 

the MOST RELEVANT features reduces this freedom -> simpler model. 

TRADE-OFF: You might lose some useful information if you remove a feature that was actually helpful. (Later courses cover automatic feature selection algorithms.) 

## **Solution 3: Regularization (Most Commonly Used)** 

## **Why It Helps - The Key Idea** 

Instead of REMOVING features entirely, regularization SHRINKS the size of the parameters (w1, w2, ..., wn) toward zero (but not exactly zero usually). This makes the model 'simpler' in effect without fully discarding any feature. 

Note: We typically do NOT regularize b (the bias term). 

## **Section 8: Regularization - The Math** 

## **8.1 The Regularized Cost Function** 

We add a NEW TERM to the cost function that penalizes large weight values: 

**Regularization** `(lambda / 2m) * sum[wj^2]   for j=1 to n` **term** 

## **Putting It All Together (Linear Regression Example)** 

J(w,b) = [Original Cost: (1/2m)*sum((f(x)-y)^2)] + (lambda/2m) * sum(wj^2) 

The same regularization term is ADDED to the logistic regression cost function too (after the cross-entropy part). 

## **8.2 The Regularization Parameter (lambda)** 

|**Lambda Value**|**Effect**|
|---|---|
|lambda = 0|No regularization at all -> may OVERFIT (high variance)|
|lambda very large|wj's forced close to ZERO -> model becomes nearly f(x)=b -> UNDERFIT<br>(high bias)|
|lambda 'just right'|Balances fitting the data well AND keeping weights small -> good<br>generalization|



## **8.3 Regularized Gradient Descent - Linear Regression** 

The update rule for wj gets ONE EXTRA TERM compared to Week 2: 

**Update wj** `wj = wj - alpha * [(1/m)*sum((f(x^(i))y^(i))*x^(i)_j) + (lambda/m)*wj]` **Update b** `b = b - alpha * (1/m) * sum(f(x^(i)) - y^(i))   [b is NOT regularized]` 

## **Intuition for the Extra Term: (lambda/m)*wj** 

This extra term effectively SHRINKS wj a little bit on every iteration, 

in addition to the normal gradient descent update. It's sometimes rewritten as: 

wj = wj * (1 - alpha*lambda/m) - alpha*(1/m)*sum(...) 

Since (1 - alpha*lambda/m) is slightly LESS than 1, wj gets multiplied by a number slightly less than 1 each step -> gradual shrinkage toward 0. 

## **8.4 Regularized Gradient Descent - Logistic Regression** 

## **Same Idea, Same Formula!** 

The update rule for wj is IDENTICAL in form to the regularized linear regression update shown above - just remember f(x) = sigmoid(w.x+b) for logistic regression instead of f(x) = w.x+b. 

wj = wj - alpha * [(1/m)*sum((f(x^(i))-y^(i))*x^(i)_j) + (lambda/m)*wj] b  = b  - alpha * (1/m) * sum(f(x^(i)) - y^(i))    [b not regularized] 

## **Practice Problems - Week 3** 

## **Q1. Sigmoid Function Calculation** 

A logistic regression model has w=1, b=-3, single feature x. 

a) For x=3, compute z = w*x + b, then f(x) = sigmoid(z). b) Will the model predict y=1 or y=0 for this x? (use threshold 0.5) 

## **Solution:** 

a) z = 1*3 + (-3) = 0 f(x) = sigmoid(0) = 1/(1+e^0) = 1/(1+1) = 0.5 

b) f(x) = 0.5, which meets the threshold (f(x)>=0.5) -> Predict y = 1 (Note: x=3 lies EXACTLY on the decision boundary here!) 

## **Q2. Finding the Decision Boundary** 

Given a logistic regression model with two features x1, x2: w1 = 2, w2 = -1, b = 4 

a) Write the equation of the decision boundary (z=0). b) For a point (x1=1, x2=1), which class is predicted? 

## **Solution:** 

a) z = 2*x1 - 1*x2 + 4 = 0  ->  2x1 - x2 + 4 = 0  -> x2 = 2x1 + 4 (This is the decision boundary line) 

b) z = 2*(1) - 1*(1) + 4 = 2 - 1 + 4 = 5 z = 5 > 0  -> Predict y = 1 

## **Q3. Loss Function Reasoning** 

For a single training example, y^(i) = 1. 

a) If the model predicts f(x)=0.99, is the loss high or low? Why? b) If the model predicts f(x)=0.01, is the loss high or low? Why? 

c) Write the simplified loss formula and verify (a) and (b) qualitatively. 

## **Solution:** 

a) LOW loss. The model is very confident y=1 and is CORRECT 

(since true y=1). -log(0.99) is close to 0. 

b) VERY HIGH loss. The model is very confident y=0, but the truth is y=1 - a confident WRONG prediction. -log(0.01) is a large number. 

c) Loss = -y*log(f(x)) - (1-y)*log(1-f(x)) 

Since y=1, second term vanishes: Loss = -log(f(x)) 

-log(0.99) ≈ 0.01 (low)  vs  -log(0.01) ≈ 4.6 (high) - confirms above. 

## **Q4. Bias vs Variance Identification** 

For each scenario, identify if it's UNDERFITTING (high bias), OVERFITTING (high variance), or 'just right': 

a) Training accuracy = 65%, test accuracy = 63% 

b) Training accuracy = 99%, test accuracy = 55% 

c) Training accuracy = 92%, test accuracy = 90% 

d) A degree-1 (straight line) model fit to clearly curved data 

## **Solution:** 

a) UNDERFITTING (high bias) - model performs poorly even on training data, so it hasn't learned the pattern well at all. 

b) OVERFITTING (high variance) - huge gap between train and test performance; model memorized training data. 

c) JUST RIGHT - both train and test accuracy are high and close together. Good generalization! 

d) UNDERFITTING (high bias) - the model (straight line) is too simple to capture the curved pattern in the data. 

## **Q5. Regularization Lambda Effect** 

You train the same model 3 times with different lambda values and 

observe the resulting decision boundaries: 

Model A: lambda = 0       -> very wiggly, complex boundary Model B: lambda = 1       -> smooth, reasonable boundary Model C: lambda = 10000   -> almost a straight line, ignores data shape 

a) Which model is likely OVERFITTING? 

b) Which model is likely UNDERFITTING? 

c) Which model would you prefer for deployment, and why? 

## **Solution:** 

a) Model A (lambda=0) - no regularization, complex wiggly boundary that likely fits training noise -> overfitting (high variance). 

b) Model C (lambda=10000) - regularization is so strong that all weights are pushed near zero -> model becomes too simple 

- -> underfitting (high bias). 

c) Model B (lambda=1) - balances fitting the data shape reasonably while avoiding extreme wiggliness -> best generalization to new data. 

## **Q6. Code Challenge - Sigmoid + Prediction** 

Write Python/NumPy code that: 

1. Defines a sigmoid function 

2. Computes f(x) for w=[1,1], b=-3, X=[[1,1],[2,2],[0,0]] 

3. Converts probabilities to class predictions (threshold 0.5) 

```
import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
w = np.array([1, 1])
b = -3
X = np.array([[1,1],[2,2],[0,0]])
z = X.dot(w) + b          # z values: [-1, 1, -3]
probs = sigmoid(z)        # probabilities
predictions = (probs >= 0.5).astype(int)  # 0 or 1
print('z:', z)
```

```
print('probabilities:', probs)
print('predictions:', predictions)
# z = [-1, 1, -3]
# probs ≈ [0.269, 0.731, 0.047]
# predictions = [0, 1, 0]
```

## **Quick Revision Cheatsheet** 

|**Concept**|**Key Point**|
|---|---|
|Classification|y takes discrete values (e.g., 0/1) - binary classification|
|Why not Linear Reg?|Outputs unbounded values, sensitive to outliers, doesn't give<br>probability|
|Sigmoid g(z)|1/(1+e^-z) - squashes any number to (0,1)|
|Logistic Regression|f(x) = sigmoid(w.x+b) = P(y=1|x)|
|Decision Boundary|w.x+b = 0 - the line/curve separating y=0 and y=1 regions|
|Threshold 0.5|f(x)>=0.5 -> predict 1; f(x)<0.5 -> predict 0|
|Why not MSE cost?|MSE + sigmoid = non-convex -> local minima issues|
|Logistic Loss|-y*log(f(x)) - (1-y)*log(1-f(x)) - convex!|
|GD for Logistic Reg|Same formula as linear reg, but f(x)=sigmoid(w.x+b)|
|Underfitting|High bias - model too simple, poor even on training data|
|Overfitting|High variance - fits training data perfectly, fails on new data|
|Fix 1: More data|Forces model to generalize instead of memorize|
|Fix 2: Feature selection|Remove irrelevant/redundant features|
|Fix 3: Regularization|Add (lambda/2m)*sum(wj^2) to cost - shrinks weights|
|Lambda=0|No regularization -> risk of overfitting|
|Lambda too large|Weights -> 0 -> risk of underfitting|
|b is never regularized|Only w1...wn get the regularization term|



_Course 1 complete, Sahil! This is the foundation for everything ahead - solid work._ 

