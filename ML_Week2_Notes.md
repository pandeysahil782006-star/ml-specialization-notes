# Machine Learning Specialization **Course 1: Supervised & Unsupervised Learning** Week 2 — Regression with Multiple Input Variables 

Andrew Ng  |  Coursera  |  by Sahil Pandey, DTU 

## **Section 1: Multiple Features** 

## **1.1 From One Feature to Many** 

In Week 1, our model used a single feature (e.g., house size). Now we extend this to use MULTIPLE features for better predictions. 

## **Example - House Price Prediction with Multiple Features** 

x1 = size in sq ft x2 = number of bedrooms x3 = number of floors x4 = age of home in years y  = price (target) 

More features usually = more information = better predictions 

## **1.2 New Notation** 

|**Symbol**|**Meaning**|
|---|---|
|n|Number of features|
|x^(i)|Feature vector of i-th training example (a row, all features)|
|x^(i)_j|Value of feature j in i-th training example|
|w1, w2,...wn|Weight for each feature|
|b|Bias term (single number)|
|w (vector)|[w1, w2, ..., wn] - all weights together|
|x (vector)|[x1, x2, ..., xn] - all features together|



## **1.3 Multiple Linear Regression Model** 

**f(x)** `f(x) = w1*x1 + w2*x2 + ... + wn*xn + b` 

In vector / dot product form (this is what's actually used in code): 

**Vector form** `f(x) = w . x + b   (dot product of w and x, plus b)` 

## **Example Calculation** 

Suppose: w = [0.1, 4, 10, -2],  b = 80 And a house: x = [1200, 3, 2, 15]  (size, bedrooms, floors, age) 

f(x) = 0.1*1200 + 4*3 + 10*2 + (-2)*15 + 80 = 120 + 12 + 20 - 30 + 80 = 202  (price in some unit, e.g., lakhs) 

## **Section 2: Vectorization** 

## **2.1 What is Vectorization?** 

## **Core Idea** 

Vectorization means writing code using vector/matrix operations (NumPy) instead of explicit for-loops. It makes code: 

1. SHORTER - one line instead of a loop 

2. FASTER - NumPy uses parallel hardware (SIMD/GPU) under the hood 

## **2.2 Without vs With Vectorization** 

## **Without Vectorization (for-loop) - SLOW** 

```
f = 0
for j in range(n):
    f = f + w[j] * x[j]
f = f + b
```

## **With Vectorization (NumPy dot product) - FAST** 

```
import numpy as np
f = np.dot(w, x) + b
```

## **2.3 Why is Vectorization Faster?** 

- NumPy uses optimized C code under the hood 

- Operations run in PARALLEL on multiple values at once (SIMD) 

- On GPUs, this parallelism scales to thousands of operations simultaneously 

- For large n (many features) or large datasets, the speed-up is HUGE 

## **2.4 Vectorized Gradient Descent Update** 

For ALL weights w1...wn at once, instead of updating one by one: 

## **Without vectorization:** 

```
for j in range(n):
```

```
    w[j] = w[j] - alpha * d[j]   # d[j] = derivative wrt w[j]
```

## **With vectorization:** 

```
w = w - alpha * d   # d is a vector of all derivatives
```

NumPy applies the subtraction and multiplication element-wise across the entire vector in one operation. 

## **Section 3: Gradient Descent for Multiple Linear Regression** 

## **3.1 Updated Cost Function** 

Same MSE idea, but now f(x) uses the vector w and dot product: 

**J(w,b)** `J(w,b) = (1/2m) * sum[(f(x^(i)) - y^(i))^2]   where f(x)=w.x+b` 

## **3.2 Gradient Descent Update Rules** 

Repeat until convergence (simultaneously update all wj and b): 

**Update wj** `wj = wj - alpha * (1/m) * sum[(f(x^(i))-y^(i)) * x^(i)_j]` **Update b** `b = b - alpha * (1/m) * sum[(f(x^(i)) - y^(i))]` 

## **Key Points** 

- This is done for EVERY feature j = 1, 2, ..., n 

- All updates happen SIMULTANEOUSLY using OLD w and b values 

- This is still BATCH gradient descent - uses all m examples each step 

## **3.3 An Alternative: Normal Equation** 

**Normal Equation (mentioned as an alternative to GD)** 

- Solves for w, b directly using a closed-form linear algebra formula 

- Only works for LINEAR REGRESSION (not other models) 

- Can be slow when number of features (n) is very large 

- Some ML libraries (e.g., scikit-learn) use this under the hood 

- Gradient Descent is still the recommended/general method to learn 

## **Section 4: Feature Scaling** 

## **4.1 The Problem: Features on Different Scales** 

## **Example - House Price Problem** 

x1 = size of house (range: 300 - 2000 sq ft)   -> LARGE range x2 = number of bedrooms (range: 0 - 5)          -> SMALL range 

Because x1's range is much bigger than x2's, the cost function J(w1,w2) becomes a very ELONGATED, narrow bowl (like a tall thin oval). 

This makes gradient descent BOUNCE BACK AND FORTH and converge SLOWLY. 

## **4.2 The Solution: Scale All Features to a Similar Range** 

After scaling, the cost function contours look more like CIRCLES, so gradient descent moves directly toward the minimum - much faster convergence. 

## **4.3 Three Common Scaling Methods** 

## **A) Simple Max Scaling (Divide by max)** 

**Formula** `x_scaled = x / max(x)` 

Example: if size ranges 300-2000, divide every value by 2000 -> range becomes 0.15 to 1.0 

## **B) Mean Normalization** 

**Formula** `x_scaled = (x - mu) / (max - min)` **Where:** mu = average (mean) value of feature x Result: values typically range from about -1 to +1, centered around 0 

## **C) Z-score Normalization** 

**Formula** `x_scaled = (x - mu) / sigma` **Where:** mu = mean of feature x 

sigma = standard deviation of feature x Result: most values fall roughly between -2 and +2 

## **4.4 Rule of Thumb for Feature Scaling** 

|**Feature Range**|**Action**|
|---|---|
|-3 to +3 (approx)|Generally OK, no scaling needed|
|-0.001 to +0.001|Too small - should rescale|
|100 to 100,000|Too large - should rescale|
|Any very different ranges between<br>features|Always rescale all features|



## **Section 5: Checking Gradient Descent for Convergence** 

## **5.1 The Learning Curve** 

## **How to Check Convergence** 

Plot J(w,b) (cost) on Y-axis vs Number of Iterations on X-axis. This is called the LEARNING CURVE. 

What to look for: 

- J should DECREASE after every iteration 

- If J ever INCREASES -> learning rate (alpha) is too large, OR there's a bug 

- When the curve FLATTENS OUT -> gradient descent has CONVERGED 

## **5.2 Automatic Convergence Test** 

## **Epsilon (e) Threshold Test** 

Declare convergence if J decreases by less than a small value epsilon (e) (e.g., 0.001) in one iteration. 

If J(w,b) decrease < epsilon -> STOP, model has converged. 

NOTE: Choosing the right epsilon is tricky - the learning curve graph is usually a more RELIABLE way to check convergence visually. 

**Section 6: Choosing the Learning Rate** 

## **6.1 Diagnosing Problems Using the Learning Curve** 

|**Symptom**|**Likely Cause & Fix**|
|---|---|
|J increases over iterations|alpha too large -> decrease alpha|
|J goes up and down (oscillates)|alpha too large OR bug in code -> check signs in update<br>equations|
|J decreases VERY slowly|alpha too small -> increase alpha|
|J decreases smoothly to a flat line|Good! alpha is appropriate|



## **6.2 Debugging Tip** 

## **Sanity Check** 

With a VERY SMALL learning rate (e.g., alpha = 0.000001), 

J should DECREASE on every single iteration (even if very slowly). 

If J does NOT decrease even with tiny alpha -> there's a BUG 

in your gradient descent code (likely a sign error, e.g., w = w + alpha*... instead of w = w - alpha*...) 

## **6.3 How to Choose Alpha - Practical Approach** 

Try a range of values, increasing roughly 3x each time, and plot the learning curve for each: 

```
alpha values to try:
0.001  ->  0.003  ->  0.01  ->  0.03  ->  0.1  ->  0.3  ->  1
Pick the largest alpha that still gives a smoothly decreasing J,
(sometimes pick slightly smaller than that for safety margin)
```

## **Section 7: Feature Engineering & Polynomial Regression** 

## **7.1 Feature Engineering** 

**Definition** 

Feature engineering = using domain knowledge/intuition to design NEW features, often by transforming or combining existing features, to make the model more accurate or easier to learn. 

## **Example - House Price Prediction** 

Original features: x1 = frontage (width), x2 = depth 

Engineered feature: x3 = x1 * x2  (this is simply the AREA/LOT SIZE) 

Now model becomes: f(x) = w1*x1 + w2*x2 + w3*x3 + b Area (x3) might be a much more useful predictor than frontage and depth separately! 

## **7.2 Polynomial Regression** 

## **Core Idea** 

If data doesn't fit a STRAIGHT LINE well, we can fit a CURVE by creating new features that are POWERS of the original feature (x^2, x^3, sqrt(x), etc.) This lets linear regression model NON-LINEAR relationships. 

## **Example: Quadratic Model** 

**Model** `f(x) = w1*x + w2*x^2 + b` 

Here x^2 is treated as ANOTHER feature. Same linear regression algorithm applies. 

## **Example: Cubic Model** 

**Model** `f(x) = w1*x + w2*x^2 + w3*x^3 + b` 

## **Example: Square Root Model** 

**Model** `f(x) = w1*x + w2*sqrt(x) + b` 

## **IMPORTANT - Feature Scaling Becomes CRUCIAL Here** 

If x ranges 1-1000, then x^2 ranges 1 to 1,000,000, and x^3 even bigger! These vastly different scales will SLOW DOWN gradient descent badly. ALWAYS apply feature scaling (e.g., z-score normalization) after creating polynomial features. 

## **7.3 Choosing What Features to Use** 

- Look at the shape of your data (plot it!) - does it look curved? 

- Try different feature combinations: x, x^2, x^3, sqrt(x), x1*x2, etc. 

- Use your DOMAIN KNOWLEDGE (e.g., area = length * width makes sense) 

- Later courses cover automated ways to select good features 

**Section 8: Linear Regression with Scikit-Learn** 

## **8.1 Why Use Scikit-Learn?** 

- Industry-standard ML library in Python 

- Implements gradient descent / normal equation internally - no need to code from scratch 

- Handles feature scaling, model fitting, predictions in a few lines 

## **8.2 Basic Workflow** 

```
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
# 1. Scale features (z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
# 2. Create and train model
model = SGDRegressor()       # uses gradient descent
# OR: model = LinearRegression()   # uses normal equation
model.fit(X_scaled, y_train)
# 3. Make predictions
predictions = model.predict(X_scaled)
# 4. View learned parameters
print('w =', model.coef_)
print('b =', model.intercept_)
```

## **8.3 SGDRegressor vs LinearRegression** 

||**SGDRegressor**|**LinearRegression**|
|---|---|---|
|Method|Gradient Descent|Normal Equation|
|Needs scaling?|YES (important)|Not required|
|Good for large n?|Yes|Slow if n very large|



## **Practice Problems - Week 2** 

## **Q1. Vector Form Calculation** 

Given: w = [2, -1, 0.5], b = 5 And feature vector: x = [4, 3, 10] 

Calculate f(x) = w . x + b 

## **Solution:** 

f(x) = (2*4) + (-1*3) + (0.5*10) + 5 = 8 - 3 + 5 + 5 = 15 

## **Q2. Feature Scaling - Identify the Method** 

For each scaled value, identify which scaling method was used (Max scaling / Mean normalization / Z-score normalization): 

a) x_scaled = x / 2000 b) x_scaled = (x - 1100) / (2000 - 300) c) x_scaled = (x - 1100) / 350   (where 350 = standard deviation) 

## **Answer:** 

a) Max scaling (simple division by maximum value) b) Mean normalization (subtract mean, divide by range: max-min) c) Z-score normalization (subtract mean, divide by std deviation) 

## **Q3. Learning Curve Diagnosis** 

You plot J(w,b) vs iterations and observe the following patterns. What should you do in each case? 

a) J increases sharply after iteration 5 b) J decreases extremely slowly, barely changing after 1000 iterations c) J oscillates up and down repeatedly d) J decreases smoothly and flattens after 200 iterations 

## **Answer:** 

a) alpha too large -> decrease the learning rate 

b) alpha too small -> increase the learning rate 

c) alpha too large (or possible bug) -> decrease alpha, check code 

d) Converged! Model training is working correctly. 

## **Q4. Feature Engineering** 

You're predicting the price of a plot of land. You have: 

x1 = length of plot (in meters) 

x2 = width of plot (in meters) 

a) What new feature could you engineer that might be more predictive of price than x1 and x2 separately? 

b) Write the new hypothesis function f(x) including this feature. 

## **Solution:** 

a) x3 = x1 * x2  (this gives the AREA of the plot, which is usually the most important factor for land price) 

b) f(x) = w1*x1 + w2*x2 + w3*x3 + b where x3 = x1 * x2 

## **Q5. Code Challenge - Vectorized Prediction** 

Given a dataset with 4 features (n=4) and m=3 training examples: 

X = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]   (shape: 3x4) 

w = [1, 0.5, -1, 2],  b = 1 

Write vectorized NumPy code to compute predictions for ALL 3 examples at once (without using a for-loop over examples). 

```
import numpy as np
```

```
X = np.array([[1, 2, 3, 4],
```

```
              [2, 3, 4, 5],
              [3, 4, 5, 6]])
w = np.array([1, 0.5, -1, 2])
b = 1
```

```
# Vectorized prediction for all rows at once
predictions = X.dot(w) + b
print(predictions)
# Manual check for row 1: [1,2,3,4]
# = 1*1 + 2*0.5 + 3*(-1) + 4*2 + 1
# = 1 + 1 - 3 + 8 + 1 = 8
# Output: [8, 9.5, 11]
```

## **Quick Revision Cheatsheet** 

|**Concept**|**Key Point**|
|---|---|
|Multiple Features|Model uses n features: x1...xn instead of just one|
|f(x) = w.x + b|Dot product of weight vector w and feature vector x, plus bias|
|Vectorization|np.dot(w,x) replaces for-loops -> faster, shorter code|
|GD for Multi-Regression|Same update rule, but for each wj using x_j feature values|
|Normal Equation|Alternative to GD; direct formula; only for linear regression|
|Feature Scaling|Bring features to similar ranges -> faster GD convergence|
|Mean Normalization|(x - mu) / (max - min)  -> roughly -1 to 1|
|Z-score Normalization|(x - mu) / sigma  -> roughly -2 to 2|
|Learning Curve|Plot J vs iterations to check convergence (should decrease)|
|Convergence Test|J decrease < epsilon -> converged (or use learning curve)|
|Choosing Alpha|Try 0.001, 0.003, 0.01, 0.03, 0.1... (3x steps)|
|Feature Engineering|Create new features using domain knowledge (e.g., area=L*W)|
|Polynomial Regression|Add x^2, x^3, sqrt(x) as features to fit curves|
|scikit-learn|LinearRegression (normal eqn) / SGDRegressor (gradient descent)|




