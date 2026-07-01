## Machine Learning Specialization **Course 1: Supervised & Unsupervised Learning** Week 1 — Complete Notes with Practice Problems 

Andrew Ng  |  Coursera  |  by Sahil Pandey, DTU 

## **Section 1: What is Machine Learning?** 

## **1.1 Definition** 

## **Arthur Samuel's Definition (1959)** 

"Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed." 

More modern take: An ML system learns from experience (data) to improve performance on a task. 

## **1.2 Why Machine Learning?** 

Some problems are too complex to solve with hand-coded rules. ML allows us to: 

- Recognize patterns in huge amounts of data 

- Adapt automatically as new data comes in 

- Solve problems where we can't write explicit rules (speech, images, recommendations) 

## **1.3 Two Main Categories** 

|**Supervised Learning**|**Unsupervised Learning**|
|---|---|
|Input-output pairs given|No labels — find structure|
|Model learns from labeled data|Model finds hidden patterns|
|Regression, Classification|Clustering, Dimensionality Reduction|
|Spam filter, Price prediction|Customer segmentation, Anomaly detection|



**Section 2: Supervised Learning** 

## **2.1 Core Concept** 

In supervised learning, you provide the algorithm with: 

- Input features: x (e.g., house size, temperature, image pixels) 

- Correct output labels: y (e.g., price, spam/not-spam) 

- The algorithm learns a function f such that: f(x) ≈ y 

## **2.2 Types of Supervised Learning** 

## **A) Regression** 

## **Regression = Predicting a CONTINUOUS output** 

Output can be any number along a range. 

Examples: 

- Predict house price given size (sq ft) -> output: Rs 45,00,000 

- Predict tomorrow's temperature -> output: 34.5°C 

- Predict a student's SGPA -> output: 7.8 

## **B) Classification** 

## **Classification = Predicting a DISCRETE category/class** 

Output is one of a finite set of categories. 

## Examples: 

- Email: spam or not spam (binary: 2 classes) 

- Tumor: malignant or benign (binary) 

- Handwritten digit: 0,1,2,...,9 (multi-class: 10 classes) 

## **Section 3: Unsupervised Learning** 

## **3.1 Core Concept** 

Data has NO labels. The algorithm must find structure or patterns on its own. You only give input x, no y. Algorithm discovers hidden structure. 

## **3.2 Types of Unsupervised Learning** 

## **A) Clustering** 

- Groups similar data points together into clusters 

- Algorithm decides how many / which clusters make sense 

- Example: Google News groups related stories automatically 

- Example: Grouping DTU students by similar course performance 

- Example: DNA microarray analysis — group people with similar genes 

## **B) Anomaly Detection** 

- Finds unusual/rare data points 

- Example: Credit card fraud detection 

- Example: Detecting faulty machines in a factory 

## **C) Dimensionality Reduction** 

- Compress data using fewer variables while preserving information 

- Example: PCA (Principal Component Analysis) 

- Useful for visualization and speeding up algorithms 

**Section 4: Regression Model — Linear Regression** 

## **4.1 The Model** 

Linear Regression is the simplest ML model. It fits a straight line to the data. 

## **Notation (Very Important!)** 

|**Symbol**|**Meaning**|
|---|---|
|m|Number of training examples (rows in dataset)|
|x|Input feature (e.g., house size)|
|y|Output / target variable (e.g., price)|
|(x, y)|Single training example|
|(x^i, y^i)|i-th training example|
|f(x) or y-hat|Model's prediction (estimated y)|
|w|Weight / slope parameter|
|b|Bias / intercept parameter|



## **4.2 The Hypothesis Function** 

**f(x)** `f(x) = wx + b` 

- w = slope — how much y changes per unit of x 

- b = bias — where line intersects y-axis (x=0) 

- Goal: find best w and b so predictions are close to actual y values 

## **4.3 Making Predictions** 

After training (finding good w, b), for any new x: 

**Prediction** `y-hat = w * x + b` 

## **Intuition Example — House Price Prediction** 

x = house size in sq ft y = price in lakhs Suppose learned model: f(x) = 0.5x + 50 

If house size = 1000 sq ft: 

y-hat = 0.5 * 1000 + 50 = 550 lakhs 

Parameters w = 0.5 (price goes up 0.5 lakh per sq ft), b = 50 (base price) 

## **Section 5: Cost Function** 

## **5.1 Why Do We Need a Cost Function?** 

We need a way to measure how WRONG our model is. The cost function quantifies the error between predictions and actual values. 

A smaller cost = better model. Training = minimizing the cost function. 

## **5.2 Mean Squared Error (MSE) — The Cost Function** 

|**J(w,b)**|`J(w,b) = (1/2m) * sum[(f(x^i) - y^i)^2]  for i=1 to m`|`J(w,b) = (1/2m) * sum[(f(x^i) - y^i)^2]  for i=1 to m`|
|---|---|---|
||||
|**Part**||**Meaning**|
|f(x^i) - y^i||Error for i-th example (prediction minus actual)|
|(...)^2||Square it so negatives don't cancel positives|
|sum[...]||Add up squared errors across all m examples|
|1/2m||Average + extra 1/2 to make derivative cleaner|



## **5.3 Cost Function Intuition** 

Imagine plotting J(w) for different values of w (fixing b=0 for simplicity): 

- If w is perfect -> predictions match exactly -> J = 0 

- If w is too big or too small -> errors are large -> J is large 

- The cost function forms a U-shape (parabola) with a minimum in the middle 

- Our goal: find the value of w,b that sits at the BOTTOM of this bowl 

## **5.4 Visualizing Cost in 3D** 

When both w and b vary, J(w,b) forms a 3D bowl/surface: 

- X-axis: w, Y-axis: b, Z-axis: J(w,b) 

- Contour plot: shows same-cost curves like topographic map rings 

- Center of smallest ring = minimum cost = best w,b 

## **Section 6: Training the Model — Gradient Descent** 

## **6.1 What is Gradient Descent?** 

## **Core Idea** 

Gradient Descent is an optimization algorithm that helps find the minimum of any function — specifically, the minimum of the cost function J(w,b). 

Analogy: You are blindfolded on a hilly terrain and want to reach the lowest valley. At each step, you feel the slope under your feet and take a small step downhill. You repeat until you reach the bottom. 

## **6.2 Gradient Descent Algorithm** 

Repeat until convergence: 

**Update w** `w = w - alpha * (d/dw) J(w,b)` **Update b** `b = b - alpha * (d/db) J(w,b)` 

## **CRITICAL: Simultaneous Update** 

Both w and b must be updated at the SAME TIME using the OLD values. 

CORRECT: temp_w = w - alpha * dJ/dw temp_b = b - alpha * dJ/db w = temp_w b = temp_b WRONG: updating w first and then using new w to compute b update. 

## **6.3 The Learning Rate (alpha)** 

Alpha (α) is the most important hyperparameter in gradient descent. 

|**Alpha Value**|**What Happens**|
|---|---|
|Too small (0.0001)|Very slow convergence — takes forever to reach minimum|
|Too large (10.0)|Overshoots minimum — may diverge (cost goes UP)|



|**Alpha Value**|**What Happens**|
|---|---|
|Just right (0.01)|Smooth, steady convergence to minimum|



Typical starting values to try: 0.001, 0.003, 0.01, 0.03, 0.1 (3x each time) 

## **6.4 Gradient Descent Intuition** 

## **On the slope (not at minimum):** 

- If on right side: derivative > 0 -> w decreases -> moves left toward minimum 

- If on left side: derivative < 0 -> w increases -> moves right toward minimum 

## **At the minimum:** 

- Derivative = 0 -> w = w - alpha * 0 = w (no change) 

- Algorithm has converged! Parameters stop changing. 

## **6.5 Gradient Descent for Linear Regression** 

Plugging in our specific cost function J(w,b) = (1/2m)*sum[(wx^i+b - y^i)^2]: 

|**dJ/dw**|`(1/m) * sum[(wx^i + b - y^i) * x^i]  for i=1 to m`|
|---|---|
|||
|**dJ/db**|`(1/m) * sum[(wx^i + b - y^i)]  for i=1 to m`|



These derivatives come from calculus (chain rule). You don't need to derive them from scratch in practice, but understand what they represent. 

## **6.6 Batch Gradient Descent** 

## **Important Term** 

The version we're using is called BATCH Gradient Descent because at each step, it uses ALL m training examples to compute the gradient. 

Other variants exist (mini-batch, stochastic) but batch GD is the foundation. 

For linear regression with MSE cost: J(w,b) is CONVEX (bowl-shaped). This means there is ONLY ONE minimum — no local minima to get stuck in! 

**Section 7: Jupyter Notebooks & Python Setup** 

## **7.1 What is Jupyter Notebook?** 

- Interactive coding environment used heavily in ML/Data Science 

- Combines code, output, visualizations, and markdown text in one file 

- Run code cell by cell — great for experimentation 

- Andrew Ng's labs run in Jupyter on Coursera — everything pre-setup 

## **7.2 Key Commands** 

|**Command / Shortcut**|**Action**|
|---|---|
|Shift + Enter|Run current cell, move to next|
|Ctrl + Enter|Run current cell, stay|
|A|Insert cell above (command mode)|
|B|Insert cell below (command mode)|
|DD|Delete cell (command mode)|
|M|Change cell to Markdown|
|Y|Change cell to Code|



**Practice Problems — Week 1** 

## **Q1. Classification or Regression? (Multiple Choice)** 

Identify whether each is a regression or classification problem: 

a) Predicting the number of units a product will sell next month 

b) Determining if a loan applicant will default (yes/no) 

c) Estimating a patient's blood sugar level from lifestyle data 

d) Classifying an X-ray as COVID-positive or COVID-negative 

e) Predicting SGPA of a student from study hours and attendance 

## **Answer:** 

a) Regression (predicting a number: units sold) 

b) Classification (binary: default / not default) 

c) Regression (predicting a number: blood sugar level) 

d) Classification (binary: positive / negative) 

e) Regression (predicting a number: SGPA 0.0 to 10.0) 

## **Q2. Supervised or Unsupervised? (Fill in the blank)** 

a) Google Photos groups your pictures by face — _______________ 

b) A spam filter trained on labeled spam/ham emails — _______________ 

c) Segmenting app users into 'casual', 'regular', 'power' users — _______________ 

d) Training a model to predict stock prices using historical data — _______________ 

e) Finding unusual transactions in banking data without labels — _______________ 

## **Answer:** 

a) Unsupervised (clustering — no labels given) 

b) Supervised (labels: spam = 1, not spam = 0) 

c) Unsupervised (clustering — algorithm groups by behavior) 

d) Supervised (historical prices are labels) 

e) Unsupervised (anomaly detection — no labeled anomalies) 

## **Q3. Cost Function Calculation** 

Given a dataset with 3 training examples and model f(x) = 2x + 1: 

x = [1, 2, 3] y = [3, 5, 7]  (actual values) 

Step 1: Compute predictions y-hat for each x Step 2: Compute the squared errors Step 3: Compute J(w, b) using MSE formula: J = (1/2m) * sum(errors^2) 

## **Solution:** 

Predictions: f(1) = 3, f(2) = 5, f(3) = 7 

Errors: (3-3)=0, (5-5)=0, (7-7)=0 

Squared errors: 0, 0, 0 

J(2, 1) = (1/6) * (0+0+0) = 0 

J = 0! This means w=2, b=1 is a PERFECT fit for this data. 

## **Q4. Gradient Descent — True or False** 

a) If learning rate is too large, gradient descent will always converge slowly. b) At the minimum of J(w,b), both derivatives dJ/dw and dJ/db equal zero. c) In batch gradient descent, we use only one training example per step. d) For linear regression, J(w,b) is convex (no local minima). 

e) We can update w using the new value of b in the same step. 

## **Answer:** 

a) FALSE — if too large, it overshoots and may diverge (cost increases!) b) TRUE — gradient = 0 means we're at the minimum (or saddle point) c) FALSE — batch GD uses ALL m examples per step d) TRUE — MSE cost for linear regression is always a bowl (convex) e) FALSE — must use OLD values of both w and b (simultaneous update) 

## **Q5. Implement in Python (Code Challenge)** 

Write Python code to perform ONE step of gradient descent for linear regression: 

Given: 

m = 3 training examples 

x = [1, 2, 3], y = [2, 4, 6] Current w = 0.5, b = 0, alpha = 0.1 

Task: Compute new values of w and b after one gradient descent step. (Remember: simultaneous update!) 

**==> picture [469 x 284] intentionally omitted <==**

**----- Start of picture text -----**<br>
import numpy as np<br>x = np.array([1, 2, 3])<br>y = np.array([2, 4, 6])<br>m = 3<br>w, b, alpha = 0.5, 0, 0.1<br># Compute predictions<br>y_hat = w * x + b<br># Compute gradients<br>dw = (1/m) * np.sum((y_hat - y) * x)<br>db = (1/m) * np.sum(y_hat - y)<br># Simultaneous update<br>w_new = w - alpha * dw<br>b_new = b - alpha * db<br>print(f'dw={dw:.4f}, db={db:.4f}')<br>print(f'w_new={w_new:.4f}, b_new={b_new:.4f}')<br># Output: dw=-3.5, db=-1.8333, w_new=0.85, b_new=0.1833<br>**----- End of picture text -----**<br>


## **Quick Revision Cheatsheet** 

|**Concept**|**Key Point**|
|---|---|
|Machine Learning|Learns from data without explicit programming|
|Supervised Learning|Uses labeled data (x, y pairs)|
|Regression|Predict continuous output (a number)|
|Classification|Predict discrete category (class)|
|Unsupervised Learning|No labels — find hidden structure|
|Linear Regression|f(x) = wx + b, fits a line to data|
|Cost Function J(w,b)|(1/2m)*sum[(f(x^i)-y^i)^2] — measures error|
|Gradient Descent|Iteratively update w,b to minimize J|
|Learning Rate alpha|Step size — too small=slow, too big=diverge|



|**Concept**|**Key Point**|
|---|---|
|Simultaneous Update|Use OLD w,b to compute BOTH updates together|
|Batch GD|Uses all m examples per gradient step|
|Convex function|Linear reg cost = bowl shape, one minimum|




