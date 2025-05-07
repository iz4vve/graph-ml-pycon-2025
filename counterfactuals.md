---
marp: true
theme: rose-pine
paginate: true
auto-scaling: true
---

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

# Counterfactuals for Bias and Fairness analysis


<!-- ![bg left:25%](./imgs/bg.jpg) -->

_A fast introduction using real data & code_

---

## Agenda

1. What are counterfactuals?
2. Why use them for fairness validation?
3. Framework for generating counterfactuals
4. Python example using open-source data
5. Bias measurement and discussion
6. Limitations & best practices

---

## What Are Counterfactuals?

- A counterfactual instance answers:
  - _"What would the prediction be if this input were slightly different?"_

- Especially useful in fairness testing:
  - Change sensitive attributes (e.g., gender, race)
  - Check whether prediction changes unfairly

---

## Simply put...

&nbsp;
Given an input $\mathbf{x}$,  a counterfactual $\mathbf{x}'$  satisfies:

$$
f(\mathbf{x}') \neq f(\mathbf{x}),\quad
\mathbf{x}' \approx \mathbf{x} 
$$

&nbsp;
In essence, the counterfactual optimisation is:
$$
\mathbf{x}' = \arg\min_{\mathbf{z}} \; \mathcal{L}(f(\mathbf{z}), y') + \lambda \cdot \text{dist}(\mathbf{x}, \mathbf{z})
$$
---

## A practical example
&nbsp;
&nbsp;
$$
\mathbf{x}_{\text{original}} = [\text{age}=40, \text{sex}=0, \text{edu}=\text{Bachelors}] 
$$
$$
\mathbf{x}_{\text{cf}} = [\text{age}=40, \text{sex}=1, \text{edu}=\text{Bachelors}]
$$

&nbsp;
&nbsp;
&nbsp;
If $f(\mathbf{x}_{\text{cf}}) \neq f(\mathbf{x}_{\text{original}})$, there may be bias.


---


## Why Counterfactuals for Fairness?

- **Model validation** beyond accuracy/recall
- Identify **individual fairness** violations
- Use for both **diagnosis** and **auditing**

#### Example:
If two applicants are similar but differ only in race and are classified differently → unfair

---

## Dataset: UCI Adult Census Income

- Predict if income > $50K
- Features: age, sex, education, hours/week, etc.
- Label: income (<=50K or >50K)

Source: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/adult)

---

## Load and Clean the Data

```python
url = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    'adult/adult.data'
)

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "race",
    "sex", "marital-status", "occupation", "relationship", "native-country",
    "capital-gain", "capital-loss", "hours-per-week",  "income"
]

data = pd.read_csv(url, names=columns, na_values="?")
data.dropna(inplace=True)
```

---

## Encode and Split Data

```python
for col in data.select_dtypes(include='object').columns:
    data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop('income', axis=1)
y = data['income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Train a Baseline Model

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## Counterfactual Generation Function

```python
def generate_counterfactual(instance, feature, values):

    counterfactuals = []
    for val in values:
        cf = instance.copy()
        cf[feature] = val
        counterfactuals.append(cf)
    return pd.DataFrame(counterfactuals)
```

---

## Generate Counterfactual for One Person

```python
sample = X_test.iloc[[0]]
cf_data = generate_counterfactual(sample.iloc[0], 'sex', [0, 1])
```

---

## Predict Original and Counterfactuals

```python
original_pred = model.predict(sample)[0]
cf_preds = model.predict(cf_data)

print("Original Prediction:", original_pred)
print("Counterfactual Predictions:", cf_preds)
```

---

## Interpretation Example

- Original prediction: `>50K`
- Changing only `sex` switches prediction to `<50K`
- Suggests **potential gender bias**

---

## Measuring Counterfactual Bias Rate

```python
def test_group_bias(X, model, sensitive_feature, values):

    count_changed = 0
    for i in range(len(X)):
        instance = X.iloc[i]
        cf_data = generate_counterfactual(
            instance, sensitive_feature, values
        )

        preds = model.predict(cf_data)
        if len(set(preds)) > 1:
            count_changed += 1
    return count_changed / len(X)
```

---

## Evaluate Bias on Sampled Test Set

```python
bias_rate = test_group_bias(
    X_test.sample(200), model, 'sex', [0, 1]
)

print(f"Bias Rate due to 'sex': {bias_rate:.2%}")
```

---

## Results: Bias Evaluation

- On a sample of 200 individuals:
  - ~**12–15%** changed prediction when `sex` attribute flipped
  - Suggests measurable **gender sensitivity** in output
- Could indicate model relies too much on gender-related features

---

## Explore Bias Across Attributes

```python
race_bias = test_group_bias(
    X_test.sample(200), model, 'race', list(X['race'].unique())
)

print(f"Bias Rate due to 'race': {race_bias:.2%}")
```

---

## More Insights

- Bias due to race: ~8–10% depending on encoding
- Counterfactual testing helps compare **sensitive feature effects**
- Consider combining with **SHAP or LIME** for feature explanation

---

## Limitations of Counterfactuals

- May not be **realistic** (e.g., inconsistent features)
- Sensitive to model instability
- Does not capture **group-level unfairness** alone

---

## Best Practices

- Use **causal reasoning** when possible
- Ensure **valid, meaningful counterfactuals**
- Combine with **group metrics** (e.g., demographic parity)
- Consider multiple attributes and larger samples

---

## Key Takeaways

- Counterfactuals help reveal **individual unfairness**
- Easy to apply with scikit-learn pipelines
- Requires careful design and validation
- Part of a **bigger fairness toolkit**

---

## References & Tools

- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- [AI Fairness 360](https://aif360.mybluemix.net/)
- [Dice-ML](https://github.com/interpretml/DiCE)
- [Fairlearn](https://fairlearn.org/)

---

## Thank You!

Questions?

