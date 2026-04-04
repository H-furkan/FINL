# 🧪 Example Questions & Expected Answers

---

## 🔎 1. Direct Fact Retrieval

### Question:

**“What are the side effects of Drug A?”**

### ✅ Expected Answer:

* Drug A may cause:

  * dizziness
  * headache
  * fatigue

👉 Good systems:

* clearly list side effects
* avoid hallucinating extra info

---

## 🔎 2. Usage-Based Question

### Question:

**“Which drug is used for diabetes?”**

### ✅ Expected Answer:

* Drug B is used for type 2 diabetes
* It helps lower blood sugar by increasing insulin sensitivity

👉 Strong answer:

* names the drug
* briefly explains how it works

---

## 🔎 3. Reverse Lookup

### Question:

**“What treats hypertension?”**

### ✅ Expected Answer:

* Drug A is used to treat hypertension
* It works by relaxing blood vessels

👉 This tests:

* retrieval from description (not side effects)

---

## 🔎 4. Multi-Document Combination

### Question:

**“What are the side effects of Drug M?”**

### ✅ Expected Answer:

* fatigue
* hair loss
* nausea

👉 Good RAG:

* retrieves the correct doc
* extracts clean list

---

## 🔎 5. Comparison-Type Question

### Question:

**“Which drugs are used for infections?”**

### ✅ Expected Answer:

* Drug C → bacterial infections
* Drug AQ → respiratory infections
* Drug AH → eye infections

👉 Strong systems:

* return multiple drugs
* group logically

---

## 🔎 6. Slightly Complex Query

### Question:

**“Which drug is used for asthma and what are its side effects?”**

### ✅ Expected Answer:

* Drug F is used for asthma
* Side effects include:

  * tremors
  * nervousness
  * increased heart rate

👉 Tests:

* retrieving **two related documents**
* combining them correctly

---

## 🔎 7. Safety Question

### Question:

**“Which drugs have dependency risks?”**

### ✅ Expected Answer:

* Drug D → dependency with long-term use
* Drug J → dependency risks

👉 Strong systems:

* detect keywords like “dependency”
* retrieve multiple relevant entries

---

## 🔎 8. Broader Query

### Question:

**“Which drugs affect the brain?”**

### ✅ Expected Answer:

Examples:

* Drug E → affects neurotransmitters (depression)
* Drug D → acts on central nervous system
* Drug O → stabilizes brain activity

👉 Good answer:

* selects relevant subset
* explains briefly

---

## 🔎 9. Unknown / Not Answerable

### Question:

**“Which drug cures cancer completely?”**

### ✅ Expected Answer:

* “I don’t know based on the provided data.”

👉 VERY IMPORTANT:

* Good RAG systems **do not hallucinate**

---

## 🔎 10. Multi-Hop Reasoning (Advanced)

### Question:

**“Which drug is used for cholesterol and what risks does it have?”**

### ✅ Expected Answer:

* Drug G is used for high cholesterol
* Risks include:

  * muscle pain
  * liver enzyme changes

👉 Tests:

* retrieving 2 docs
* combining into one answer

---

# 🏆 What a GOOD Answer Looks Like

* ✅ grounded in dataset
* ✅ concise (not too long)
* ✅ structured (bullet points help)
* ✅ no hallucination

---

# 🚫 What a BAD Answer Looks Like

* ❌ adds info not in dataset
* ❌ wrong drug
* ❌ ignores question
* ❌ too vague

---

# 🎯 How You Can Use These

You can:

* use them during **live demo evaluation**
* give 2–3 of them as **test queries to all teams**
* create a **scoring sheet** based on correctness

