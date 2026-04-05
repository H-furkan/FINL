# Evaluation Questions for RAG Pipeline

50 questions across 8 categories, covering the full drug dataset.

---

## Category 1: Direct Fact Retrieval (Side Effects)

### Q1
**"What are the side effects of Drug E?"**
- **Expected:** insomnia, dry mouth, changes in appetite
- **Type:** direct_fact

### Q2
**"What side effects does Drug O cause?"**
- **Expected:** dizziness, fatigue, coordination problems
- **Type:** direct_fact

### Q3
**"What are the side effects of Drug AE?"**
- **Expected:** weight gain, tremors, fatigue
- **Type:** direct_fact

### Q4
**"Does Drug Z have any side effects?"**
- **Expected:** blurred vision, eye irritation, dryness
- **Type:** direct_fact

### Q5
**"What adverse reactions can Drug K cause?"**
- **Expected:** stomach pain, increased risk of bleeding
- **Type:** direct_fact

### Q6
**"What are the side effects of Drug AD?"**
- **Expected:** insomnia, increased heart rate, anxiety
- **Type:** direct_fact

### Q7
**"Tell me the side effects of Drug AN."**
- **Expected:** bleeding, bruising, fatigue
- **Type:** direct_fact

---

## Category 2: Usage-Based Questions

### Q8
**"Which drug is used to treat epilepsy?"**
- **Expected:** Drug O — stabilizes electrical activity in the brain
- **Type:** usage

### Q9
**"What drug should be prescribed for gout?"**
- **Expected:** Drug AF — reduces uric acid levels
- **Type:** usage

### Q10
**"Which drug is used for Parkinson disease?"**
- **Expected:** Drug Y — improves motor control
- **Type:** usage

### Q11
**"What drug treats chronic kidney disease?"**
- **Expected:** Drug X — treats chronic kidney disease complications
- **Type:** usage

### Q12
**"Which drug is prescribed for ADHD?"**
- **Expected:** Drug W — improves focus and attention
- **Type:** usage

### Q13
**"What drug is used for glaucoma?"**
- **Expected:** Drug Z — reduces eye pressure
- **Type:** usage

### Q14
**"Which drug treats multiple sclerosis?"**
- **Expected:** Drug AM — reduces nerve damage
- **Type:** usage

---

## Category 3: Reverse Lookup (Condition → Drug)

### Q15
**"What treats bipolar disorder?"**
- **Expected:** Drug AE — stabilizes mood
- **Type:** reverse_lookup

### Q16
**"What can I take for acid reflux?"**
- **Expected:** Drug T — reduces stomach acid production
- **Type:** reverse_lookup

### Q17
**"Is there a drug for osteoporosis?"**
- **Expected:** Drug S — strengthens bones and reduces fracture risk
- **Type:** reverse_lookup

### Q18
**"What medication is available for insomnia?"**
- **Expected:** Drug L — helps patients fall asleep faster
- **Type:** reverse_lookup

### Q19
**"What treats anemia?"**
- **Expected:** Drug AA — increases red blood cell production
- **Type:** reverse_lookup

### Q20
**"What drug can help with obesity?"**
- **Expected:** Drug AD — reduces appetite and increases metabolism
- **Type:** reverse_lookup

---

## Category 4: Comparison / Multi-Drug Questions

### Q21
**"Which drugs cause nausea as a side effect?"**
- **Expected:** Multiple drugs including Drug B, Drug H, Drug I, Drug M, Drug S, Drug AF, Drug AO, Drug AW, and others
- **Type:** comparison

### Q22
**"Which drugs are used to treat anxiety?"**
- **Expected:** Drug J (anxiety disorders), Drug AT (anxiety and panic attacks)
- **Type:** comparison

### Q23
**"Which drugs cause dizziness?"**
- **Expected:** Multiple — Drug A, Drug H, Drug J, Drug O, Drug Q, Drug V, Drug Y, Drug AG, Drug AL, Drug AM, Drug AP, Drug AV, and others
- **Type:** comparison

### Q24
**"Which drugs treat skin conditions?"**
- **Expected:** Drug P (bacterial skin infections), Drug AC (severe acne), Drug AR (skin inflammation)
- **Type:** comparison

### Q25
**"Which drugs cause fatigue as a side effect?"**
- **Expected:** Many drugs — Drug A, Drug I, Drug M, Drug O, Drug Q, Drug V, Drug X, Drug AB, Drug AE, Drug AM, Drug AN, Drug AO, Drug AQ, Drug AS, Drug AU, Drug AV, Drug AX, among others
- **Type:** comparison

### Q26
**"What drugs treat hormonal or thyroid problems?"**
- **Expected:** Drug R (thyroid disorders — regulates hormone levels), Drug AJ (hormonal imbalances)
- **Type:** comparison

### Q27
**"Which drugs are used for migraines?"**
- **Expected:** Drug H (narrows blood vessels in the brain), Drug AP (treats migraines and severe headaches)
- **Type:** comparison

---

## Category 5: Complex / Combined Questions

### Q28
**"Which drug is used for depression and what are its side effects?"**
- **Expected:** Drug E — depression; side effects: insomnia, dry mouth, changes in appetite
- **Type:** complex

### Q29
**"What treats arthritis and what risks does it have?"**
- **Expected:** Drug K — treats arthritis by reducing inflammation; risks: stomach pain, increased risk of bleeding
- **Type:** complex

### Q30
**"Which drug is prescribed for heart failure and what are its side effects?"**
- **Expected:** Drug Q — improves heart function and blood circulation; side effects: dizziness, low blood pressure, fatigue
- **Type:** complex

### Q31
**"What drug treats COPD and what are the side effects?"**
- **Expected:** Drug AG — improves airflow; side effects: cough, throat irritation, dizziness
- **Type:** complex

### Q32
**"Which drug treats fungal infections and how does it work?"**
- **Expected:** Drug U — disrupts fungal cell membranes; side effects: nausea, headache, skin rash
- **Type:** complex

### Q33
**"What drug is used for cancer therapy and what side effects should I expect?"**
- **Expected:** Drug M — slows growth of tumor cells; side effects: fatigue, hair loss, nausea
- **Type:** complex

### Q34
**"Which drug treats ulcers and what are its side effects?"**
- **Expected:** Drug AK — protects the stomach lining; side effects: constipation, nausea, headache
- **Type:** complex

---

## Category 6: Safety / Risk Questions

### Q35
**"Which drugs can cause liver problems?"**
- **Expected:** Drug G (liver enzyme changes), Drug AB (liver toxicity)
- **Type:** safety

### Q36
**"Are there any drugs that increase infection risk?"**
- **Expected:** Drug AB (suppresses immune response — can cause infections), Drug AX (can cause infections)
- **Type:** safety

### Q37
**"Which drugs can cause weight gain?"**
- **Expected:** Drug B (weight gain), Drug AE (weight gain), Drug AJ (weight gain)
- **Type:** safety

### Q38
**"Which drugs have risks related to the heart?"**
- **Expected:** Drug F (increased heart rate), Drug AD (increased heart rate), Drug AV (chest discomfort)
- **Type:** safety

### Q39
**"Is Drug D safe for long-term use?"**
- **Expected:** Drug D has a risk of potential dependency with long-term use
- **Type:** safety

### Q40
**"Which drugs can cause allergic reactions?"**
- **Expected:** Drug C — allergic reactions including rash and itching
- **Type:** safety

### Q41
**"Does Drug AC have sensitivity warnings?"**
- **Expected:** Yes — Drug AC may cause sensitivity to sunlight, along with dry skin and irritation
- **Type:** safety

---

## Category 7: Unanswerable / Out-of-Scope Questions

### Q42
**"What is the recommended dosage for Drug B?"**
- **Expected:** "I don't know based on the provided data." (dosage info not in dataset)
- **Type:** unanswerable

### Q43
**"Can Drug A and Drug V be taken together?"**
- **Expected:** "I don't know based on the provided data." (drug interactions not in dataset)
- **Type:** unanswerable

### Q44
**"Which drug is used for malaria?"**
- **Expected:** "I don't know based on the provided data." (no malaria drug in dataset)
- **Type:** unanswerable

### Q45
**"What is the price of Drug M?"**
- **Expected:** "I don't know based on the provided data." (pricing not in dataset)
- **Type:** unanswerable

### Q46
**"Is Drug F safe during pregnancy?"**
- **Expected:** "I don't know based on the provided data." (pregnancy safety not in dataset)
- **Type:** unanswerable

---

## Category 8: Multi-Hop / Reasoning Questions

### Q47
**"If a patient has both hypertension and a dry cough, which drug might be causing the cough?"**
- **Expected:** Drug V — used for high blood pressure and lists dry cough as a side effect
- **Type:** multi_hop

### Q48
**"A patient on a medication for chronic pain is experiencing confusion. Which drug are they likely taking?"**
- **Expected:** Drug D — used for chronic pain; side effects include confusion
- **Type:** multi_hop

### Q49
**"Which drugs used for mental health conditions can cause weight gain?"**
- **Expected:** Drug AE (bipolar disorder — weight gain), Drug AJ (hormonal imbalances — weight gain, if considered). Primarily Drug AE.
- **Type:** multi_hop

### Q50
**"A patient is being treated for allergies but is feeling very drowsy. What drug might they be on and why?"**
- **Expected:** Drug N — used for allergic conditions such as hay fever; drowsiness is a known side effect
- **Type:** multi_hop

---

## Summary

| Category | Question IDs | Count |
|---|---|---|
| Direct Fact Retrieval | Q1–Q7 | 7 |
| Usage-Based | Q8–Q14 | 7 |
| Reverse Lookup | Q15–Q20 | 6 |
| Comparison / Multi-Drug | Q21–Q27 | 7 |
| Complex / Combined | Q28–Q34 | 7 |
| Safety / Risk | Q35–Q41 | 7 |
| Unanswerable | Q42–Q46 | 5 |
| Multi-Hop Reasoning | Q47–Q50 | 4 |
| **Total** | | **50** |
