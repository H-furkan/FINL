from rag.data import (
    _classify_doc_type,
    _extract_conditions,
    _extract_drug_name,
    _extract_side_effects,
    build_condition_index,
    build_drug_index,
    build_side_effect_index,
    load_data,
)


class TestExtractDrugName:
    def test_single_letter(self):
        assert _extract_drug_name("Drug A is used for hypertension.") == "Drug A"

    def test_double_letter(self):
        assert _extract_drug_name("Drug AX may cause fatigue.") == "Drug AX"

    def test_no_drug(self):
        assert _extract_drug_name("This has no drug name.") is None


class TestClassifyDocType:
    def test_side_effect_doc(self):
        assert _classify_doc_type("Drug A may cause dizziness.") == "side_effect"
        assert (
            _classify_doc_type("Side effects of Drug D include drowsiness.")
            == "side_effect"
        )

    def test_usage_doc(self):
        assert (
            _classify_doc_type("Drug A is commonly used to treat hypertension.")
            == "usage"
        )


class TestExtractConditions:
    def test_hypertension(self):
        text = "Drug A is commonly used to treat hypertension."
        assert "hypertension" in _extract_conditions(text)

    def test_diabetes(self):
        text = "Drug B is prescribed for type 2 diabetes."
        assert "diabetes" in _extract_conditions(text)

    def test_multiple_conditions(self):
        text = "Drug C is used to treat bacterial infections such as pneumonia and bronchitis."
        conds = _extract_conditions(text)
        assert "infections" in conds


class TestExtractSideEffects:
    def test_may_cause(self):
        effects = _extract_side_effects(
            "Drug A may cause side effects such as dizziness headache and fatigue in some patients."
        )
        assert len(effects) > 0

    def test_include_pattern(self):
        effects = _extract_side_effects(
            "Side effects of Drug D include drowsiness confusion and potential dependency with long-term use."
        )
        assert len(effects) > 0
        assert "drowsiness" in effects[0] or any("drowsiness" in e for e in effects)


class TestBuildDrugIndex:
    def test_all_50_drugs_found(self):
        df = load_data()
        drug_index = build_drug_index(df)
        assert len(drug_index) == 50

    def test_drug_a_structure(self):
        df = load_data()
        drug_index = build_drug_index(df)
        a = drug_index["Drug A"]
        assert len(a["usage_docs"]) == 1
        assert len(a["side_effect_docs"]) == 1
        assert len(a["all_docs"]) == 2
        assert "hypertension" in a["conditions"]
        assert len(a["side_effects"]) > 0

    def test_every_drug_has_both_doc_types(self):
        df = load_data()
        drug_index = build_drug_index(df)
        for drug_name, info in drug_index.items():
            assert len(info["usage_docs"]) >= 1, f"{drug_name} missing usage doc"
            assert len(info["side_effect_docs"]) >= 1, (
                f"{drug_name} missing side_effect doc"
            )

    def test_every_drug_has_conditions(self):
        df = load_data()
        drug_index = build_drug_index(df)
        for drug_name, info in drug_index.items():
            assert len(info["conditions"]) >= 1, f"{drug_name} has no conditions"

    def test_every_drug_has_side_effects(self):
        df = load_data()
        drug_index = build_drug_index(df)
        for drug_name, info in drug_index.items():
            assert len(info["side_effects"]) >= 1, (
                f"{drug_name} has no side effects extracted"
            )


class TestConditionIndex:
    def test_hypertension_maps_to_drug_a(self):
        df = load_data()
        drug_index = build_drug_index(df)
        cond_index = build_condition_index(drug_index)
        assert "Drug A" in cond_index["hypertension"]

    def test_diabetes_maps_to_drug_b(self):
        df = load_data()
        drug_index = build_drug_index(df)
        cond_index = build_condition_index(drug_index)
        assert "Drug B" in cond_index["diabetes"]

    def test_infections_has_multiple_drugs(self):
        df = load_data()
        drug_index = build_drug_index(df)
        cond_index = build_condition_index(drug_index)
        assert len(cond_index["infections"]) >= 3


class TestSideEffectIndex:
    def test_dependency_maps_to_drugs(self):
        df = load_data()
        drug_index = build_drug_index(df)
        se_index = build_side_effect_index(drug_index)
        dependency_drugs = []
        for key, drugs in se_index.items():
            if "dependency" in key:
                dependency_drugs.extend(drugs)
        assert len(dependency_drugs) >= 2
