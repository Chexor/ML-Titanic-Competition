# Experiment Logboek - Titanic Survival Prediction

## Baseline Referentie
| Metric | Waarde |
|---|---|
| Final notebook CV F1 | 0.792 (Gradient Boosting) |
| Final notebook Kaggle | 0.76555 (v5) |
| Beste Kaggle (eerlijk) | 0.76794 (8-feat tuned GB) |

---

## Experimenten Overzicht

### Exp 1: IsCrew (Fare=0 indicator)
| Parameter | Waarde |
|---|---|
| **Features** | 9 (8 base + IsCrew) |
| **Config** | use_is_crew=True, overige default |
| **CV F1 (GB)** | 0.784 |
| **Δ vs baseline** | -0.008 |
| **Kaggle** | Niet getest |
| **Beslissing** | ❌ Niet toevoegen |

### Exp 2: title_pclass Age Imputation
| Parameter | Waarde |
|---|---|
| **Features** | 8 (base) |
| **Config** | age_imputation='title_pclass' |
| **CV F1 (GB)** | 0.781 |
| **Δ vs baseline** | -0.011 |
| **Kaggle** | Niet getest |
| **Beslissing** | ❌ Niet toevoegen |

### Exp 3: AgeMissing Indicator
| Parameter | Waarde |
|---|---|
| **Features** | 9 (8 base + AgeMissing) |
| **Config** | use_age_missing=True |
| **CV F1 (GB)** | 0.784 |
| **Δ vs baseline** | -0.008 |
| **Kaggle** | Niet getest |
| **Beslissing** | ❌ Niet toevoegen |

### Exp 4: FarePerPerson (ipv FareLog)
| Parameter | Waarde |
|---|---|
| **Features** | 8 (FarePerPerson ipv FareLog) |
| **Config** | use_fare_per_person=True, use_fare_log=False |
| **CV F1 (GB)** | 0.784 |
| **Δ vs baseline** | -0.008 |
| **Kaggle** | Niet getest |
| **Beslissing** | ❌ Niet toevoegen |

### Exp 5: 6-Feature Minimal Set
| Parameter | Waarde |
|---|---|
| **Features** | 6 (Pclass, Title, Age, Sex, FareLog, FamilySize) |
| **Config** | use_has_cabin=False, use_embarked_c=False |
| **CV F1 (RF)** | 0.792 |
| **CV F1 (GB)** | 0.788 |
| **Kaggle (ensemble)** | 0.75837 |
| **Beslissing** | ❌ Slechter dan 8 features |

### Exp 6: 8-Feature + Bredere RF Grid
| Parameter | Waarde |
|---|---|
| **Features** | 8 (Pclass, Title, Age, Sex, FareLog, FamilySize, HasCabin, Embarked_C) |
| **RF Grid** | n_est[50-300], max_depth[2-8+None], min_samples_split[2,5,10], min_samples_leaf[1,2,4], max_features[sqrt,log2,None] |
| **GB Grid** | n_est[100-250], lr[0.03-0.15], max_depth[2-5] |
| **CV F1 (GB)** | 0.790 |
| **CV F1 (RF)** | 0.782 |
| **CV F1 (Ensemble)** | 0.787 |
| **Kaggle (GB)** | **0.76794** |
| **Beslissing** | ✅ Nieuwe baseline |

### Exp 7: ExtraTrees Toevoegen
| Parameter | Waarde |
|---|---|
| **Features** | 8 (zelfde als Exp 6) |
| **Modellen** | RF + ET + GB |
| **CV F1 (GB)** | 0.790 |
| **CV F1 (ET)** | 0.773 |
| **CV F1 (Ensemble)** | 0.784 |
| **Kaggle** | Niet getest |
| **Beslissing** | ❌ ExtraTrees helpt ensemble niet |

### Exp 8: InFemaleOnlyGroup (geen volwassen mannen)
| Parameter | Waarde |
|---|---|
| **Features** | 9 (8 base + InFemaleOnlyGroup) |
| **Definitie** | Ticket-groep zonder volwassen mannen (Age ≥ 18) |
| **Groepen** | 58 groepen, 153 passagiers, 66.1% overleving |
| **CV F1 (GB)** | 0.787 |
| **CV F1 (RF)** | 0.788 |
| **CV F1 (ET)** | 0.789 |
| **Kaggle** | Niet getest |
| **Beslissing** | ❌ Geen significante verbetering |

### Exp 9: GroupSurvived OOF (HUIDIG)
| Parameter | Waarde |
|---|---|
| **Features** | 7 (Pclass, Title, Age, Sex, FareLog, GroupSize, GroupSurvived) |
| **Dropped** | Cabin, HasCabin, Embarked_C, Embarked, FamilySize |
| **GroupSize** | Aantal passagiers metzelfde ticket |
| **GroupSurvived** | OOF berekend (per fold, geen leakage) |
| **OOF Stats** | Train: mean=0.173, std=0.364; Test: mean=0.195, std=0.379 |
| **Zero-filled** | 717/891 train, 326/418 test (nieuwe tickets) |
| **CV F1 (GB)** | **0.800** |
| **CV F1 (RF)** | 0.798 |
| **CV F1 (Ensemble)** | 0.799 |
| **Kaggle (GB)** | **0.76076** |
| **Δ Kaggle vs baseline** | -0.007 |
| **Beslissing** | ⚠️ CV beter, Kaggle slechter - mogelijk overfitting op GroupSurvived |

---

### Exp 10: 6 Features ZONDER GroupSurvived
| Parameter | Waarde |
|---|---|
| **Features** | 6 (Pclass, Title, Age, Sex, FareLog, GroupSize) |
| **Dropped** | GroupSurvived (veroorzaakte overfitting) |
| **GB Config** | Early stopping: n_estimators=500, validation_fraction=0.2, n_iter_no_change=20 |
| **CV F1 (GB)** | 0.801 |
| **CV F1 (RF)** | 0.800 |
| **CV F1 (Ensemble)** | 0.801 |
| **Kaggle (GB)** | **0.77033** |
| **Δ Kaggle vs baseline** | **+0.0024** |
| **CV-Kaggle gap** | 0.031 (was 0.039 met GroupSurvived) |
| **Beslissing** | ✅ Nieuwe beste Kaggle score! |

### Exp 11: Feature Importance Analyse
| Methode | Resultaat |
|---|---|
| **Belangrijkste feature** | Title (0.31-0.33 permutation importance) |
| **2e belangrijkst** | Pclass (0.10-0.24) |
| **Sex impact GB** | 0.005 (bijna nul!) |
| **GroupSurvived impact** | 0.055 (laag, veroorzaakt overfitting) |
| **GroupSize impact** | 0.028-0.049 (laag maar nuttig) |

---

### Exp 12: EvacPriority + Sample Age Imputation
| Parameter | Waarde |
|---|---|
| **Features** | 4 (Age, FareLog, GroupSize, EvacPriority) |
| **Age Imputation** | Sample uit echte distributie per Title |
| **EvacPriority** | 1-3=Vrouwen, 4-6=Masters, 7-9=Mannen (per Pclass) |
| **CV F1 (RF)** | 0.799 |
| **CV F1 (GB)** | 0.797 |
| **Kaggle (RF)** | **0.76555** |
| **Δ Kaggle vs beste** | -0.005 |
| **Feature Importance** | EvacPriority: 0.43 (RF), 0.42 (GB) ← DOMINANT |
| **Beslissing** | ❌ Kaggle slechter, maar EvacPriority is sterkste feature |

---

### Exp 13: EvacPriority + FamilyRatio (6 features)
| Parameter | Waarde |
|---|---|
| **Features** | 6 (EvacPriority, Title, Age, FareLog, GroupSize, FamilyRatio) |
| **CV F1 (RF)** | 0.790 |
| **CV F1 (GB)** | 0.790 |
| **Kaggle** | **0.75598** |
| **FamilyRatio Importance** | 0.019-0.033 (zeer laag) |
| **EvacPriority Importance** | 0.23-0.40 (hoog) |
| **Δ Kaggle vs beste** | -0.014 |
| **Beslissing** | ❌ Slechtste resultaat tot nu toe |

---

## Samenvatting Kaggle Scores (geupdate)

| Exp | Beschrijving | CV F1 | Kaggle | Gap |
|---|---|---|---|---|
| **Exp 10** | **6 feat, no GroupSurvived** | **0.801** | **0.77033** | **0.031** |
| Exp 6 | 8 feat, tuned GB | 0.790 | 0.76794 | 0.022 |
| Final notebook v5 | 8 feat, GB | 0.792 | 0.76555 | 0.027 |
| Exp 12 | 4 feat + EvacPriority + Sample Age | 0.799 | 0.76555 | 0.034 |
| Exp 9 | 7 feat + GroupSurvived OOF | 0.800 | 0.76076 | 0.039 |
| Exp 5 | 6 feat, ensemble | 0.792 | 0.75837 | 0.034 |
| Exp 13 | 6 feat + EvacPriority + FamilyRatio | 0.790 | 0.75598 | 0.034 |

## Samenvatting Kaggle Scores (geupdate)

| Exp | Beschrijving | CV F1 | Kaggle | Gap |
|---|---|---|---|---|
| **Exp 10** | **6 feat, no GroupSurvived** | **0.801** | **0.77033** | **0.031** |
| Final notebook v5 | 8 feat, GB | 0.792 | 0.76555 | 0.027 |
| Exp 12 | 4 feat + EvacPriority + Sample Age | 0.799 | 0.76555 | 0.034 |
| Exp 6 | 8 feat, tuned GB | 0.790 | 0.76794 | 0.022 |
| Exp 9 | 7 feat + GroupSurvived OOF | 0.800 | 0.76076 | 0.039 |
| Exp 5 | 6 feat, ensemble | 0.792 | 0.75837 | 0.034 |

### Exp 13: EvacPriority + FamilyRatio (6 features)
| Parameter | Waarde |
|---|---|
| **Features** | 6 (EvacPriority, Title, Age, FareLog, GroupSize, FamilyRatio) |
| **CV F1 (RF)** | 0.790 |
| **CV F1 (GB)** | 0.790 |
| **Kaggle** | **0.75598** |
| **FamilyRatio Importance** | 0.019-0.033 (zeer laag) |
| **EvacPriority Importance** | 0.23-0.40 (hoog) |
| **Δ Kaggle vs beste** | -0.014 |
| **Beslissing** | ❌ Slechtste resultaat tot nu toe |

---

## Samenvatting Kaggle Scores (geupdate)

| Exp | Beschrijving | CV F1 | Kaggle | Gap |
|---|---|---|---|---|
| **Exp 10** | **6 feat, no GroupSurvived** | **0.801** | **0.77033** | **0.031** |
| Exp 6 | 8 feat, tuned GB | 0.790 | 0.76794 | 0.022 |
| Final notebook v5 | 8 feat, GB | 0.792 | 0.76555 | 0.027 |
| Exp 12 | 4 feat + EvacPriority + Sample Age | 0.799 | 0.76555 | 0.034 |
| Exp 9 | 7 feat + GroupSurvived OOF | 0.800 | 0.76076 | 0.039 |
| Exp 5 | 6 feat, ensemble | 0.792 | 0.75837 | 0.034 |
| Exp 13 | 6 feat + EvacPriority + FamilyRatio | 0.790 | 0.75598 | 0.034 |

## Samenvatting Kaggle Scores (geupdate)

| Exp | Beschrijving | CV F1 | Kaggle | Gap |
|---|---|---|---|---|
| Final notebook v5 | 8 feat, GB | 0.792 | 0.76555 | 0.027 |
| Exp 6 | 8 feat, tuned GB | 0.790 | 0.76794 | 0.022 |
| **Exp 10** | **6 feat, no GroupSurvived** | **0.801** | **0.77033** | **0.031** |
| Exp 9 | 7 feat + GroupSurvived OOF | 0.800 | 0.76076 | 0.039 |
| Exp 5 | 6 feat, ensemble | 0.792 | 0.75837 | 0.034 |

## Samenvatting Kaggle Scores

| Exp | Beschrijving | CV F1 | Kaggle |
|---|---|---|---|
| Final notebook v5 | 8 feat, GB | 0.792 | 0.76555 |
| Exp 6 | 8 feat, tuned GB | 0.790 | **0.76794** |
| Exp 5 | 6 feat, ensemble | 0.792 | 0.75837 |
| Exp 9 | 7 feat + GroupSurvived OOF | **0.800** | 0.76076 |

## Belangrijkste Vindingen

1. **Bredere hyperparameter grid** → +0.002 Kaggle (Exp 6)
2. **GroupSurvived OOF** → CV +0.010 maar Kaggle -0.007 (Exp 9)
3. **Minder features** → altijd slechtere Kaggle scores
4. **Extra features** (IsCrew, AgeMissing, FarePerPerson, InFemaleOnlyGroup) → geen verbetering
5. **CV-Kaggle gap** → GroupSurvived vergroot de gap (0.800 CV → 0.761 Kaggle = 0.039 gap)

## Huidige Sandbox Status
- **Features**: 7 (Pclass, Title, Age, Sex, FareLog, GroupSize, GroupSurvived)
- **Modellen**: RF + GB + ExtraTrees (ExtraTrees is per ongeluk nog aanwezig)
- **Beste CV**: GB 0.800
- **Beste Kaggle**: 0.76794 (Exp 6, 8-feat tuned GB)

## Openstaande Vragen
1. Waarom doet GroupSurvived het goed in CV maar slecht op Kaggle?
2. Is de OOF berekening correct?
3. Zou GroupSurvived zonder OOF beter generaliseren?
4. Kunnen we de 8-feature set combineren met GroupSurvived?
