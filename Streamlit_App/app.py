# Application Streamlit (Databricks Apps) — Vehicle Fraud Detection
# Charge le modèle champion depuis le Model Registry MLflow (Unity Catalog) et,
# pour un dossier de sinistre saisi via formulaire (29 features), affiche la
# probabilité de fraude et un verdict feu tricolore au seuil métier ajustable.

import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st

# Dans une Databricks App, le conteneur n'embarque pas le runtime Databricks :
# sans ces deux URI, MLflow chercherait le modèle dans un registre local
# (« Registered Model not found ») alors qu'il vit dans Unity Catalog.
# L'authentification est automatique : Databricks Apps injecte dans
# l'environnement les identifiants OAuth du service principal de l'app.
mlflow.set_tracking_uri("databricks")      # API MLflow du workspace
mlflow.set_registry_uri("databricks-uc")   # registre Unity Catalog (modèles)

st.set_page_config(page_title="Vehicle Fraud Detection", page_icon=":guardsman:", layout="wide")

st.title("Vehicle Fraud Detection")

# ============================================================================
# Onglets : 1) la valeur métier du modèle  2) le formulaire + l'évaluation
# Le premier onglet est affiché au chargement de l'app.
# ============================================================================
tab_benefits, tab_form = st.tabs(["Model business value", "Claim form & assessment"])

# --- Onglet 1 : la valeur métier ---------------------------------------------
with tab_benefits:
    st.header("What the model is worth to the business")
    st.caption("Final evaluation on the temporal test set : 19,963 claims = the most recent 20% of history — "
               "\nXGBoost tuned at the calibrated business threshold (0.1754). "
               "\nSource : Notebook 4 — Expériences ML")

    # ROI, capture, montant moyen : les trois chiffres de l'argumentaire.
    col_roi, col_catch, col_amount = st.columns(3)
    with col_roi:
        st.metric("Model ROI", "×8.6", "€ of fraud avoided per € of investigation")
    with col_catch:
        st.metric("Fraud caught", "88%", "of real frauds detected at threshold 0.1754")
    with col_amount:
        st.metric("Average fraudulent claim", "€7,031", help="Mean claim amount among fraudulent claims (test set)")

    # Comptabilité annuelle complète au point de fonctionnement calibré.
    st.subheader("Annual economics at the calibrated threshold")
    st.markdown(
        "| Item | Amount |\n"
        "| --- | --- |\n"
        "| Alerts raised (3,513 true + 6,053 false positives) | 9,566 |\n"
        "| Investigation budget (9,566 × €300) | €2,869,800 |\n"
        "| Fraud payments avoided (3,513 × €7,031) | €24,701,281 |\n"
        "| **Net benefit per year** | **€21,831,481** |"
    )

    st.success("Investigating all 19,963 claims would cost €5,988,900/year — "
               "the model divides the investigation cost by 2.1 while catching 88% of real frauds.")

    # Le coût d'investigation (300 €) est la seule hypothèse non mesurée dans
    # la donnée ; le montant moyen de 7 031 € est mesuré sur les fraudes du test.
    st.info("Business assumption : €300 investigation cost per claim. "
            "\nThe €7,031 average fraudulent claim is measured from the data; the €300 cost is an hypothesis.")

# --- Onglet 2 : formulaire + évaluation ----------------------------------------
# Chargement du modèle champion (mis en cache pour la durée de vie de l'app ;
# si l'alias « champion » est déplacé sur une nouvelle version, l'app la recharge).
model_version_uri = "models:/workspace.projet_alpha.insurance_fraud_detector@champion"

@st.cache_resource
def load_model(model_version_uri):
    return mlflow.sklearn.load_model(model_version_uri)

with tab_form:

    # RÈGLE DU BLOC SAISIE : chaque widget renvoie sa valeur -> stockée dans une
    # variable portant exactement le nom de la feature du modèle ; les champs de
    # contexte (customer_name, incident_type, incident_location) sont de
    # l'affichage pur et ne partent jamais au modèle. Défauts = médianes de la
    # Gold (dossier typique). Chaque label est unique dans l'app (identifiant de
    # widget Streamlit).
    with st.expander("Customer informations", expanded=True):

        # Affichage pur, jamais envoyé au modèle.
        customer_name = st.text_input("Customer name", value="")

        age = st.number_input("Age (years)", min_value=18, max_value=100, value=51, step=1)

        credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=575, step=1)

        years_with_company = st.number_input("Years with the company", min_value=0, value=9, step=1)

        previous_claims = st.number_input("Previous claims", min_value=0, value=1, step=1)

        annual_income = st.number_input("Annual income ($)", min_value=0.0, value=137840.3, format="%.1f")

    with st.expander("Policy details"):

        policy_duration_years = st.number_input(
            "Policy duration (years)",
            min_value=0, max_value=60,
            value=5,
            step=1
        )

        premium_amount = st.number_input("Premium amount ($)", min_value=0.0, value=2496.9, format="%.1f")

        coverage_amount = st.number_input("Coverage amount ($)", min_value=0.0, value=129342.8, format="%.1f")

        deductible = st.number_input("Deductible ($)", min_value=0.0, value=1000.0, step=50.0, format="%.1f")

    with st.expander("Incident details"):

        witnesses = st.number_input("Witnesses", min_value=0, max_value=10, value=2, step=1)

        injuries = st.number_input("Injuries", min_value=0, max_value=10, value=0, step=1)

        # Libellés au caractère près : c'est ce texte exact que le OneHotEncoder
        # du pipeline reconnaîtra.
        accident_severity = st.selectbox(
            "Accident severity",
            options=["MINOR", "MODERATE", "MAJOR", "TOTAL LOSS"],
            index=0
        )

        # Chaîne "YES"/"NO" -> convertie en 0/1 dans le bloc de dérivation.
        police_report = st.selectbox("Police report filed", options=["YES", "NO"], index=0)

        # Affichage pur (contexte du dossier) : absents des 29 features.
        incident_type = st.selectbox(
            "Incident type",
            options=["SINGLE VEHICLE COLLISION", "PARKED VEHICLE", "VANDALISM",
                     "ANIMAL IMPACT", "THEFT", "MULTI-VEHICLE COLLISION"]
        )

        incident_location = st.selectbox(
            "Incident location",
            options=["PARKING LOT", "RURAL", "HIGHWAY", "URBAN", "SUBURBAN"]
        )

    with st.expander("Vehicle informations"):

        vehicle_age = st.number_input("Vehicle age (years)", min_value=0, max_value=50, value=9, step=1)

        mileage = st.number_input("Mileage (miles)", min_value=0, value=92466, step=1000)

        # min_value > 0 : vehicle_value est le diviseur des deux ratios.
        vehicle_value = st.number_input("Vehicle value ($)", min_value=1.0, value=14241.5, format="%.1f")

    with st.expander("Financial details"):

        repair_cost = st.number_input("Repair cost ($)", min_value=0.0, value=2438.4, format="%.1f")

        towing_cost = st.number_input("Towing cost ($)", min_value=0.0, value=0.0, format="%.1f")

        medical_cost = st.number_input("Medical cost ($)", min_value=0.0, value=0.0, format="%.1f")

        rental_vehicle_cost = st.number_input("Rental vehicle cost ($)", min_value=0.0, value=373.8, format="%.1f")

    with st.expander("Claim details"):

        claim_amount = st.number_input("Claim amount ($)", min_value=0.0, value=3585.4, format="%.1f")

        # Les dates alimentent reporting_delay_days ; l'heure alimente
        # is_night_claim. Défauts : incident J-3, déclaration aujourd'hui
        # (délai typique de 3 jours).
        incident_date = st.date_input(
            "Incident date",
            value=datetime.date.today() - datetime.timedelta(days=3)
        )

        claim_date = st.date_input("Claim date", value=datetime.date.today())

        # 14:30 par défaut : ~85% des dossiers Gold sont déclarés en journée.
        claim_submission_hour = st.time_input("Claim submission time", value=datetime.time(14, 30))

    # Dans un vrai système, ces contrôles s'exécutent automatiquement au dépôt du
    # dossier (analyse des justificatifs, confrontation GPS au lieu déclaré).
    # L'app les simule : le démonstrateur peut les cocher et voir la probabilité
    # de fraude évoluer.
    with st.expander("Automatic verifications"):

        st.caption("Simulated automatic checks performed at claim submission time.")

        # Défaut décoché : 91.9% des dossiers Gold n'ont pas de documents
        # suspects — et quand il y en a, c'est un signal de fraude majeur.
        is_suspicious = st.checkbox("Suspicious documents detected", value=False)

        # Défaut coché : 88% des dossiers Gold ont une position GPS conforme.
        is_gps_location_match = st.checkbox(
            "GPS location matches the declared incident location",
            value=True
        )

    # ============================================================================
    # Dérivation : 22 features viennent des widgets, 7 se calculent ici selon
    # des règles validées sur la table Gold.
    # ============================================================================

    # Déclarer avant l'incident n'existe pas dans la Gold (min = 0).
    reporting_delay_days = max(0, (claim_date - incident_date).days)

    # Règle validée : is_delayed_reported = 1 ssi délai > 5 jours.
    is_delayed_reported = 1 if reporting_delay_days > 5 else 0

    # Approximation documentée : nuit = 22h-6h (l'heure de déclaration n'étant
    # pas dans le dataset, la fenêtre exacte du générateur est inconnue).
    is_night_claim = 1 if (claim_submission_hour.hour >= 22 or claim_submission_hour.hour < 6) else 0

    has_police_report = 1 if police_report == "YES" else 0

    # Ratios rapportés à la valeur du véhicule (vehicle_value > 0 garanti par le widget).
    repair_cost_ratio = repair_cost / vehicle_value
    claim_amount_ratio = claim_amount / vehicle_value

    # Accident grave (MAJOR ou TOTAL LOSS) déclaré sans rapport de police.
    severe_no_police_report = 1 if (accident_severity in ("MAJOR", "TOTAL LOSS") and has_police_report == 0) else 0

    # ============================================================================
    # Prédiction : DataFrame d'une ligne au format de la signature du champion
    # ============================================================================
    # Clés = noms exacts de la signature enregistrée ; checkboxs -> 0/1.
    claim_dict = {
        # --- saisies directes (widgets) ---
        "age": age, "credit_score": credit_score,
        "years_with_company": years_with_company, "previous_claims": previous_claims,
        "annual_income": annual_income, "policy_duration_years": policy_duration_years,
        "premium_amount": premium_amount, "coverage_amount": coverage_amount,
        "deductible": deductible, "witnesses": witnesses, "injuries": injuries,
        "vehicle_age": vehicle_age, "mileage": mileage, "vehicle_value": vehicle_value,
        "repair_cost": repair_cost, "towing_cost": towing_cost,
        "medical_cost": medical_cost, "rental_vehicle_cost": rental_vehicle_cost,
        "claim_amount": claim_amount,
        "accident_severity": accident_severity,      # la seule STRING des 29
        # --- contrôles automatiques simulés (checkboxes -> 0/1) ---
        "is_suspicious": int(is_suspicious),
        "is_gps_location_match": int(is_gps_location_match),
        # --- features dérivées ---
        "is_delayed_reported": is_delayed_reported,
        "is_night_claim": is_night_claim,
        "has_police_report": has_police_report,
        "reporting_delay_days": reporting_delay_days,
        "repair_cost_ratio": repair_cost_ratio,
        "claim_amount_ratio": claim_amount_ratio,
        "severe_no_police_report": severe_no_police_report,
    }

    # Les 29 features, dans l'ordre de la signature.
    feats = ["age", "credit_score", "years_with_company", "previous_claims", "annual_income",
            "policy_duration_years", "premium_amount", "coverage_amount", "deductible",
            "witnesses", "injuries", "vehicle_age", "mileage", "vehicle_value",
            "repair_cost", "towing_cost", "medical_cost", "rental_vehicle_cost", "claim_amount",
            "accident_severity", "is_suspicious", "is_gps_location_match", "is_delayed_reported",
            "is_night_claim", "has_police_report", "reporting_delay_days", "repair_cost_ratio",
            "claim_amount_ratio", "severe_no_police_report"]

    X = pd.DataFrame([claim_dict])[feats]

    # La signature attend 28 doubles + 1 string : cast des colonnes numériques
    # en float (accident_severity exclue).
    num_cols = [f for f in feats if f != "accident_severity"]
    X[num_cols] = X[num_cols].astype(float)

    model = load_model(model_version_uri)

    # Décision brute du modèle à son seuil par défaut (0.5).
    prediction = int(model.predict(X)[0])

    # Colonne 1 de predict_proba = probabilité de FRAUDE.
    fraud_probability = float(model.predict_proba(X)[0, 1])

    # ============================================================================
    # Affichage : verdict, probabilité, feu tricolore au seuil métier
    # ============================================================================
    st.divider()
    st.subheader("Fraud assessment")

    # Le seuil métier AJUSTABLE : défaut = 0.1754, seuil calibré sur le champion
    # XGB (Notebook 4, point mort coût/risque). Le point de fonctionnement de
    # l'alerte se règle côté métier, indépendamment de la décision brute du
    # modèle (0.5).
    business_threshold = st.slider(
        "Business alert threshold",
        min_value=0.0, max_value=1.0,
        value=0.1754, step=0.0001,
        format="%.4f",
        help="Fraud probability above which a claim is flagged for manual review"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model verdict (threshold 0.5)", "FRAUDULENT" if prediction == 1 else "LEGITIMATE")
    with col2:
        st.metric("Fraud probability", f"{fraud_probability:.1%}")

    st.progress(fraud_probability, text=f"Fraud probability: {fraud_probability:.1%}")

    # FEU TRICOLORE au seuil MÉTIER :
    #   vert   : proba < seuil métier        -> traitement standard
    #   orange : seuil métier <= proba < 0.5 -> revue manuelle (triage)
    #   rouge  : proba >= 0.5                -> action directe
    # Un dossier LEGITIMATE (verdict du modèle à 0.5) peut être ORANGE au seuil
    # métier : l'écart entre décision statistique et décision d'entreprise.
    if fraud_probability >= 0.5:
        st.error("🔴 HIGH RISK — immediate action recommended")
    elif fraud_probability >= business_threshold:
        st.warning("🟠 MODERATE RISK — manual review recommended (triage)")
    else:
        st.success("🟢 LOW RISK — standard processing")

    st.caption("The model verdict uses the default 0.5 threshold; the traffic light "
               "uses the business alert threshold. Showing both is deliberate.")
