import importlib.util
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def ensure_statsmodels():
    if importlib.util.find_spec("statsmodels") is None:
        st.error(
            "Le module `statsmodels` est requis. Installez les dépendances avec "
            "`pip install -r requirements.txt` puis relancez l'application."
        )
        st.stop()
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    return STL, adfuller, SARIMAX

st.set_page_config(page_title="Séries temporelles - Streamlit", layout="wide")

st.title("📈 Application Streamlit - Séries temporelles")

st.markdown(
    """
Cette application vous permet de :
- charger un fichier CSV ou Excel contenant une série temporelle,
- visualiser la série, la décomposition STL et un test de stationnarité,
- entraîner un modèle ARIMA ou SARIMA,
- générer des prévisions et comparer observé vs prédit.
"""
)

uploaded_file = st.file_uploader(
    "Chargez un fichier CSV ou Excel", type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("Aperçu des données")
    st.write(data.head())
    st.write(f"Dimensions : {data.shape[0]} lignes, {data.shape[1]} colonnes")

    columns = data.columns.tolist()
    date_column = st.selectbox("Colonne de date/temps", columns)
    value_column = st.selectbox("Colonne de valeurs", columns)

    series_data = data.copy()
    series_data[date_column] = pd.to_datetime(series_data[date_column], errors="coerce")
    series_data[value_column] = pd.to_numeric(series_data[value_column], errors="coerce")
    series_data = series_data.dropna(subset=[date_column, value_column])
    series_data = series_data.sort_values(date_column)
    series_data = series_data.set_index(date_column)

    st.subheader("Série originale")
    st.line_chart(series_data[value_column])

    st.subheader("Décomposition STL")
    seasonal_period = st.number_input(
        "Période saisonnière (ex: 12 pour mensuel)", min_value=2, value=12
    )

    if len(series_data) >= seasonal_period * 2:
        STL, _, _ = ensure_statsmodels()
        stl = STL(series_data[value_column], period=seasonal_period)
        stl_result = stl.fit()

        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(series_data.index, series_data[value_column], label="Série")
        axes[0].set_title("Série")
        axes[1].plot(series_data.index, stl_result.trend, label="Tendance", color="orange")
        axes[1].set_title("Tendance")
        axes[2].plot(
            series_data.index,
            stl_result.seasonal,
            label="Saisonnalité",
            color="green",
        )
        axes[2].set_title("Saisonnalité")
        axes[3].plot(series_data.index, stl_result.resid, label="Résidu", color="red")
        axes[3].set_title("Résidu")
        for axis in axes:
            axis.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("La série est trop courte pour la décomposition STL.")

    st.subheader("Stationnarité (ADF)")
    _, adfuller, _ = ensure_statsmodels()
    series_values = series_data[value_column].dropna()
    if len(series_values) < 10:
        st.warning("Pas assez de données pour le test ADF (minimum 10 observations).")
    else:
        try:
            adf_stat, adf_p_value, _, _, _, _ = adfuller(series_values)
            st.write(f"Statistique ADF : {adf_stat:.4f}")
            st.write(f"Valeur p : {adf_p_value:.4f}")
            if adf_p_value < 0.05:
                st.success("La série semble stationnaire (p < 0.05).")
            else:
                st.info("La série semble non stationnaire (p ≥ 0.05).")
        except ValueError as error:
            st.warning(f"Test ADF impossible : {error}")

    st.subheader("Modélisation")
    model_choice = st.radio("Choix du modèle", ["ARIMA", "SARIMA"], horizontal=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        p_order = st.number_input("p", min_value=0, value=1, step=1)
    with col2:
        d_order = st.number_input("d", min_value=0, value=1, step=1)
    with col3:
        q_order = st.number_input("q", min_value=0, value=1, step=1)

    seasonal_order = (0, 0, 0, 0)
    if model_choice == "SARIMA":
        st.markdown("### Paramètres saisonniers")
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            seasonal_p = st.number_input("P", min_value=0, value=1, step=1)
        with col5:
            seasonal_d = st.number_input("D", min_value=0, value=1, step=1)
        with col6:
            seasonal_q = st.number_input("Q", min_value=0, value=1, step=1)
        with col7:
            seasonal_m = st.number_input("m (période)", min_value=2, value=12, step=1)
        seasonal_order = (seasonal_p, seasonal_d, seasonal_q, seasonal_m)

    if st.button("Entraîner le modèle"):
        _, _, SARIMAX = ensure_statsmodels()
        model = SARIMAX(
            series_data[value_column],
            order=(p_order, d_order, q_order),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False)
        st.session_state["model_results"] = results
        st.success("Modèle entraîné avec succès.")
        st.text(results.summary())

    st.subheader("Prévision")
    horizon = st.number_input("Horizon de prévision", min_value=1, value=12, step=1)
    zoom_length = st.number_input(
        "Nombre de points récents pour comparaison", min_value=10, value=50, step=10
    )

    freq_guess = pd.infer_freq(series_data.index)
    frequency = st.text_input(
        "Fréquence (si date/temps, ex: D, M, H)", value=freq_guess or "D"
    )

    if st.button("Générer la prévision"):
        results = st.session_state.get("model_results")
        if results is None:
            st.warning("Veuillez d'abord entraîner un modèle.")
        else:
            forecast = results.get_forecast(steps=horizon)
            forecast_series = forecast.predicted_mean
            last_index = series_data.index[-1]

            if isinstance(last_index, pd.Timestamp):
                forecast_index = pd.date_range(
                    start=last_index, periods=horizon + 1, freq=frequency
                )[1:]
                forecast_series.index = forecast_index
            else:
                forecast_series.index = range(len(series_data), len(series_data) + horizon)

            st.subheader("Résultats de la prévision")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series_data.index, series_data[value_column], label="Observé")
            ax.plot(forecast_series.index, forecast_series, label="Prévision", color="orange")
            ax.set_title("Prévision de la série temporelle")
            ax.set_xlabel("Temps")
            ax.set_ylabel("Valeurs")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            st.subheader("Comparaison Observé vs Prévision (période récente)")
            recent_observed = series_data[value_column].iloc[-zoom_length:]
            fig_zoom, ax_zoom = plt.subplots(figsize=(10, 4))
            ax_zoom.plot(recent_observed.index, recent_observed, label="Observé")
            ax_zoom.plot(forecast_series.index, forecast_series, label="Prévision", color="orange")
            ax_zoom.set_title("Zoom sur la période récente")
            ax_zoom.set_xlabel("Temps")
            ax_zoom.set_ylabel("Valeurs")
            ax_zoom.legend()
            ax_zoom.grid(True, alpha=0.3)
            st.pyplot(fig_zoom)

            st.subheader("Tableau des prévisions")
            st.write(forecast_series.to_frame(name="Prévision"))
