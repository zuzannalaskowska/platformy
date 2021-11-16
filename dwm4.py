# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model_dwm.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

#Xbox_d = {0:"PlayStation",1:"Xbox"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="Aplikacja do predykcji platformy")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://www.cnet.com/a/img/1GbaIKM64K1D3F4g0Ku5Nmlj1D8=/940x0/2020/08/13/37136d82-9e71-4642-a4c1-4d5e75da9583/ps5-v-xbox.png")

	with overview:
		st.title("Aplikacja do predykcji platformy")

	#with left:
	#	Xbox_radio = st.radio( "Platforma", list(Xbox_d.keys()), format_func=lambda x : Xbox_d[x] )

	with right:
		year_slider = st.slider("Rok wydania", min_value=1994, max_value=2016)
		NA_sales_slider = st.slider("Sprzedaż w Ameryce Północnej", min_value=0.0, max_value=15.0)
		JP_sales_slider = st.slider("Sprzedaż w Japonii", min_value=0.0, max_value=4.13)
		EU_sales_slider = st.slider("Sprzedaż w Europie", min_value=0.0, max_value=9.09)
		Other_sales_slider = st.slider("Pozostała sprzedaż", min_value=0.0, max_value=10.57)
		user_score_slider = st.slider("Ocena użytkowników", min_value=0.0, max_value=9.0)
		critic_score_slider = st.slider("Ocena recenzentów", min_value=13.0, max_value=98.0)
		critic_count_slider = st.slider("Liczba recenzentów", min_value=3.0, max_value=113.0)
		user_count_slider = st.slider("Liczba użytkowników", min_value=4.0, max_value=10179.0)

	data = [[year_slider, NA_sales_slider,  JP_sales_slider, EU_sales_slider, Other_sales_slider, user_score_slider, critic_score_slider, critic_count_slider, user_count_slider]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Która to platforma?")
		st.subheader(("Xbox" if survival[0] == 1 else "PlayStation"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
