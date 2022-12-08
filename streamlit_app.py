import streamlit as st
from multiapp import MultiApp
from apps import home,modelFFNN,modelDT,modelKNN,modelCluster # import your app modules here model2

app = MultiApp()

st.markdown("""
#  Inteligencia de Negocios - Grupo B

""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Feed Forward Neural Network", modelFFNN.app)
app.add_app("Modelo Clustering k means", modelCluster.app)
app.add_app("Modelo KNN", modelKNN.app)
app.add_app("Modelo Decision Tree", modelDT.app)
# The main app
app.run()


st.write(st.__version__)



