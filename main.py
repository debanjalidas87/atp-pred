import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
import plotly.express as px
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load
from lime import lime_tabular
# %matplotlib inline
#from tabulate import tabulate

header = st.container()
dataset = st.container()
dataanalysis = st.container()
modeling = st.container()
modelaccuracy = st.container()

tab1 , tab2, tab3, tab4 = st.tabs(["Predication :bicyclist:", "Data Analysis :tea:", "Global Performance :weight_lifter:","Data :clipboard:",])

#@st.cache_data
def get_data(filename):
	atp_data = pd.read_csv(filename)

	return atp_data


#with header:
	
	#st.markdown('**Context:** Sports betting projects can be integrated into different types of businesses, depending on the goals, target markets, and resources available as well as the different perspective, such as technical, economic, scientific point of view. The integration will be more accurate if market conditions, and the specific project are analyzed in detail.')

with tab4:
	st.title(' Data Analysis Project on Sporting Bets ')
	st.header('ATP Sports betting dataset Analysis')
	atp_data = get_data('/Users/debanjalidas/Desktop/atp/data/atp_data.csv')
	st.write(atp_data.head())

with tab2:
	st.title('Data Exploration')
	st.header('Please refer to Notebook')
	#st.write(atp_data.describe())
	#st.header('Correlation')
	##st.write(atp_data.corr())
	#fig, ax = plt.subplots()
	#sns.heatmap(atp_data.corr(), ax=ax)
	#st.write(fig)
	#st.header('Statistical Analysis and Data Vizualization')
	#st.markdown('**Lets assess whether a given dataset follows a specific probability distribution or not**')
	#mu, sigma = 32, 18
	#ech = np.random.normal(loc = mu, scale= sigma, size = 4708)
	#plot = sm.qqplot(ech, fit = True, line = '45')
	#st.write(plot)
	#st.markdown('**Pearson Correlation test**')
	#correlation, p_value = pearsonr(x=atp_data["proba_elo"], y=atp_data["WRank"])
	#pvalue = print("p-value:", p_value)
	#st.write('    correlation:' , correlation,'    p-value:' ,p_value)
	##missing_percentages = (atp_data.isnull().sum() / len(atp_data)) * 100
	##st.write(missing_percentages)
	#st.header('**Vizualization the matches are being distributed among the top players**')
	#st.markdown('a pie chart for the most competitive players')
	#player_counts = atp_data['Winner'].value_counts()
	#top_players = player_counts.head(25)
	##plt.figure(figsize=(15, 10))
	#fig_pie = px.pie(top_players,values=top_players.values, names=top_players.index)
	##plt.title('Distribution of Matches Among Top Players')
	#st.write(fig_pie)
	#st.markdown('comparison between the elo winners and the probability of their winner')
	#sorted_data = atp_data.sort_values('elo_winner')
	#elo_ranges = [(1800, 2000), (2000, 2200), (2200, 2400), (2400, 2600), (2600, 2800)]
	#colors = plt.cm.viridis(np.linspace(0, 1, len(elo_ranges)))
	#plt.figure(figsize=(15, 8))
	#for i, (elo_min, elo_max) in enumerate(elo_ranges):
	#	range_data = sorted_data[(sorted_data['elo_winner'] >= elo_min) & (sorted_data['elo_winner'] < elo_max)]
	#	plt.scatter(range_data['elo_winner'], range_data['proba_elo'], color=colors[i], marker='o', label=f'Elo Range {elo_min}-{elo_max}')
	#
	#plt.title('Winning Probability vs. Elo Rating')
	#plt.xlabel('Elo Rating')
	#plt.ylabel('Winning Probability')
	#plt.legend()
	#st.pyplot(plt.gcf())
	#st.markdown('Pie Chart vizualization for the Distribution of Surface')
	#surface_distribution = atp_data['Surface'].value_counts()
	#plt.figure(figsize=(8, 8))
	#plt.pie(surface_distribution, labels=surface_distribution.index, autopct='%1.1f%%')
	#plt.title('Surface Distribution')
	#st.pyplot(plt.gcf())
	#st.markdown('Vizualization of top ten winnwers according to the ground/surface')
	#top_10_players = atp_data['Winner'].value_counts().nlargest(10).index
	#top_10_winners_by_surface = atp_data[atp_data['Winner'].isin(top_10_players)].groupby(['Surface', 'Winner']).size().reset_index(name='Wins')
	#top_10_winners_by_surface = top_10_winners_by_surface.sort_values(['Surface', 'Wins'], ascending=[True, False])
	#surfaces = top_10_winners_by_surface['Surface'].unique()
	#num_surfaces = len(surfaces)
	#colors = ['red', 'green', 'blue', 'orange']
	#plt.figure(figsize=(12, 8))
	#for i, surface in enumerate(surfaces):
	#	surface_data = top_10_winners_by_surface[top_10_winners_by_surface['Surface'] == surface][:10]
	#	plt.subplot(num_surfaces, 1, i+1)
	#	bars = plt.bar(surface_data['Winner'], surface_data['Wins'], color=colors[i % len(colors)])
	#	plt.title(f'Top 10 Tournament Winners by Surface: {surface}')
	#	plt.xlabel('Player')
	#	plt.ylabel('Number of Wins')
	#	plt.xticks(rotation=45)
	#	plt.tight_layout()
	#	for bar in bars:
	#		height = bar.get_height()
	#		plt.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')
	#st.pyplot(plt.gcf())
	#st.header('Cleaned data')
	#st.markdown('**Cleaned:** change text to numeric Drop rows with missing values in PSW , PSL, B365W, B365L removed irrelevent columns Best of, Wsets, Lsets, Comment')
    ##st.markdown('**Context:** Sports betting projects can be integrated into different types of businesses, depending on the goals, target markets, and resources available as well as the different perspective, such as technical, economic, scientific point of view. The integration will be more accurate if market conditions, and the specific project are analyzed in detail.')
    ##st.markdown('**Cleaned:** change text to numeric,Drop rows with missing values in PSW , PSL, B365W, B365L removed irrelevent columns Best of, Wsets, Lsets, Comment')
	atp_cleaned_data = get_data('/Users/debanjalidas/Desktop/atp/data/atp_data_cleaned.csv')
	#st.write(atp_cleaned_data.head())

with tab3:
	#atp_cleaned_data = get_data('/Users/debanjalidas/Desktop/atp/data/atp_data_cleaned.csv')
	st.title(' Modeling Report ')
	#st.header('LogisticRegression')
	X = atp_cleaned_data.drop(columns=['Winning Player'])
	y = atp_cleaned_data['Winning Player']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,)
	feature_columns = ['ATP', 'Location', 'Tournament', 'Series', 'Court', 'Surface','Round', 'Player 2', 'Player 1','Rank_P2','Rank_P1','PS_P2','PS_P1','B365_P2','B365_P1','elo_P2','elo_P1','proba_elo','Year','Month','Day']
	#model1_lr = LogisticRegression()
	#model1_lr.fit(X_train, y_train)
	#train_score_lr = model1_lr.score(X_train, y_train)
	#test_score_lr = model1_lr.score(X_test, y_test)
	#y_pred_lr = model1_lr.predict(X_test)
	#classification_rep = classification_report(y_test, y_pred_lr)
	#conf_mat_fig=plt.figure(figsize=(6,6))
	#ax1 = conf_mat_fig.add_subplot(111)
	#skplt.metrics.plot_confusion_matrix(y_test, y_pred_lr, ax=ax1)
	##st.pyplot(conf_mat_fig,use_continer_width=True)
	#st.write('model: LogisticRegression\n ')
	#st.write('train score :' , train_score_lr,'test score :' ,test_score_lr)
	#st.text('Model Report:\n '+classification_rep)
	#st.pyplot(conf_mat_fig,use_container_width=True)
	##st.text('Confusion Metrics:\n '+confusion_mat)
	#st.header('Decision Tree')
	#model2_dt = DecisionTreeClassifier()
	#model2_dt.fit(X_train, y_train)
	#train_score_dt = model2_dt.score(X_train, y_train)
	#test_score_dt = model2_dt.score(X_test, y_test)
	#y_pred_dt = model2_dt.predict(X_test)
	#classification_rep_dt = classification_report(y_test, y_pred_dt)
	#conf_mat_fig_dt=plt.figure(figsize=(6,6))
	#ax2 = conf_mat_fig_dt.add_subplot(111)
	#skplt.metrics.plot_confusion_matrix(y_test, y_pred_dt, ax=ax2)
	#st.write('model: Decision Tree\n ')
	#st.write('train score :' , train_score_dt,'test score :' ,test_score_dt)
	#st.text('Model Report:\n '+classification_rep_dt)
	#st.pyplot(conf_mat_fig_dt,use_container_width=True)

	st.header('Random Forest Classifier')
	model3_rf = RandomForestClassifier()
	model3_rf.fit(X_train, y_train)
	train_score_rf = model3_rf.score(X_train, y_train)
	test_score_rf = model3_rf.score(X_test, y_test)
	y_pred_rf = model3_rf.predict(X_test)
	classification_rep_rf = classification_report(y_test, y_pred_rf)
	conf_mat_fig_rf=plt.figure(figsize=(6,6))
	ax3 = conf_mat_fig_rf.add_subplot(111)
	skplt.metrics.plot_confusion_matrix(y_test, y_pred_rf, ax=ax3)
	col1, col2 = st.columns(2)
	with col1:
		conf_mat_fig_rf = plt.figure(figsize=(6,6))
		ax1 = conf_mat_fig_rf.add_subplot(111)
		skplt.metrics.plot_confusion_matrix(y_test, y_pred_rf, ax=ax1, normalize=True)
		st.pyplot(conf_mat_fig_rf, use_container_width=True)
	with col2:
		feat_imp_fig = plt.figure(figsize=(6,6))
		ax1 = feat_imp_fig.add_subplot(111)
		skplt.estimators.plot_feature_importances(model3_rf,feature_names=['ATP', 'Location', 'Tournament', 'Series', 'Court', 'Surface','Round', 'Player 2', 'Player 1','Rank_P2','Rank_P1','PS_P2','PS_P1','B365_P2','B365_P1','elo_P2','elo_P1','proba_elo','Year','Month','Day'], ax=ax1, x_tick_rotation=90)
		st.pyplot(feat_imp_fig, use_container_width=True)
	st.divider()
	st.header("Classification Report")
	st.code(classification_report(y_test, y_pred_rf))
	#st.write('model: Decision Tree\n ')
	#st.write('train score :' , train_score_rf,'test score :' ,test_score_rf)
	#st.text('Model Report:\n '+classification_rep_rf)
	#st.pyplot(conf_mat_fig_rf,use_container_width=True)

	#st.header('Support Vector Machine')
	#model4_svm = SVC(max_iter=11)
	#model4_svm.fit(X_train, y_train)
	#train_score_svm = model4_svm.score(X_train, y_train)
	##test_score_svm = model4_svm.score(X_test, y_test)
	#y_pred_svm = model4_svm.predict(X_test)
	#classification_rep_svm = classification_report(y_test, y_pred_svm)
	#conf_mat_fig_svm=plt.figure()
	#ax4 = conf_mat_fig_svm.add_subplot(111)
	#skplt.metrics.plot_confusion_matrix(y_test, y_pred_svm, ax=ax4)
	#st.write('model: Decision Tree\n ')
	#st.write('train score :' , train_score_svm,'test score :' ,test_score_svm)
	#st.text('Model Report:\n '+classification_rep_svm)
	#confusion_mat = confusion_matrix(y_test, y_pred_svm)
	#st.write(confusion_mat)
	#st.pyplot(conf_mat_fig_svm,use_container_width=True)

	#st.header('k-nearest neighbors algorithm ')
	#model5_knn = KNeighborsClassifier()
	#model5_knn.fit(X_train, y_train)
	#train_score_knn = model5_knn.score(X_train, y_train)
	#test_score_knn = model5_knn.score(X_test, y_test)
	#y_pred_knn = model5_knn.predict(X_test)
	#classification_rep_knn = classification_report(y_test, y_pred_knn)
	#conf_mat_fig_knn=plt.figure(figsize=(6,6))
	#ax5 = conf_mat_fig_knn.add_subplot(111)
	#skplt.metrics.plot_confusion_matrix(y_test, y_pred_knn, ax=ax5)
	#st.write('model: Decision Tree\n ')
	#st.write('train score :' , train_score_knn,'test score :' ,test_score_knn)
	#st.text('Model Report:\n '+classification_rep_knn)
	#st.pyplot(conf_mat_fig_knn,use_container_width=True)
	#confusion_mat = confusion_matrix(y_test, y_pred_knn)
	#st.write(confusion_mat)


with tab1:
	st.title(' Live Prediction ')
	#dump(model3_rf,'/Users/debanjalidas/Desktop/atp/model/rf_atp.model')
	load_model3_rf = load('/Users/debanjalidas/Desktop/atp/model/rf_atp.model')
	sliders = []
	col1, col2 = st.columns(2)
	with col1:
		for features in feature_columns:
			feature_slider = st.slider(label=features, min_value=float(atp_cleaned_data[features].min()), max_value=float(atp_cleaned_data[features].max()))
			sliders.append(feature_slider)

	with col2:
		col1, col2 = st.columns(2, gap="medium")

		prediction = load_model3_rf.predict([sliders])
		with col1:
			st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format('Win' if [prediction[0]]==[1] else 'Lost'), unsafe_allow_html=True)

		probs = load_model3_rf.predict_proba([sliders])
		probability = probs[0][prediction[0]]

		with col2:
			st.metric(label="Model Confidence", value="{:.2f} %".format(probability*100), delta="{:.2f} %".format((probability-0.5)*100))

		explainer = lime_tabular.LimeTabularExplainer(X_train.values[:,:], mode="classification",feature_names = X_train.columns)
		explanation = explainer.explain_instance(np.array(sliders), load_model3_rf.predict_proba, num_features=21, top_labels=3)
		interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
		st.pyplot(interpretation_fig, use_container_width=True)




