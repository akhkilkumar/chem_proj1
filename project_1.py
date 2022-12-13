######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
#import pickle
import io
#from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from math import pi
# rdkit library
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
#from rdkit.Chem import AllChem
#from rdkit.Chem import Draw
#from rdkit.Chem import PandasTools
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Fragments
#for similarity
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

#What we'll need for analysis, clustering, etc.

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import datasets, decomposition
from sklearn.manifold import TSNE
#All we need for clustering
from scipy.cluster.hierarchy import dendrogram, linkage
#from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode


original_title = '<p style="font-family:URW Chancery L; color:cyan ; text-align: center; font-size: 40px;">AaDya:A Chemical Sapce Navigator for Dual–Target-Directed-Inhibitor design</p>'
st.markdown(original_title, unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: gold;font-size: 20px'>''Chemical space is a concept in cheminformatics referring to the property space spanned by all possible molecules and chemical compounds adhering to a given set of construction principles and boundary conditions'' </h2>", unsafe_allow_html=True)


# detail and use of app
st.markdown("<h2 style='text-align: center; color: pink;font-size: 18px'>'Chemical space is extremely large and screening of entire libraries is impractical and time consuming. Further, most of the molecules are inactive for particular drug targets. Thus, our focus in this work was to calculate the physicochemical descriptors of known inhibitor molecules to know the physicochemical descriptor distribution among inhibitors. The goal of the present study was to characterize the chemical space of BACE-1 and GSK3 to guide focused library design and reduce the time and computational cost. </h2>", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: pink;font-size: 18px'>'Physicochemical descriptors have been successfully used in earlier studies to design molecules with drug-like oral drug-like properties, and also used to optimize the solubility, permeability and other properties of a molecules (Ajay et al., 1998; Lipinski, 2000; Lajiness et al., 2004; Muegge, 2003). The most accepted and popular implementation of physicochemical descriptors study is Lipinski’s rule of five (Lipinski et al., 2001) and similar kind of implementation were performed for non-oral routes of drug delivery system (Choy and Prausnitz, 2011). These physicochemical descriptors like molecular weight, number of hydrogen bond donor acceptor, logP, polar surface area, number of rotatable bond etc. are extensively used in different studies. These descriptors are used to differentiate between various molecules like human metabolites, toxins, lead like molecules and also used to filter/reduce the size of database for screening </h2>", unsafe_allow_html=True)



# this line of code take file from user and calculate the descriptor and convert into the df1 dataframe
 


#working---------------------
#reading multiple files

with st.sidebar.header('1. For dual inhibitors'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["sdf"], accept_multiple_files=True)
    st.sidebar.markdown("""[Example CSV input file])""")

# get data from the file uploder in target1 and target2 
target1 = pd.DataFrame()
target2 = pd.DataFrame()
#get_data=pd.DataFrame()
for i in range(len(uploaded_file)):
	#st.write(uploaded_file[i].name)
	#st.write(Chem.SDMolSupplier(uploaded_file[i].name))	
	get_data=Chem.SDMolSupplier(uploaded_file[i].name)

	if i ==0:
		target1=get_data
	else:
		target2=get_data







# create a function that  cal. properties of mol
def properties(m):
	tpsa=Descriptors.TPSA(m)
	mlogp=Descriptors.MolLogP(m)
	AR=rdMolDescriptors.CalcNumAromaticRings(m)
	ALC=rdMolDescriptors.CalcNumAliphaticCarbocycles(m)# number of aliphatic (containing at least one non-aromatic bond) carbocycles for a molecule
	ALHR=rdMolDescriptors.CalcNumAliphaticHeterocycles(m)#returns the number of aliphatic (containing at least one non-aromatic bond) heterocycles for a molecule
	AHR=rdMolDescriptors.CalcNumAromaticHeterocycles(m)
	
	ARC=rdMolDescriptors.CalcNumAromaticCarbocycles(m)
	
	HBA=rdMolDescriptors.CalcNumHBA(m)
	HBD=rdMolDescriptors.CalcNumHBD(m)
	HA=rdMolDescriptors.CalcNumHeavyAtoms(m)
	RB=rdMolDescriptors.CalcNumRotatableBonds(m)
	RG=rdMolDescriptors.CalcRadiusOfGyration(m)
	mw=rdMolDescriptors.CalcExactMolWt(m)
	Hmw=Descriptors.HeavyAtomMolWt(m)
	AMolWt=Descriptors.MolWt(m)#
	#analysis of fragment
	Al_COO=Fragments.fr_Al_COO(m,countUnique=True)#Number of aliphatic carboxylic acids
	Al_OH=Fragments.fr_Al_OH(m,countUnique=True)#Number of aliphatic hydroxyl groups
	Al_OH_noTert=Fragments.fr_Al_OH_noTert(m,countUnique=True)#Number of aliphatic hydroxyl groups excluding tert-OH
	ArN=Fragments.fr_ArN(m,countUnique=True)#Number of N functional groups attached to aromatics
	Ar_COO=Fragments.fr_Ar_COO(m,countUnique=True)#Number of Aromatic carboxylic acide
	Ar_N=Fragments.fr_Ar_N(m,countUnique=True)#Number of aromatic nitrogens
	Ar_NH=Fragments.fr_Ar_NH(m,countUnique=True)#Number of aromatic amines
	Ar_OH=Fragments.fr_Ar_OH(m,countUnique=True)#Number of aromatic hydroxyl groups
	COO=Fragments.fr_COO(m,countUnique=True)#Number of carboxylic acids
	COO2=Fragments.fr_COO2(m,countUnique=True)#Number of carboxylic acids
	C_O=Fragments.fr_C_O(m,countUnique=True)#Number of carbonyl O
	C_O_noCOO=Fragments.fr_C_O_noCOO(m,countUnique=True)#Number of carbonyl O, excluding COOH
	C_S=Fragments.fr_C_S(m,countUnique=True)#Number of thiocarbonyl
	HOCCN=Fragments.fr_HOCCN(m,countUnique=True)#Number of C(OH)CCN-Ctert-alkyl or C(OH)CCNcyclic
	Imine=Fragments.fr_Imine(m,countUnique=True)#Number of Imines
	NH0=Fragments.fr_NH0(m,countUnique=True)#Number of T amines
	NH1=Fragments.fr_NH1(m,countUnique=True)#Number of Secondary amines
	NH2=Fragments.fr_NH2(m,countUnique=True)#Number of Primary amines
	N_O=Fragments.fr_N_O(m,countUnique=True)#Number of hydroxylamine groups
	Ndealkylation1=Fragments.fr_Ndealkylation1(m,countUnique=True)#Number of XCCNR groups
	Ndealkylation2=Fragments.fr_Ndealkylation2(m,countUnique=True)#Number of tert-alicyclic amines (no heteroatoms, not quinine-like bridged N)
	Nhpyrrole=Fragments.fr_Nhpyrrole(m,countUnique=True)#Number of H-pyrrole nitrogens
	SH=Fragments.fr_SH(m,countUnique=True)#Number of thiol groups
	aldehyde=Fragments.fr_aldehyde(m,countUnique=True)#Number of aldehydes
	alkyl_carbamate=Fragments.fr_alkyl_carbamate(m,countUnique=True)#Number of alkyl carbamates
	alkyl_halide=Fragments.fr_alkyl_halide(m,countUnique=True)#Number of alkyl halides
	allylic_oxid=Fragments.fr_allylic_oxid(m,countUnique=True)#Number of allylic oxidation sites excluding steroid dienone
	amide=Fragments.fr_amide(m,countUnique=True)#Number of amides
	amidine=Fragments.fr_amidine(m,countUnique=True)#Number of amidine groups
	aniline=Fragments.fr_aniline(m,countUnique=True)#Number of anilines
	aryl_methyl=Fragments.fr_aryl_methyl(m,countUnique=True)#Number of aryl methyl sites for hydroxylation
	azide=Fragments.fr_azide(m,countUnique=True)#Number of azide
	azo=Fragments.fr_azo(m,countUnique=True)#Number of azo
	barbitur=Fragments.fr_barbitur(m,countUnique=True)#Number of barbiturate
	benzene=Fragments.fr_benzene(m,countUnique=True)#Number of benzene rings
	benzodiazepine=Fragments.fr_benzodiazepine(m,countUnique=True)#Number of benzodiazepines with no additional fused rings
	bicyclic=Fragments.fr_bicyclic(m,countUnique=True)#Bicyclic
	diazo=Fragments.fr_diazo(m,countUnique=True)#diazo
	dihydropyridine=Fragments.fr_dihydropyridine(m,countUnique=True)#dihydropyridine
	epoxide=Fragments.fr_epoxide(m,countUnique=True)#epoxide
	ester=Fragments.fr_ester(m,countUnique=True)#ester
	ether=Fragments.fr_ether(m,countUnique=True)#ether
	furan=Fragments.fr_furan(m,countUnique=True)#furan
	guanido=Fragments.fr_guanido(m,countUnique=True)#guanido
	halogen=Fragments.fr_halogen(m,countUnique=True)#number of halogens
	hdrzine=Fragments.fr_hdrzine(m,countUnique=True)#number of hdrzine groups
	hdrzone=Fragments.fr_hdrzone(m,countUnique=True)#number of hdrzone groups
	imidazole=Fragments.fr_imidazole(m,countUnique=True)#number of imidazole rings
	imide=Fragments.fr_imide(m,countUnique=True)#number of imide groups
	isocyan=Fragments.fr_isocyan(m,countUnique=True)#Number of isocyanates
	isothiocyan=Fragments.fr_isothiocyan(m,countUnique=True)#Number of isothiocyanates
	ketone=Fragments.fr_ketone(m,countUnique=True)#Number of ketone
	ketone_Topliss=Fragments.fr_ketone_Topliss(m,countUnique=True)#num of ketones excluding diaryl, a,b-unsat. dienones, heteroatom on Calpha
	lactam=Fragments.fr_lactam(m,countUnique=True)#Number of beta lactams
	lactone=Fragments.fr_lactone(m,countUnique=True)#Number of cyclic esters (lactones)
	methoxy=Fragments.fr_methoxy(m,countUnique=True)#Number of methoxy groups -OCH3
	morpholine=Fragments.fr_morpholine(m,countUnique=True)#Number of morpholine rings
	nitrile=Fragments.fr_nitrile(m,countUnique=True)#Number of nitrile 
	nitro=Fragments.fr_nitro(m,countUnique=True)#    Number of nitro groups
	nitro_arom=Fragments.fr_nitro_arom(m,countUnique=True)#Number of nitro benzene ring substituents
	nitro_arom_nonortho=Fragments.fr_nitro_arom_nonortho(m,countUnique=True)#Number of non-ortho nitro benzene ring substituents
	nitroso=Fragments.fr_nitroso(m,countUnique=True)#Number of nitroso groups, excluding NO2
	oxazole=Fragments.fr_oxazole(m,countUnique=True)#Number of oxazole rings
	oxime=Fragments.fr_oxime(m,countUnique=True)#Number of oxime groups
	para_hydroxylation=Fragments.fr_para_hydroxylation(m,countUnique=True)#Number of para-hydroxylation sites
	phenol=Fragments.fr_phenol(m,countUnique=True)#Number of phenols
	phenol_noOrthoHbond=Fragments.fr_phenol_noOrthoHbond(m,countUnique=True)#    Number of phenolic OH excluding ortho intramolecular Hbond substituents
	phos_acid=Fragments.fr_phos_acid(m,countUnique=True)#Number of phosphoric acid groups
	phos_ester=Fragments.fr_phos_ester(m,countUnique=True)#Number of phosphoric acid groups
	piperdine=Fragments.fr_piperdine(m,countUnique=True)#Number     Number of piperdine rings
	piperzine=Fragments.fr_piperzine(m,countUnique=True)#Number     Number of piperzine rings
	priamide=Fragments.fr_priamide(m,countUnique=True)#Number of primary amides
	prisulfonamd=Fragments.fr_prisulfonamd(m,countUnique=True)#Number of primary sulfonamides
	pyridine=Fragments.fr_pyridine(m,countUnique=True)#    Number of pyridine rings
	quatN=Fragments.fr_quatN(m,countUnique=True)#    Number of quarternary nitrogens
	sulfide=Fragments.fr_sulfide(m,countUnique=True)#Number of thioether
	sulfonamd=Fragments.fr_sulfonamd(m,countUnique=True)#    Number of sulfonamides
	sulfone=Fragments.fr_sulfone(m,countUnique=True)#    Number of sulfone groups
	term_acetylene=Fragments.fr_term_acetylene(m,countUnique=True)#Number of terminal acetylenes
	tetrazole=Fragments.fr_tetrazole(m,countUnique=True)#Number of tetrazole rings
	thiazole=Fragments.fr_thiazole(m,countUnique=True)#    Number of thiazole rings
	thiocyan=Fragments.fr_thiocyan(m,countUnique=True)#Number of thiocyanates
	thiophene=Fragments.fr_thiophene(m,countUnique=True)#Number of thiophene rings
	unbrch_alkane=Fragments.fr_unbrch_alkane(m,countUnique=True)#Number of unbranched alkanes of at least 4 members (excludes halogenated alkanes)
	urea=Fragments.fr_urea(m,countUnique=True)#    Number of urea groups
	#SHAPE ANALYSIS
	NPR1=rdMolDescriptors.CalcNPR1(m)
	NPR2=rdMolDescriptors.CalcNPR2(m)


	




	


	
	



	return (tpsa, mlogp, AR, ALC, ALHR, AHR, HBA,HBD, HA,RB, RG, mw,Hmw,AMolWt,Al_COO,Al_OH,Al_OH_noTert,ArN,Ar_COO,Ar_N,Ar_NH,Ar_OH,COO,COO2,C_O,C_O_noCOO,C_S,HOCCN,Imine,NH0,NH1,NH2,N_O,
Ndealkylation1,Ndealkylation2,Nhpyrrole,SH,aldehyde,alkyl_carbamate,alkyl_halide,allylic_oxid,amide,amidine,aniline,aryl_methyl,
azide,azo,barbitur,benzene,benzodiazepine,bicyclic,diazo,dihydropyridine,epoxide,ester,ether,furan,guanido,
halogen,hdrzine,hdrzone,imidazole,imide,isocyan,isothiocyan,ketone,ketone_Topliss,lactam,lactone,methoxy,morpholine,nitrile,nitro,
nitro_arom,nitro_arom_nonortho,nitroso,oxazole,oxime,para_hydroxylation,phenol,phenol_noOrthoHbond,phos_acid,
phos_ester,piperdine,piperzine,priamide,prisulfonamd,pyridine,quatN,sulfide,sulfonamd,sulfone,term_acetylene,
tetrazole,thiazole,thiocyan,thiophene,unbrch_alkane,urea,NPR1,NPR2,ARC)

# create a empty dataframe that with descriptor name


#create a function loop over all the molecules in target1 or target 2 and call function properties to cal. descriptors of each molecule than convert into dataframe.


def convert_to_df(target):
	df= pd.DataFrame(columns=['tpsa', 'mlogp', 'AR', 'ALC', 'ALHR', 'AHR', 'HBA','HBD', 'HA','RB', 'RG','mw','Hmw','AMolWt','Al_COO','Al_OH','Al_OH_noTert','ArN','Ar_COO','Ar_N','Ar_NH','Ar_OH','COO','COO2','C_O','fr_C_O_noCOO','C_S','HOCCN','Imine','NH0',
'NH1','NH2','N_O','Ndealkylation1','Ndealkylation2','Nhpyrrole','SH','aldehyde','alkyl_carbamate','alkyl_halide','allylic_oxid','amide','amidine',
'aniline','aryl_methyl','azide','azo','barbitur','benzene','benzodiazepine','bicyclic','diazo','dihydropyridine','epoxide','ester','ether','furan',
'guanido','halogen','hdrzine','hdrzone','imidazole','imide','isocyan','isothiocyan','ketone','ketone_Topliss','lactam','lactone','methoxy','morpholine','nitrile',
'nitro','nitro_arom','nitro_arom_nonortho','nitroso','oxazole','oxime','para_hydroxylation','phenol','phenol_noOrthoHbond','phos_acid',
'phos_ester','piperdine','piperzine','priamide','prisulfonamd','pyridine','quatN','sulfide','sulfonamd','sulfone','term_acetylene','tetrazole',
'thiazole','thiocyan','thiophene','unbrch_alkane','urea','NPR1','NPR2','ARC'])
	for mol in target:
		aa=properties(mol)
		#convert tuple into list	
		templist=list(aa)
		#convert Pandas Series is a one-dimensional labeled array capable of holding data of any type
		a=pd.Series(templist, index=df.columns)
		df=df.append(a, ignore_index=True)
	return df

#convert data to dataframe
df_target1=convert_to_df(target1)	
df_target2=convert_to_df(target2)	




#show data table and descriptive analysis
st.header("Descriptive Statistics")

st.markdown("<h2 style='text-align: center; color: gold;font-size: 20px'>''This procedure is used to summarize continuous data. Large volumes of such data may be easily summarized instatistical tables of means, counts, standard deviations, etc. Categorical group variables may be used to calculate summaries for individual groups. '' </h2>",unsafe_allow_html=True)
# list of dtypes to include
df_target1= df_target1.astype(float)
desc1 = df_target1.describe(include='all').transpose()
# for decoration of table
st.header("Target1")
cm = sns.light_palette("green", as_cmap=True)
st.dataframe(desc1.style.set_properties(**{'border': '2.5px solid green','font-size': '14pt','font-weight': 'bold','color': 'red',**{'background-color': 'white'} }))





df_target2= df_target2.astype(float)
desc2 = df_target2.describe(include='all').transpose()
st.subheader('Target2')
st.dataframe(desc1.style.set_properties(**{'border': '2.5px solid green','font-size': '14pt','font-weight': 'bold','color': 'red',**{'background-color': 'white'} }))




#st.dataframe(df2)
#for mol in target2:
	
#	for atom in mol.GetAtomWithIdx(3).GetSymbol():
#		st.write(atom)
	
#frame = PandasTools.LoadSDF(xyz)
#st.write(frame)

#function to findout maximun and min value  in dataframe for specific col. to set the limit of x axis in graph 
def max_min(dfm,dfm2):
	dfm11=dfm.max()
	dfm22=dfm2.max()
	#st.write(dfm11,dfm22)
	return (max(dfm11, dfm22))




# Header
st.header("Analysis of RING")
# plot settings for dual inhibitors


fig2, axes2 = plt.subplots(2,4, figsize=(20,8))
plt.rcParams["font.family"] = "Times New Roman"
#plot using seaborn library

#AromaticRings
axes2[0][0].set_title('AromaticRings', fontsize=20)
axes2[0][0].set_xlabel('AromaticRings/molecule', fontsize = 16.0)
axes2[0][0].set_ylabel('Count', fontsize = 16.0)
axes2[0][0].tick_params(axis='both', which='major', labelsize=16)
ar=max_min(df_target1["AR"], df_target2["AR"])
axes2[0][0].set_xlim(0, ar)

sns.histplot( data=df_target1, x="AR", ax=axes2[0][0], color="darkblue", binwidth=1,alpha=0.5,discrete=True)
sns.histplot(x="AR", data=df_target2, ax=axes2[0][0],color="fuchsia", binwidth=2,alpha=0.8)

#AromaticHeterocycles
axes2[0][1].set_title('AromaticHeterocycles', fontsize=20)
axes2[0][1].set_xlabel('AromaticHeterocycles/molecule', fontsize = 16.0)
axes2[0][1].set_ylabel('Count', fontsize = 16.0)
axes2[0][1].tick_params(axis='both', which='major', labelsize=16)
ahr=max_min(df_target1["AHR"], df_target2["AHR"])
axes2[0][1].set_xlim(0, ahr)

sns.histplot(x="AHR", data=df_target1, ax=axes2[0][1], color="darkblue", binwidth=1)
sns.histplot(x="AHR", data=df_target2, ax=axes2[0][1],color="fuchsia", binwidth=1)

#AliphaticCarbocycles
axes2[0][2].set_title('AliphaticCarbocycles', fontsize=20)
axes2[0][2].set_xlabel('AliphaticCarbocycles/molecule', fontsize = 16.0)
axes2[0][2].set_ylabel('Count', fontsize = 16.0)
axes2[0][2].tick_params(axis='both', which='major', labelsize=16)

alc=max_min(df_target1["ALC"], df_target2["ALC"])
axes2[0][2].set_xlim(0, alc)

sns.histplot(x="ALC", data=df_target1, ax=axes2[0][2], color="darkblue",binwidth=1)
sns.histplot(x="ALC", data=df_target2, ax=axes2[0][2], color="fuchsia",binwidth=1)

#AliphaticHeterocycles
axes2[0][3].set_title('AliphaticHeterocycles', fontsize=20)
axes2[0][3].set_xlabel('AliphaticHeterocycles/molecule', fontsize = 16.0)
axes2[0][3].set_ylabel('Count', fontsize = 16.0)
axes2[0][3].tick_params(axis='both', which='major', labelsize=16)

alhr=max_min(df_target1["ALHR"], df_target2["ALHR"])
axes2[0][3].set_xlim(0, alhr)

sns.histplot(x="ALHR", data=df_target1, ax=axes2[0][3], color="darkblue",binwidth=1)
sns.histplot(x="ALHR", data=df_target2, ax=axes2[0][3], color="fuchsia",binwidth=1 )


#AliphaticHeterocycles
axes2[1][0].set_title('AromaticCarbocycles', fontsize=20)
axes2[1][0].set_xlabel('AromaticCarbocycles/molecule', fontsize = 16.0)
axes2[1][0].set_ylabel('Count', fontsize = 16.0)
axes2[1][0].tick_params(axis='both', which='major', labelsize=16)

alhr=max_min(df_target1["ARC"], df_target2["ARC"])
axes2[1][0].set_xlim(0, alhr)

sns.histplot(x="ARC", data=df_target1, ax=axes2[0][3], color="darkblue",binwidth=1)
sns.histplot(x="ARC", data=df_target2, ax=axes2[0][3], color="fuchsia",binwidth=1 )

plt.tight_layout()
st.pyplot(fig2)

#next section deal with bond information
st.header("Analysis of Bonds")

fig3, axes3 = plt.subplots(2,3, figsize=(20,8))
plt.rcParams["font.family"] = "Times New Roman"
#plot using seaborn library

#NumHBA
axes3[0][0].set_title('NumHBA', fontsize=20)
axes3[0][0].set_xlabel('NumHBA/molecule', fontsize = 16.0)
axes3[0][0].set_ylabel('Count', fontsize = 16.0)
axes3[0][0].tick_params(axis='both', which='major', labelsize=16)

hba=max_min(df_target1["HBA"], df_target2["HBA"])
axes3[0][0].set_xlim(0, hba)

sns.histplot(x="HBA", data=df_target1, ax=axes3[0][0], color="darkblue", binwidth=1)
sns.histplot(x="HBA", data=df_target2, ax=axes3[0][0],color="fuchsia", binwidth=1)

#NumHBD
axes3[0][1].set_title('NumHBD', fontsize=20)
axes3[0][1].set_xlabel('NumHBD/molecule', fontsize = 16.0)
axes3[0][1].set_ylabel('Count', fontsize = 16.0)
axes3[0][1].tick_params(axis='both', which='major', labelsize=16)

hbd=max_min(df_target1["HBD"], df_target2["HBD"])
axes3[0][1].set_xlim(0, hbd)

sns.histplot(x="HBD", data=df_target1, ax=axes3[0][1], color="darkblue", binwidth=1)
sns.histplot(x="HBD", data=df_target2, ax=axes3[0][1],color="fuchsia", binwidth=1)

#NumRotatableBonds
axes3[0][2].set_title('NumRotatableBonds', fontsize=20)
axes3[0][2].set_xlabel('NumRotatableBonds/molecule', fontsize = 16.0)
axes3[0][2].set_ylabel('Count', fontsize = 16.0)
axes3[0][2].tick_params(axis='both', which='major', labelsize=16)

rb=max_min(df_target1["RB"], df_target2["RB"])
axes3[0][2].set_xlim(0, rb)

sns.histplot(x="RB", data=df_target1, ax=axes3[0][2], color="darkblue", binwidth=1)
sns.histplot(x="RB", data=df_target2, ax=axes3[0][2], color="fuchsia", binwidth=1)

#

plt.tight_layout()
st.pyplot(fig3)


# Subheader
st.subheader("These plots will show you the distribution of various atom realted properties of both the inhibitors")
st.write("It will tell you the overlap region between two inhibitors and give idea of atom requirment for the the inhibitor design")
#for 

with st.spinner('we are calculating fragmet in each molecule...'):


	st.header("Fragment Analysis")






	fig7, axes7= plt.subplots(22,4, figsize=(16,60),constrained_layout=True, sharey=True)
	for col, ax7 in zip(df_target1.columns[14:118], axes7.flat):
		
		var1=col+'/molecule'
		sns.histplot(x=col, data=df_target1 ,ax=ax7, color="aqua",alpha=0.3, legend=True)
		sns.histplot(x=col, data=df_target2 ,ax=ax7, color="gold",alpha=0.2,legend=True)
		ax7.set_title(col, fontsize=16)
		ax7.set_xlabel(var1, fontsize = 16.0)
		ax7.tick_params(axis='both', which='major', labelsize=16)
	st.pyplot(fig7)
	#################################################333
	#test for fingerprint analysis
	fps=[FingerprintMols.FingerprintMol(mol) for mol in target1]
	st.write(DataStructs.FingerprintSimilarity(fps[0],fps[1]))

	size=len(target1)
	hmap=np.empty(shape=(size,size))
	table=pd.DataFrame()
	for index, i in enumerate(fps):
		for jndex, j in enumerate(fps):
			similarity=DataStructs.FingerprintSimilarity(i,j)
			hmap[index,jndex]=similarity
			table.loc[target1[index].GetProp('_Name'),target1[jndex].GetProp('_Name')]=similarity

	#st.write(table.head())

st.success('Done!')
#HCL clustering




####################################################




#advance analysis
if st.sidebar.button("Advance Analysis"):
	st.text("In this section you are get to about molecular shape and PCA")
    	# Rest of the logic
	st.header("Normalized Principal Moment of Inertia ratios (NPR) plot to describe molecules shapes")
	fig8, axes8 = plt.subplots(1,1, figsize=(6,4))
	plt.rcParams["font.family"] = "Times New Roman"
	plt.rcParams['axes.linewidth'] = 1.0

	sns.scatterplot(x='NPR1',y='NPR2',data=df_target1,s=10,linewidth=0.5,alpha=1,color="r",ax=axes8,label="target1")
	sns.scatterplot(x='NPR1',y='NPR2',data=df_target2,s=10,linewidth=0.5,alpha=1,color="g",ax=axes8,label="target2")
	#plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
	plt.legend()
	x1, y1 = [0.5, 0], [0.5, 1]
	x2, y2 = [0.5, 1], [0.5, 1]
	x3, y3 = [0,1],[1,1]

	plt.plot(x1, y1,x2,y2,x3,y3,c='gray',ls='--',lw=2)

	plt.xlabel ('NPR1',fontsize=14,fontweight='bold')

	plt.ylabel ('NPR2',fontsize=14,fontweight='bold')

	axes8.spines['top'].set_visible(False)
	axes8.spines['right'].set_visible(False)

	plt.text(0, 1.01,s='Rod',fontsize=14,horizontalalignment='center',verticalalignment='center',fontweight='bold')
	plt.text(1, 1.01,s='Sphere',fontsize=14,horizontalalignment='center',verticalalignment='center',fontweight='bold')
	plt.text(0.5, 0.49,s='Disc',fontsize=14,horizontalalignment='center',verticalalignment='center',fontweight='bold')

	plt.tick_params ('both',width=2,labelsize=14)
	plt.tight_layout()
	st.pyplot(fig8)


###############################
###############################
	with st.spinner('similarity Heat Map  was loading please wait...'):
    

		linked = linkage(hmap,'single')
		labelList = [mol.GetProp('_Name') for mol in target1]


		o=dendrogram(linked,  
			    orientation='left',
			    labels=labelList,
			    distance_sort='descending',
			    show_leaf_counts=True)

		# This will give us the clusters in order as the last plot
		new_data=list(reversed(o['ivl']))

		
				

		figure100= plt.figure(figsize=(20,20))
		gs1 = gridspec.GridSpec(2,7)
		gs1.update(wspace=0.01)
		#gs1[0:-1, :2]
		#gs1[0:-1,6:7]
		ax2 = plt.subplot(gs1[0:-1, :5])
		f=ax2.imshow (hmap, cmap='PRGn_r', interpolation='nearest')

		ax2.set_title('Fingerprint Similarity',fontsize=20,weight='bold')
		ax2.set_xticks (range(len(new_data)))
		ax2.set_yticks (range(len(new_data)))
		ax2.set_xticklabels (new_data,rotation=90,size=2)
		ax2.set_yticklabels (new_data,size=2)

		ax3 = plt.subplot(gs1[0:-1,5:6])
		m=plt.colorbar(f,cax=ax3,shrink=1.0,orientation='vertical',spacing='uniform', pad=0.01)
		m.set_label ('Fingerprint Similarity',size=20)
		ax3.tick_params(labelsize=20)

		plt.tick_params ('both',width=2)
		plt.tight_layout()
		st.pyplot(figure100)

	st.success('Done!')
################################
#################################

#-------------------------------------------------------------------------------------------
#Principal Component Analysis (PCA)
#The molecules are described by their phisicpochemycal terms
	st.header("Principal Component Analysis (PCA)")
	st.text("PCA")
	descriptors = df_target1[['mw','mlogp','tpsa','AR','HBA','HBD', 'RB']].values #The non-redundant molecular descriptors chosen for PCA
	descriptors_std = StandardScaler().fit_transform(descriptors) #Important to avoid scaling problems between our different descriptors
	pca = PCA()
	descriptors_2d = pca.fit_transform(descriptors_std)
	descriptors_pca= pd.DataFrame(descriptors_2d) # Saving PCA values to a new table
	descriptors_pca.index = df_target1.index
	descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]
	st.write(descriptors_pca.head(5)) #Displays the PCA table
	## This normalization will be performed just for PC1 and PC2, but can be done for all the components.
	#The normalization is to plot PCA values in 0-1 sacle and include the vectors (features to the plot)

	scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1'])) 
	scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))

	# And we add the new values to our PCA table
	descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
	descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]

	fig9, axes9 = plt.subplots(1,1, figsize=(6,6))
	plt.rcParams["font.family"] = "Times New Roman"
	plt.rcParams['axes.linewidth'] = 1.0


	sns.scatterplot(x='PC1_normalized',y='PC2_normalized',data=descriptors_pca,s=20,palette=sns.color_palette("Set2", 3),linewidth=0.2,alpha=1,ax=axes9)

	plt.xlabel ('PC1',fontsize=14,fontweight='bold')
	axes9.xaxis.set_label_coords(0.98, 0.45)
	plt.ylabel ('PC2',fontsize=14,fontweight='bold')
	axes9.yaxis.set_label_coords(0.45, 0.98)

	axes9.spines['left'].set_position(('data', 0))
	axes9.spines['bottom'].set_position(('data', 0))
	axes9.spines['top'].set_visible(False)
	axes9.spines['right'].set_visible(False)


	lab=['mw','mlogp','tpsa','AR','HBA','HBD', 'RB'] #Feature labels

	l=np.transpose(pca.components_[0:2, :]) ## We will get the components eigenvectors (main features) for PC1 and PC2

	n = l.shape[0]
	for i in range(n):
	    plt.arrow(0, 0, l[i,0], l[i,1],color= 'k',alpha=0.5,linewidth=1.8,head_width=0.025)
	    plt.text(l[i,0]*1.25, l[i,1]*1.25, lab[i], color = 'k',va = 'center', ha = 'center',fontsize=16)

	circle = plt.Circle((0,0), 1, color='gray', fill=False,clip_on=True,linewidth=1.5,linestyle='--')

	plt.tick_params ('both',width=2,labelsize=18)

	axes9.add_artist(circle)
	plt.xlim(-1.2,1.2)
	plt.ylim(-1.2,1.2)
	plt.tight_layout()
	st.pyplot(fig9)
	
	st.header("Radar chart of Beyond Lipinski's Rule of Five")
	Ro5=pd.DataFrame() # I'll create a new table containing the normalized bRo5 values of our compounds

	Ro5['MolWt']=[i/500 for i in df_target1['mw']]
	#Ro5['MLogP']=[i/5 for i in df_target1['mlogp']]
	#Ro5['HBA']=[i/10 for i in table['NumHAcceptors']]
	#Ro5['HBD']=[i/5 for i in table['NumHDonors']]
	#Ro5['RotB']=[i/10 for i in table['NumRotatableBonds']]
	#Ro5['TPSA']=[i/140 for i in table['TPSA']]
		
	categories=list(Ro5.columns)  # This will set up the parameters for the angles of the radar plot. 
	N = len(categories)
	values=Ro5[categories].values[0]
	values=np.append(values,values[:1])
	angles = [n / float(N) * 2 * pi for n in range(N)]
	angles += angles[:1]

	Ro5_up=[1,1,1,1,1,1,1] #The upper limit for bRo5
	Ro5_low=[0.5,0.1,0,0.25,0.1,0.5,0.5]  #The lower limit for bRo5

	fig=plt.figure(figsize=(6,6))

	ax = fig.add_axes([1, 1, 1, 1],projection='polar')

	plt.xticks(angles[:-1], categories,color='k',size=20,ha='center',va='top',fontweight='book')

	plt.tick_params(axis='y',width=4,labelsize=12, grid_alpha=0.05)

	ax.set_rlabel_position(0)

	ax.plot(angles, Ro5_up, linewidth=2, linestyle='-',color='red')
	ax.plot(angles, Ro5_low, linewidth=2, linestyle='-',color='red')

	#ax.fill(angles, Ro5_up, 'red', alpha=0.2)
	ax.fill(angles, Ro5_low, 'orangered', alpha=0.2)

	for i in data.index[:500]: #I'll just show the profile for the first 250 molecules in the table for clarity of the plot
	    values=data[categories].values[i]
	    values=np.append(values,values[:1])
	    ax.plot(angles, values, linewidth=0.7 ,color='steelblue',alpha=0.5)
	    #ax.fill(angles, values, 'C2', alpha=0.025)

	ax.grid(axis='y',linewidth=1.5,linestyle='dotted',alpha=0.8)
	ax.grid(axis='x',linewidth=2,linestyle='-',alpha=1)

	plt.show()



st.write("check out this [link](https://www.rdkit.org/)")

"""

st.header("Interactive table")

st.write(df_target1)
st.write(df_target2)


"""



