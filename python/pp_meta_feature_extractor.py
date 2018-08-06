# -*- coding: utf-8 -*-
import pandas as pd
from IPython import embed
from tqdm import tqdm
import math
import numpy as np
from nltk.corpus import wordnet
from nltk.wsd import lesk
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import scipy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class CentroidTransformer(BaseEstimator, TransformerMixin):
    """Meta-features creation using the similarity of the documents with
    the centroids of the classes
    Attributes
    ----------
    centroids_: array-like, shape = [n_classes, n_features]
        Centroid of each class
    """

    def fit(self, X, y):
        """
        Fit the centroids according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array, shape = [n_samples]
            Target values (integers)
        """
        if y.ndim < 2:
            self.centroids_ = NearestCentroid().fit(X, y).centroids_
            return self

        n_features = X.shape[1]
        n_samples, n_classes = y.shape

        self.centroids_ = np.zeros((n_classes, n_features))
        for j in range(n_classes):
            idx = y[:, j] == 1
            if idx.sum() != 0:
                self.centroids_[j, :] = X[idx].mean(0)

        self.centroids_ = scipy.sparse.csr_matrix(self.centroids_)
        return self

    def transform(self, X):
        """
        Transforms data in to the similarity to the centroids.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Test vector, where n_samples in the number of samples and
            n_features is the number of features.

        Returns
        -------
        {array-like, sparse matrix} Transformed data
        """
        cos = cosine_similarity(X, self.centroids_)
        # euc = 1. / (1. + euclidean_distances(X, self.centroids_))

        if scipy.sparse.issparse(X):
            return scipy.sparse.hstack((X, cos))

        return np.hstack((X, cos))

NUM_DOC = 0
NUM_TERMS = 0
term_doc_frequencies = {}
terms_idfs = {}
collection_term_frequencies = {}
min_val_avoid_zero_div = 0.000000001

def add_token_count(df):
	tqdm.pandas(desc="Calculating tokenCount")
	df["tokenCount"] = df.progress_apply(lambda r: len(r["terms"]), axis=1)

def add_term_count(df):
	tqdm.pandas(desc="Calculating termCount")
	df["termCount"] = df.progress_apply(lambda r: len(set(r["terms"])), axis=1)

def add_avg_ql(df):
	tqdm.pandas(desc="Calculating AvQL")
	df["AvQL"] = df.progress_apply(lambda r: sum([len(q) for q in (r["terms"])])\
		/float(len(r["terms"])), axis=1)

def add_avg_idf(df):
	def calculate_avg_idf(r, terms_idfs):
		idfs = [terms_idfs[term] for term in r["terms"]]
		return sum(idfs)/float(len(idfs))
	tqdm.pandas(desc="Calculating AvIDF")
	df["AvIDF"] = df.progress_apply(lambda r, f=calculate_avg_idf, s=terms_idfs: f(r, s), axis=1)	

def add_max_idf(df):
	def calculate_max_idf(r, terms_idfs):
		idfs = [terms_idfs[term] for term in r["terms"]]
		return max(idfs)
	tqdm.pandas(desc="Calculating MaxIDF")
	df["MaxID"] = df.progress_apply(lambda r, f=calculate_max_idf, s=terms_idfs: f(r, s), axis=1)	

def add_dev_idf(df):
	def calculate_dev_idf(r, terms_idfs):
		idfs = [terms_idfs[term] for term in r["terms"]]
		return np.std(idfs)
	tqdm.pandas(desc="Calculating DevIDF")
	df["DevID"] = df.progress_apply(lambda r, f=calculate_dev_idf, s=terms_idfs: f(r, s), axis=1)	

def add_avg_ictf(df):
	def calculate_avg_ictf(r):
		ictfs = []
		for term_freq in r["term_freq"].values():
			ictfs.append(np.log2(NUM_TERMS) - np.log2(term_freq))
		return sum(ictfs)/float(len(ictfs))	
	tqdm.pandas(desc="Calculating AvICTF")
	df["AvICTF"] = df.progress_apply(lambda r, f=calculate_avg_ictf: f(r), axis=1)

def add_scs(df):
	""" this function requires that add_avg_ictf is run before on the df"""
	tqdm.pandas(desc="Calculating SCS")
	df["SCS"] = df.progress_apply(lambda r: r["AvICTF"] + \
		np.log2(1/float(len(r["terms"]))), axis=1)

def calculate_scq(r):
	scqs = []
	for term in r["terms"]:
		df = term_doc_frequencies[term]
		cf = collection_term_frequencies[term]
		scq = (1 + np.log(cf)) *\
			np.log(1+(NUM_DOC/float(df+min_val_avoid_zero_div)))
		scqs.append(scq)
	return scqs

def add_avg_scq(df):
	tqdm.pandas(desc="Calculating AvSCQ")
	df["AvSCQ"] = df.progress_apply(lambda r, f=calculate_scq,\
		: np.mean(f(r)), axis=1)	

def add_sum_scq(df):
	tqdm.pandas(desc="Calculating SumSCQ")
	df["SumSCQ"] = df.progress_apply(lambda r, f=calculate_scq,\
		: sum(f(r)), axis=1)

def add_max_scq(df):
	tqdm.pandas(desc="Calculating MaxSCQ")
	df["MaxSCQ"] = df.progress_apply(lambda r, f=calculate_scq,\
		: max(f(r)), axis=1)

def add_av_path(df):
	tqdm.pandas(desc="Calculating AvPath")
	def calculate_av_path(r):
		terms = r["terms"]
		path_sims = []
		for term1 in terms:
			for term2 in terms:
				if(term1 != term2):
					w1 = wordnet.synsets(term1)
					w2 = wordnet.synsets(term2)
					if(len(w1)>0 and len(w2)>0):
						sim = wordnet.path_similarity(w1[0], w2[0])
						if(sim is not None):
							path_sims.append(sim)
		return np.mean(path_sims) if len(path_sims)!=0 else 0
	df["AvPath"] = df.progress_apply(lambda r, f=calculate_av_path: f(r), axis=1)

def add_av_lch(df):
	tqdm.pandas(desc="Calculating AvLCH")
	def calculate_av_lesk(r):
		terms = r["terms"]
		path_sims = []
		for term1 in terms:
			for term2 in terms:
				if(term1 != term2):
					w1 = wordnet.synsets(term1)
					w2 = wordnet.synsets(term2)
					if(len(w1)>0 and len(w2)>0 and w1[0].pos() == w2[0].pos()):
						sim = wordnet.lch_similarity(w1[0], w2[0])
						if(sim is not None):
							path_sims.append(sim)
		return np.mean(path_sims) if len(path_sims)!=0 else 0
	df["AvLCH"] = df.progress_apply(lambda r, f=calculate_av_lesk: f(r), axis=1)

def add_av_wup(df):
	tqdm.pandas(desc="Calculating AvWUP")
	def calculate_av_lesk(r):
		terms = r["terms"]
		path_sims = []
		for term1 in terms:
			for term2 in terms:
				if(term1 != term2):
					w1 = wordnet.synsets(term1)
					w2 = wordnet.synsets(term2)
					if(len(w1)>0 and len(w2)>0):
						sim = wordnet.wup_similarity(w1[0], w2[0])
						if(sim is not None):
							path_sims.append(sim)
		return np.mean(path_sims) if len(path_sims)!=0 else 0
	df["AvWUP"] = df.progress_apply(lambda r, f=calculate_av_lesk: f(r), axis=1)

def add_av_p(df):
	tqdm.pandas(desc="Calculating AvP")
	df["AvP"] = df.progress_apply(lambda r: np.mean([len(wordnet.synsets(term)) for term in r["terms"]]), axis=1)

def add_av_np(df):
	tqdm.pandas(desc="Calculating AvNP")
	df["AvNP"] = df.progress_apply(lambda r: np.mean([len([syn for syn in wordnet.synsets(term) if syn.pos()=="n"])\
		for term in r["terms"]]), axis=1)

def extract_doc_pp_features(docs):
	docs["terms"] = docs.apply(lambda r: r["doc_text"].split(" "),axis=1)

	#Calculates terms idf
	global term_doc_frequencies
	global terms_idfs
	global NUM_DOC
	NUM_DOC = docs.shape[0]
	for idx, row in tqdm(docs.iterrows(), "Calculatig term doc frequencies"):
		for term in set(row["doc_text"].split(" ")):
			if term not in term_doc_frequencies:
				term_doc_frequencies[term] = 0
			term_doc_frequencies[term] = term_doc_frequencies[term] +1	
	for term in term_doc_frequencies.keys():
		terms_idfs[term] = np.log2(NUM_DOC/(float(term_doc_frequencies[term]+min_val_avoid_zero_div)))	

	#Calculates terms freq
	global NUM_TERMS
	NUM_TERMS = len(term_doc_frequencies.keys())
	tqdm.pandas(desc="Calculating term frequencies")
	def add_tf(r):
		global collection_term_frequencies
		tf = {}
		for term in set(r["terms"]):
			tf[term] = r["doc_text"].count(term)
			if term not in collection_term_frequencies:
				collection_term_frequencies[term] = 0
			collection_term_frequencies[term] = collection_term_frequencies[term] + tf[term]
		return tf
	docs["term_freq"] = docs.progress_apply(lambda r, f=add_tf: f(r),axis=1)

	#Specificity
	add_token_count(docs)
	add_term_count(docs)
	add_avg_ql(docs)
	add_avg_idf(docs)
	add_max_idf(docs)
	add_dev_idf(docs)
	add_avg_ictf(docs)
	add_scs(docs)
	add_avg_scq(docs)
	add_sum_scq(docs)
	add_max_scq(docs)

	#term relatedness
	add_av_wup(docs)
	add_av_lch(docs)
	add_av_path(docs)

	# ambiguity
	add_av_p(docs)
	add_av_np(docs)

	return docs

# def add_centroid_features(df, label_dataset):	
# 	X, y = load_svmlight_file(label_dataset)
#     ct = CentroidTransformer()
#     ct.fit(X, y) 
#     df = pd.DataFrame(ct.transform(X).todense()[:,-len(set(y)):])
#     df.columns = ["dist_to_"+str(i) for i in df.columns]
#     return df

def main():
	docs = [
		["id1","la la le lo"],
		["id2","la meu amigo que bacana le le le"],
		["id3", "unico"]
	]
	datasets_folder = "/home/guz/ssd/msc-gustavo-penha/experiment5/stacking-bagged-boosted-forests/release/datasets/sentiment_analysis/"
	for dataset_name, label_dataset  in [
		("raw/vader_amazon_sem_score.txt", "tf/amazon.svm"),
		("raw/sentistrength_bbc_sem_score.txt", "tf/bbc.svm"),\
		("raw/debate_sem_score.txt", "tf/debate.svm"),\
		("raw/sentistrength_digg_sem_score.txt", "tf/digg.svm"),\
		("raw/sentistrength_myspace_sem_score.txt", "tf/myspace.svm"),\
		("raw/vader_nyt_sem_score.txt", "tf/nyt.svm"),\
		("raw/vader_twitter_sem_score.txt", "tf/tweets.svm"),\
		("raw/yelp_reviews_sem_score.txt", "tf/yelp.svm"),\
		("raw/sentistrength_youtube_sem_score.txt", "tf/youtube.svm")\
		]:
		
		dataset = datasets_folder+dataset_name
		# docs_df = pd.DataFrame(docs, columns = ["doc_id", "doc_text"])
		# docs_df = pd.read_csv(dataset, sep="\n", names=["doc_text"]).reset_index()
		lines = [[line.decode('utf-8').rstrip('\n')] for line in open(dataset)]
		docs_df = pd.DataFrame(lines, columns = ["doc_text"]).reset_index()
		docs_df.columns = ["doc_id", "doc_text"]
		pp_features = extract_doc_pp_features(docs_df)
		print(pp_features)
		# pp_features = add_centroid_features(pp_features, datasets_folder+ label_dataset)
		pp_features["dummy_label"] = 1
		dump_svmlight_file(pp_features.drop(columns=["doc_id","doc_text","terms", "dummy_label", "term_freq"]).as_matrix(),\
			pp_features["dummy_label"].as_matrix(),datasets_folder+label_dataset.replace("tf/","").replace(".svm","")+"_pp_meta_features.svm")
if __name__ == '__main__':
	main()
