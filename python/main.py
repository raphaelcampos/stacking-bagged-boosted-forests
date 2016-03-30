
from app import TextClassificationApp

app = TextClassificationApp()

app.run(app.parse_arguments())




"""
if args.method == 'lazy':
	estimator = instantiator.get_instance(args.method)
	tuned_parameters = [{'n_neighbors': [30,100,500],
							 'n_estimators': [50, 100, 200, 400]}]
elif args.method == 'lazy_xt':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_neighbors': [30,100,500], 'n_estimators': [50, 100, 200, 400]}]
elif args.method == 'lazy_broof':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_neighbors': [30,100,500], 'n_estimators': [50, 100, 200, 400]}]
elif args.method == 'adarf':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_estimators': [50, 100, 200, 400, 600]}]
elif args.method == 'broof':
	estimator = instantiator.get_instance(args.method)
	tuned_parameters = [{'n_trees': [5], 'n_estimators': [10, 20, 50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]},{'n_trees': [10, 30, 50], 'n_estimators': [10, 20, 30], 'learning_rate': [0.1, 0.5, 1.0]}]
elif args.method == 'bert':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_trees': [5], 'n_estimators': [10, 20, 50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]},{'n_trees': [10, 30, 50], 'n_estimators': [10, 20, 30], 'learning_rate': [0.1, 0.5, 1.0]}]
elif args.method == 'xt':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_estimators': [200], 'criterion':['gini', 'entropy'], 'max_features': ['sqrt', 'log2', 0.03, 0.08, 0.15, 0.3]}]
elif args.method == 'mlr':
	estimator = instantiate_estimator(args.method, args)
elif args.method == 'comb1':
	estimators_stack = list()
	args.c = 1
	estimators_stack.append(
		[#instantiate_estimator("svm", args),
		Broof(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=8,
		 learning_rate=args.learning_rate, max_features=args.max_features,
		 random_state=123),
		 LazyNNRF(n_neighbors=args.kneighbors, n_estimators=args.trees, n_jobs=args.jobs,
		  max_features='auto', criterion='gini',
		  random_state=123, n_gpus=args.gpus)])
	estimators_stack.append(ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features='sqrt', verbose=0))
	#estimators_stack.append(ExtraTreesClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features='auto', verbose=0))
	#estimators_stack.append(RidgeClassifierCV(cv=5, fit_intercept=False))
	#estimators_stack.append(MLR())
	#estimators_stack.append(LogisticRegression(solver="newton-cg", tol=1e-3))
	estimator = StackingClassifier(estimators_stack, n_folds=10, random_state=1234)
elif args.method == 'comb2':
	estimators_stack = list()
	estimators_stack.append(
		[Bert(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=5,
		 learning_rate=args.learning_rate, max_features=args.max_features),
		 LazyNNExtraTrees(n_neighbors=args.kneighbors, n_estimators=args.trees, n_jobs=args.jobs,
		  max_features='auto', criterion='gini', n_gpus=args.gpus)])
	estimators_stack.append(ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features=args.max_features, verbose=0))
	#estimators_stack.append(RidgeClassifierCV(cv=5))
	estimator = StackingClassifier(estimators_stack)
elif args.method == 'comb3':
	estimators_stack = list()
	# Level 0 classifiers
	estimators_stack.append(
		[Broof(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=5,
			 learning_rate=args.learning_rate, max_features=args.max_features),
		 LazyNNRF(n_neighbors=args.kneighbors, n_estimators=args.trees,
		 							 n_jobs=args.jobs, max_features='auto', 
		 							 criterion='gini', n_gpus=args.gpus),
		 Bert(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=5,
		 								 learning_rate=args.learning_rate,
		 								 max_features=args.max_features),
		 LazyNNExtraTrees(n_neighbors=args.kneighbors, n_estimators=args.trees,
		 							 n_jobs=args.jobs, max_features='auto',
		 							 criterion='gini', n_gpus=args.gpus)])
	# Level 1 classifier (Aggregator)
	estimators_stack.append(ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features=args.max_features, verbose=0))
	#estimators_stack.append(RidgeClassifierCV(cv=5))
	estimator = StackingClassifier(estimators_stack)
elif args.method == 'comb4':
	estimators_stack = list()
	level0 = []

	args_copy = copy(args)

	args.c = 1
	level0.append(instantiate_estimator("svm", args))
	args.max_features = 'auto'
	level0.append(instantiate_estimator("lazy", args))
	level0.append(instantiate_estimator("lxt", args))
	args.kneighbors = 10
	level0.append(instantiate_estimator("knn", args))
	args = args_copy
	args.trees = 8
	level0.append(instantiate_estimator("broof", args))
	level0.append(instantiate_estimator("bert", args))


	# Level 0 classifiers
	estimators_stack.append(level0)
	# Level 1 classifier (Aggregator)
	estimators_stack.append(ForestClassifier(n_estimators=200, n_jobs=args.jobs,
			 criterion='gini', max_features='sqrt', verbose=0))
	#estimators_stack.append(RidgeClassifierCV(cv=5))
	estimator = StackingClassifier(estimators_stack, n_folds=10,
						 random_state=1234)
else:
	estimator = ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features=args.max_features, verbose=10)
	tuned_parameters = [{'n_estimators': [200], 'criterion':['gini', 'entropy'], 'max_features': ['sqrt', 'log2', 0.03, 0.08, 0.15, 0.3]}]
""" 