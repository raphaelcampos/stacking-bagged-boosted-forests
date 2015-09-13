/*********************************************************************
11	
12	 Copyright (C) 2015 by Wisllay Vitrio
13	
14	 This program is free software; you can redistribute it and/or modify
15	 it under the terms of the GNU General Public License as published by
16	 the Free Software Foundation; either version 2 of the License, or
17	 (at your option) any later version.
18	
19	 This program is distributed in the hope that it will be useful,
20	 but WITHOUT ANY WARRANTY; without even the implied warranty of
21	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
22	 GNU General Public License for more details.
23	
24	 You should have received a copy of the GNU General Public License
25	 along with this program; if not, write to the Free Software
26	 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
27	
28	 ********************************************************************/

#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include <tclap/CmdLine.h>

#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"
#include  <cuda.h>

#include "Dataset.h"
#include "cuLazyNN_RF.h"
#include "cuNearestNeighbors.h"

#include <map>
using namespace std;

class CustomHelpVisitor : public TCLAP::HelpVisitor
{
        protected:
                TCLAP::ValueArg<std::string>* _modelArg;
        public:
                CustomHelpVisitor(TCLAP::CmdLineInterface *cmd, TCLAP::CmdLineOutput **out, TCLAP::ValueArg<std::string> *modelArg) : TCLAP::HelpVisitor(cmd, out), _modelArg(modelArg) {} ;
                void visit() { 
                    if(!_modelArg->isSet())
                        TCLAP::HelpVisitor::visit(); 
                    };
};

struct FileStats {
    int num_docs;
    int num_terms;

    std::map<int, int> doc_to_class;

    FileStats() : num_docs(0), num_terms(0) {}
};



FileStats readTrainingFile(std::string &file, std::vector<Entry> &entries);
void readTestFile(InvertedIndex &index, FileStats &stats, std::string &file, int K, std::string distance, ofstream &fileout, ofstream &filedists);
void updateStatsMaxFeatureTest(std::string &filename, FileStats &stats);

bool makeQuery(InvertedIndex &inverted_index, FileStats &stats, std::string &line, int K,
               void (*distance)(InvertedIndex, Entry*, int*, cuSimilarity*, int D), ofstream &fileout, ofstream &filedists);

void write_output(ofstream &fileout, int trueclass, int guessedclass, int docid);

int get_class(std::string token);


void teste_lazynn(std::string trainingFileName, std::string testFileName, std::string resultsFileName, int k, int trial, bool append = true){

    Dataset training_set, test_set;
    int correct_cosine = 0, wrong_cosine = 0;
    
    training_set.loadGtKnnFormat(trainingFileName.c_str());

    cuLazyNN_RF cLazy(training_set);
    test_set.loadGtKnnFormat(testFileName.c_str());
    double start, end, total = 0;

    ofstream file;

    if(append) 
        file.open(resultsFileName.data(), std::ios_base::app);
    else 
        file.open(resultsFileName.data());


    file << "#" << trial << endl;
    #pragma omp parallel for num_threads(8) schedule(dynamic)
    for (int i = 0; i < test_set.getSamples().size(); ++i)
    {
        start = gettime();
        int guessed_class = cLazy.classify(test_set.getSamples()[i].features, k);
        end = gettime();
        
        total += end - start;
	
        if(guessed_class == test_set.getSamples()[i].y) {
            correct_cosine++;   
        } else {
            wrong_cosine++;
        }

        std::cerr.precision(4);
        std::cerr.setf(std::ios::fixed);
        std::cerr << "\r" << double(i+1)/test_set.getSamples().size() * 100 << "%" << " - " << double(correct_cosine) / (i+1);

        file << i << " CLASS=" <<  test_set.getSamples()[i].y << " CLASS=" << guessed_class << ":1" << endl;
    }

    printf("Total time taken to classify all queries: %lf seconds\n", total);

    printf("Cosine similarity\n");
    printf("Correct: %d Wrong: %d\n", correct_cosine, wrong_cosine);
    printf("Accuracy: %lf%%\n\n", double(correct_cosine) / double(test_set.size()));


    file.close();
}

template <class InputIterator1>
  int size (InputIterator1 first1, InputIterator1 last1)
{
    int counter = 0;
    for (; first1 != last1; ++first1)
    {
        counter++;
    }
    return counter;
}

template <class InputIterator1, class InputIterator2>
  int count_distinct (InputIterator1 first1, InputIterator1 last1,
                            InputIterator2 first2, InputIterator2 last2)
{
   int counter = 0;
  while (true)
  {
    if (first1==last1) return counter + size(first2,last2);
    if (first2==last2) return counter + size(first1,last1);

    if (first1->first<first2->first) { counter++; ++first1; }
    else if (first2->first<first1->first) { counter++; ++first2; }
    else { counter++; ++first1; ++first2; }
  }
}

void teste_cuNN(std::string trainingFileName, std::string testFileName, std::string resultsFileName, int k, int trial, bool append = true){

    srand(time(NULL));

    Dataset training_set, test_set;
    int tp = 0, wrong_cosine = 0;
    
    training_set.loadGtKnnFormat(trainingFileName.c_str());

    cuNearestNeighbors cuNN(training_set);
    test_set.loadGtKnnFormat(testFileName.c_str());
    double start, end, total = 0;
    
    printf("train (dim : %d, class: %d ) - test (dim: %d, class: %d) - total classes : %d \n", training_set.dimension(), training_set.num_class(), test_set.dimension(), test_set.num_class(), count_distinct(training_set.doc_per_class.begin(),training_set.doc_per_class.end(),test_set.doc_per_class.begin(),test_set.doc_per_class.end()));

    int test_set_size = test_set.getSamples().size();
    int documents_processed = 0;
    std::vector<sample>::iterator end_it = test_set.sample_end();

    int num_class = count_distinct(training_set.doc_per_class.begin(),training_set.doc_per_class.end(),test_set.doc_per_class.begin(),test_set.doc_per_class.end());

    int **confusion_matrix = new int*[num_class];
    
    for (int i = 0; i < num_class; ++i)
    {
        confusion_matrix[i] = new int[num_class];
        for (int j = 0; j < num_class; ++j)
        {
            confusion_matrix[i][j] = 0;
        }
    }

    ofstream file;

    if(append) 
        file.open(resultsFileName.data(), std::ios_base::app);
    else 
        file.open(resultsFileName.data());

    file << "#" << trial << endl;
    for (std::vector<sample>::iterator it = test_set.sample_begin(); it != end_it; ++it)
    {

        start = gettime();
        int guessed_class = cuNN.classify(it->features, k);
        end = gettime();
        
        total += end - start;

        confusion_matrix[it->y][guessed_class]++;

        if(guessed_class == it->y) {
            tp++;   
        } else {
            wrong_cosine++;
        }
        ++documents_processed;
        std::cerr.precision(4);
        std::cerr.setf(std::ios::fixed);
        std::cerr << "\r" << double(documents_processed)/test_set_size * 100 << "%" << " - " << double(tp) / (documents_processed);
    
        file << documents_processed << " CLASS=" <<  it->y << " CLASS=" << guessed_class << ":1" << endl;
    }

    printf("\nTotal time taken to classify all queries: %lf seconds\n", total);

    printf("Cosine similarity\n");
    printf("Correct: %d Wrong: %d\n", tp, wrong_cosine);
    printf("Accuracy: %lf%%\n\n", double(tp) / double(test_set_size));

    int tps = 0, fps = 0, fns;
    double macro_avg_prec = 0, macro_avg_recall = 0;
    for (int i = 0; i < num_class; ++i)
    {
        int tp = confusion_matrix[i][i], fp = 0, fn = 0;
        for (int j = 0; j < num_class; ++j)
        {
            fp += (i != j)? confusion_matrix[i][j] : 0;
            fn += (i != j)? confusion_matrix[j][i] : 0;
            //cout << setw(5) << confusion_matrix[i][j] << " ";
        }

        //cout << endl;

        macro_avg_prec += (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
        macro_avg_recall += (tp + fn) > 0 ?(double)tp / (tp + fn) : 0;

        tps += tp;
        fps += fp;
        fns += fn;
    

    }
    double micro_avg_prec = (double)tps / (test_set_size);
    double micro_avg_recall = (double)tps / (tps + fns);
    double microF1 = 2*micro_avg_recall*micro_avg_prec / (micro_avg_recall+micro_avg_prec);

    
    macro_avg_prec /= test_set.num_class();
    macro_avg_recall /= test_set.num_class();

    double macroF1 = 2*macro_avg_recall*macro_avg_prec / (macro_avg_recall+macro_avg_prec);
    printf("microF1 : %f, macroF1 : %f\n", microF1, macroF1);

    for (int i = 0; i < num_class; ++i)
    {
        delete[] confusion_matrix[i];
    }
    delete[] confusion_matrix;
    file.close();
}


/**
 * Receives as parameters the training file name and the test file name
 */
int main(int argc, char **argv) {
    initCudpp(); //initializes the CUDPP library
    cuInit(0);
    cudaDeviceSynchronize();

    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {  

        // Define the command line object, and insert a message
        // that describes the program. The "Command description message" 
        // is printed last in the help text. The second argument is the 
        // delimiter (usually space) and the last one is the version number. 
        // The CmdLine object parses the argv array based on the Arg objects
        // that it contains. 
        TCLAP::CmdLine cmd("Command description message", ' ', "0.9");

        vector<string> allowed;
        allowed.push_back("knn");
        allowed.push_back("knn_rf");
        TCLAP::ValuesConstraint<string> allowedVals( allowed );

        TCLAP::ValueArg<std::string> modelArg("m", "model", "Classifier model (default : knn).", false, allowed[0], &allowedVals);

        cmd.add( modelArg );

       
        // Define a value argument and add it to the command line.
        // A value arg defines a flag and a type of value that it expects,
        // such as "-n Bishop".
        TCLAP::UnlabeledValueArg<std::string> trainArg("train","Traning dataset location.", true, "", "training set");

        // Add the argument nameArg to the CmdLine object. The CmdLine object
        // uses this Arg to parse the command line.
        cmd.add( trainArg );

        TCLAP::UnlabeledValueArg<std::string> testArg("test", "Test dataset location.", true, "", "test set");

        // Add the argument nameArg to the CmdLine object. The CmdLine object
        // uses this Arg to parse the command line.
        cmd.add( testArg );

        TCLAP::ValueArg<std::string> resultsArg("r", "results", "Results output file (default : results.out).", false, "results.out", "string");

        // Add the argument nameArg to the CmdLine object. The CmdLine object
        // uses this Arg to parse the command line.
        cmd.add( resultsArg );

        TCLAP::ValueArg<int> trialArg("","trial","Trial number.", false, 0, "int");

        cmd.add( trialArg );

        TCLAP::SwitchArg appendSwitch("a","append","Append results to result file.", cmd);

        TCLAP::ValueArg<int> kArg("k","K","K nearest neirghbor to be searched.(default : 30)", false, 30, "int");

        cmd.add( kArg );

        TCLAP::ValueArg<int> numTreesArg("n","number-trees","Maximum number of trees in the ensemble.(default : 100)", false, 100, "int");

        cmd.add( numTreesArg );

        TCLAP::ValueArg<int> heightTreesArg("H","height","Maximum height of trees in the ensemble(default : 0). H=0 means unpruned otherwise prune with H top.", false, 100, "int");

        cmd.add( heightTreesArg );

        // Parse the argv array.
        cmd.parse( argc, argv );

        std::string model = modelArg.getValue();
        if(model == "knn_rf"){
            teste_lazynn(trainArg.getValue(), testArg.getValue(), resultsArg.getValue(), kArg.getValue(), trialArg.getValue(), appendSwitch.getValue());               
        }else{
            teste_cuNN(trainArg.getValue(), testArg.getValue(), resultsArg.getValue(), kArg.getValue(), trialArg.getValue(), appendSwitch.getValue());
        }

    } catch (TCLAP::ArgException &e)  // catch any exceptions
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
   
    // tsalles 
    // tf_idf = (1 + log(tf)) * (idf/MAXIDF)

    /*
    if(argc < 7) {
        std::cerr << "Wrong parameters. Correct usage: <executable> <training_file> <test_file> <k> <cosine | l2 | l1> <output_classifications_file> <output_distances_file>" << std::endl;
        exit(1);
    }

    std::string trainingFileName(argv[1]);
    std::string testFileName(argv[2]);
    int k = atoi(argv[3]);
    std::string distanceFunction(argv[4]);
    std::string outputFileName(argv[5]);
    std::string outputFileDistancesName(argv[6]);

    printf("olá\n");

    std::vector<Entry> entries;

    double start, end;

    printf("Reading file...\n");

    start = gettime();
    FileStats stats = readTrainingFile(trainingFileName, entries);
    end = gettime();
    updateStatsMaxFeatureTest(testFileName, stats);

    printf("time taken: %lf seconds\n", end - start);

    start = gettime();
    InvertedIndex inverted_index = make_inverted_index(stats.num_docs, stats.num_terms, entries);
    end = gettime();

    printf("Total time taken for insertion: %lf seconds\n", end - start);

    ofstream ofsfileoutput (outputFileName.c_str(), ios::out);
    ofstream ofsfiledistances (outputFileDistancesName.c_str(), ios::out);

    readTestFile(inverted_index, stats, testFileName,  k, distanceFunction, ofsfileoutput, ofsfiledistances);
    ofsfileoutput.close();
    ofsfiledistances.close();*/

    return EXIT_SUCCESS;
}


FileStats readTrainingFile(std::string &filename, std::vector<Entry> &entries) {
    std::ifstream input(filename.c_str());
    std::string line;

    FileStats stats;

    while(!input.eof()) {
        std::getline(input, line);
        if(line == "") continue;

        int doc_id = stats.num_docs++;
        std::vector<std::string> tokens = split(line, ' ');

        stats.doc_to_class[doc_id] = get_class(tokens[1]);

        for(int i = 2, size = tokens.size(); i + 1 < size; i+=2) {
            int term_id = atoi(tokens[i].c_str());
            int term_count = atoi(tokens[i+1].c_str());
            stats.num_terms = std::max(stats.num_terms, term_id + 1);
            entries.push_back(Entry(doc_id, term_id, term_count));
        }
    }

    input.close();

    return stats;
}

void updateStatsMaxFeatureTest(std::string &filename, FileStats &stats) {
    std::ifstream input(filename.c_str());
    std::string line;

    while(!input.eof()) {
        std::getline(input, line);
        if(line == "") continue;

        std::vector<std::string> tokens = split(line, ' ');
        for(int i = 2, size = tokens.size(); i + 1 < size; i+=2) {
            int term_id = atoi(tokens[i].c_str());
            stats.num_terms = std::max(stats.num_terms, term_id + 1);
        }
    }
}


void readTestFile(InvertedIndex &index, FileStats &stats, std::string &filename, int K, std::string distance, ofstream &outputfile, ofstream &outputdists) {
    std::ifstream input(filename.c_str());
    std::string line;

    std::vector<Entry> query;

    int num_tests = 0;
    int correct_l2 = 0, correct_cosine = 0, correct_l1=0;
    int wrong_l2 = 0, wrong_cosine = 0, wrong_l1=0;

    double start = gettime();

    while(!input.eof()) {
        std::getline(input, line);
        if(line == "") continue;
        num_tests++;

        if(distance == "cosine" || distance == "both") {
            if(makeQuery(index, stats, line, K, CosineDistance, outputfile, outputdists)) {
                correct_cosine++;
            } else {
                wrong_cosine++;
            }
        }

        if(distance == "l2" || distance == "both") {

            if(makeQuery(index, stats, line, K, EuclideanDistance,outputfile, outputdists)) {
                correct_l2++;
            } else {
                wrong_l2++;
            }
        }

        if(distance == "l1" || distance == "both") {
            if(makeQuery(index, stats, line, K, ManhattanDistance,outputfile, outputdists)) {
                correct_l1++;
            } else {
                wrong_l1++;
            }
        }

    }

    double end = gettime();

    printf("Time taken for %d queries: %lf seconds\n\n", num_tests, end - start);

    if(distance == "cosine" || distance == "both") {
        printf("Cosine similarity\n");
        printf("Correct: %d Wrong: %d\n", correct_cosine, wrong_cosine);
        printf("Accuracy: %lf%%\n\n", double(correct_cosine) / double(num_tests));
    }

    if(distance == "l2" || distance == "both") {
        printf("L2 distance\n");
        printf("Correct: %d Wrong: %d\n", correct_l2, wrong_l2);
        printf("Accuracy: %lf%%\n\n", double(correct_l2) / double(num_tests));
    }

    if(distance == "l1" || distance == "both") {
        printf("L1 distance\n");
        printf("Correct: %d Wrong: %d\n", correct_l1, wrong_l1);
        printf("Accuracy: %lf%%\n\n", double(correct_l1) / double(num_tests));
    }

}

bool makeQuery(InvertedIndex &inverted_index, FileStats &stats, std::string &line, int K,
               void (*distance)(InvertedIndex, Entry*, int*, cuSimilarity*, int D), ofstream &outputfile, ofstream &outputdists) {
    std::vector<Entry> query;

    std::vector<std::string> tokens = split(line, ' ');

    int clazz = get_class(tokens[1]);
    int docid = atoi(tokens[0].c_str());

    for(int i = 2, size = tokens.size(); i + 1 < size; i+=2) {
        int term_id = atoi(tokens[i].c_str());
        int term_count = atoi(tokens[i+1].c_str());

        query.push_back(Entry(0, term_id, term_count));
    }

    //Creates an empty document if there are no terms
    if(query.empty()) {
        query.push_back(Entry(0, 0, 0));
    }


    cuSimilarity *k_nearest = KNN(inverted_index, query, K, distance);
    std::map<int, int> vote_count;
    std::map<int, int>::iterator it;

    for(int i = 0; i < K; i++) {
        cuSimilarity &sim = k_nearest[i];
        vote_count[stats.doc_to_class[sim.doc_id]]++;
        outputdists<<sim.distance<<" ";
    }
    outputdists<<std::endl;

    int guessed_class = -1;
    int max_votes = 0;

    for(it = vote_count.begin(); it != vote_count.end(); it++) {
        if(it->second > max_votes) {
            max_votes = it->second;
            guessed_class = it->first;
        }
    }


    write_output(outputfile, clazz, guessed_class, docid);


    return clazz == guessed_class;
}


void write_output(ofstream &outputfile, int trueclass, int guessedclass, int docid) {

    outputfile << docid<<" CLASS="<<trueclass<<" CLASS="<<guessedclass<<":1"<<std::endl;
}

int get_class(std::string token) {
    std::vector<std::string> class_tokens = split(token, '=');

    if(class_tokens.size() == 1) {
        return atoi(class_tokens[0].c_str());
    } else {
        return atoi(class_tokens[1].c_str());
    }
}
