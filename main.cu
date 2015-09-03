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

#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"
#include  <cuda.h>

#include "Dataset.h"
#include "cuLazyNN_RF.h"

#include <map>
using namespace std;

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
               void (*distance)(InvertedIndex, Entry*, int*, Similarity*, int D), ofstream &fileout, ofstream &filedists);

void write_output(ofstream &fileout, int trueclass, int guessedclass, int docid);

int get_class(std::string token);


void teste_lazynn(int argc, char **argv){

    if(argc != 7) {
        std::cerr << "Wrong parameters. Correct usage: <executable> <training_file> <test_file> <k> <cosine | l2 | l1> <output_classifications_file> <output_distances_file>" << std::endl;
        exit(1);
    }

    std::string trainingFileName(argv[1]);
    std::string testFileName(argv[2]);
    int k = atoi(argv[3]);
    std::string distanceFunction(argv[4]);
    std::string outputFileName(argv[5]);
    std::string outputFileDistancesName(argv[6]);

    Dataset training_set, test_set;
    int correct_cosine = 0, wrong_cosine = 0;
    
    training_set.loadGtKnnFormat(trainingFileName.c_str());

    cuLazyNN_RF cLazy(training_set);
    test_set.loadGtKnnFormat(testFileName.c_str());
    double start, end, total = 0;
    for (int i = 0; i < test_set.getSamples().size(); ++i)
    {
        start = gettime();
        int guessed_class = cLazy.classify(test_set.getSamples()[i].features, k);
        end = gettime();
        printf("Total time taken for classification: %lf seconds\n", end - start);

        total += end - start;
	printf("Guessed class : %d - Real class : %d\n", guessed_class, test_set.getSamples()[i].y);
        if(guessed_class == test_set.getSamples()[i].y) {
            correct_cosine++;   
        } else {
            wrong_cosine++;
        }
    }

    printf("Total time taken to classify all queries: %lf seconds\n", total);

    printf("Cosine similarity\n");
    printf("Correct: %d Wrong: %d\n", correct_cosine, wrong_cosine);
    printf("Accuracy: %lf%%\n\n", double(correct_cosine) / double(test_set.size()));

}


/**
 * Receives as parameters the training file name and the test file name
 */
int main(int argc, char **argv) {
    initCudpp(); //initializes the CUDPP library
    cuInit(0);
    cudaDeviceSynchronize();

    teste_lazynn(argc, argv);

    if(argc != 7) {
        std::cerr << "Wrong parameters. Correct usage: <executable> <training_file> <test_file> <k> <cosine | l2 | l1> <output_classifications_file> <output_distances_file>" << std::endl;
        exit(1);
    }

    std::string trainingFileName(argv[1]);
    std::string testFileName(argv[2]);
    int k = atoi(argv[3]);
    std::string distanceFunction(argv[4]);
    std::string outputFileName(argv[5]);
    std::string outputFileDistancesName(argv[6]);

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
    ofsfiledistances.close();

    return 1;
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
               void (*distance)(InvertedIndex, Entry*, int*, Similarity*, int D), ofstream &outputfile, ofstream &outputdists) {
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


    Similarity *k_nearest = KNN(inverted_index, query, K, distance);
    std::map<int, int> vote_count;
    std::map<int, int>::iterator it;

    for(int i = 0; i < K; i++) {
        Similarity &sim = k_nearest[i];
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
