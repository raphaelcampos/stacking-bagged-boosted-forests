#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define __USE_FILE_OFFSET64
#define __USE_LARGEFILE64

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define SIZE_OF_LINE 100000

int main (int argc, char *argv[]){
	FILE *inFile;
	FILE **outFiles;
	int fildesIn,fildesOut;
	int k,minSize, foldSize, fileSize;
	int selectedPart;
	int numPartes;
	int *numLines;
	char *inputFileName = NULL;
	char *outputFileName = NULL;
	char *name = (char *) malloc(sizeof(char) * 200);
	char *line = (char*) malloc(sizeof(char)*SIZE_OF_LINE);
	struct timeval semente;

	// utilizar o tempo como semente para a funcao srand48()
    	gettimeofday(&semente,NULL); 
    	srand48((int)(semente.tv_sec + 1000000*semente.tv_usec));

	//pega os parametros de entrada
	inputFileName = argv[1];
	outputFileName = argv[2];
	numPartes = atoi(argv[3]);
	fileSize = atoi(argv[4]);

	// calcula o tamanho de cada fold
	foldSize = fileSize/numPartes;

	//aloca espaco
	outFiles = (FILE**) malloc(sizeof(FILE*)*numPartes);
	numLines = (int*) malloc(sizeof(int)*numPartes);
	
	//abre os arquivos de saida: um arquivo para cada parte
	for(k=0;k<numPartes;k++){
		sprintf(name,"%s%d",outputFileName,k);
		fildesOut = open(name, O_WRONLY|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
		if(fildesOut < 0){
			perror("Arquivo de saida");
			exit(1);
		}
		outFiles[k] = fdopen (fildesOut, "w");
		numLines[k] = 0;
	}
	
	//abre arquivo de entrada
	fildesIn  = open(inputFileName,O_RDONLY);
	if(fildesIn < 0){
		perror("Arquivo de Entrada");
		exit(1);
	}
	inFile   = fdopen (fildesIn, "r");

	//le o arquivo de entrada
	char *rc=fgets(line ,SIZE_OF_LINE, inFile); (void) rc;
	while (!feof(inFile)){
		// seleciona aleatoriamente fold a ser impressa a linha atual
		selectedPart = (int) (drand48()*numPartes);

		// seleciona fold com menor numero de linhas
		if( !(numLines[selectedPart] < foldSize) ){
			selectedPart = 0;
			minSize = numLines[0];
			for(k=1;k<numPartes;k++){
				if(numLines[k] < minSize){
					selectedPart = k;
					minSize = numLines[k];
				}	
			}	
		}

		//imprime no arquivo de saida
		fprintf(outFiles[selectedPart],"%s",line); fflush(outFiles[selectedPart]);
		numLines[selectedPart]++;

		// le a proxima linha
		rc=fgets(line,SIZE_OF_LINE, inFile); (void) rc;
	}

	// fecha arquivos de saida
	for(k=0;k<numPartes;k++){
		fclose(outFiles[k]);
	}

	// fecha arquivo de entrada
	fclose(inFile);
	
	//libera memorias alocadas
	free(line);
	free(outFiles);
	free(numLines);
	free(name);

	return 0;
}
