#pragma once
#include"globals.h"
/**
 * Retorna Matrizes com pontos e rotulos de um arquivo
 * @param inputFolder - endereco da pasta do arquivo
 * @param setN - numero do set do arquivo
 * @param X - referencia para matriz
 * @param Y - referencia para vetor coluna
 *
 * @pre	arquivos relvenates para X e Y dentro da pasta tem nome
 *   "SSL,set=" + std::to_string(setN) + ",X.tab";
 *   "SSL,set=" + std::to_string(setN) + ",y.tab";}
 * @post X possui informacao de coordenadas dos pontos
 * @post Y possui rotulos dos pontos
 */
void getInputMatrices(std::string inputFolder, int setN,Mat &X,  CVec &Y){
	    std::string filenameX = inputFolder + "SSL,set=" + std::to_string(setN) + ",X.tab";
		std::string filenameY = inputFolder + "SSL,set=" +  std::to_string(setN)  + ",y.tab";
		std::ifstream Xstream;
		Xstream.open (filenameX.c_str(),std::ios::in);

		double val;

		int sizeX = 0;
		std::cout << "Reading X data from " << filenameX << std::endl;
		std::vector<double> entriesX;
		while (Xstream >> val) {
			entriesX.push_back(val);
			sizeX++;
		}
		std::cout << "Number of values in X data: " << sizeX << std::endl;
		Xstream.close();

		std::ifstream Ystream;
		Ystream.open (filenameY.c_str(),std::ios::in);
		int sizeY = 0;
		std::cout << "Reading Y data from " << filenameY << std::endl;
		std::vector<double> entriesY;
		while (Ystream >> val) {
			if (val == -1) val = 0;
			entriesY.push_back(val);
			sizeY++;
		}
		std::cout << "Number of values in Y data: " << sizeY << std::endl;
		Ystream.close();

		long rows(sizeY);
		long cols = sizeX/sizeY;
		std::pair<long,long> dim(rows,cols);
		std::cout << "Assumed Matrix dimension: " << dim.first << " by " << dim.second << std::endl;
		//Mapear para as matrizes
		X.resize(sizeX,1);
		for (int i = 0; i < sizeX; i++) {
			X(i,0) = entriesX[i];
		}
		X.resize(rows,cols);

		Y.resize(sizeY);
		for (int i = 0; i < sizeY; i++) {
			Y(i) = entriesY[i];
		}
}
/**
 * Escreve matriz para arquivo
 */
void writeToFile(Mat m, std::string outputFile) {
	std::ofstream stream;
	stream.open (outputFile.c_str());
	//std::cout << "Writing to " << outputFile << std::endl;
	stream << m;
	stream.close();
}

/**
 * Seleciona aleatoriamente sem repeticao numeros de 0 ateh certo valor
 * @param N valor max
 * @param M quantidade de numeros a serem escolhidos
 * @return vetor com numeros escolhidos
 */
std::vector<int> getSamples(int N, int M, CVec Y) {
	std::vector<int> vec;
	for (int i = 0; i < N; i++) {
		vec.push_back(i);
	}
	std::random_shuffle(vec.begin(),vec.end());
	std::vector<int> vec2;

	int* count = (int*) calloc(NUM_CLASSES, sizeof(int));

	for (int i = 0; i < M; i++) {
		int c = std::round(Y[vec[i]]);
		if (count[c] > M/NUM_CLASSES) {
			continue;
		}
		vec2.push_back(vec[i]);
		count[c]++;
	}
	return vec2;
}



/**
 * Retorna matriz de probabilidades iniciais, consistente com rotulos dos samples fornecidos
 *
 * @param Y vetor  coluna com os rotulos
 * @param samples indices cujos rotulos serao utilizados como conhecimento a priori
 * @return matriz I com probabilidades iniciais
 * @pre rotulos de Y estao entre [0,NUM_CLASSES]
 * @post (I.rows(),I.cols()) == (N,C) sendo
 * 			N - numero de pontos
 * 			C - NUM_CLASSES
 * 		e I(i,j) = 1.0 <==> Y(j) == i
 */
Mat getInitialProb(CVec Y, std::vector<int> samples){
	Mat Init(Y.rows(),NUM_CLASSES);

	for (int i = 0; i < Y.rows(); i++) {
		for (int j = 0; j < NUM_CLASSES; j++) {
			Init(i,j) = 0.0;
		}
	}

	for (int i = 0; i < samples.size(); i++) {
		Init(samples[i], Y(samples[i]) ) = 1.0;
	}

	return Init;
}

CVec getClassification(Mat probs) {
	CVec classification(probs.rows());
	for (int i = 0; i < probs.rows(); i++) {
		double max = probs(i,0);
		int maxID = 0;
		for (int j = 1; j < probs.cols(); j++) {
			if (probs(i,j) > max) {
				max = probs(i,j);
				maxID = j;
			}
		}
		classification(i) = maxID;
	}
	return classification;
}

double getErrorRate(CVec Y, CVec classification) {
	double count = 0;
	for (int i = 0; i < Y.rows(); i++) {
		if (std::round(Y(i)) != std::round(classification(i)) ) {
			count++;
		}
	}
	return count/(double) Y.rows();
}
