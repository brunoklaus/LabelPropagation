#pragma once
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include"globals.h"



CVec getClassFreq(CVec M);


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
		for (int i = 0; i < cols; i++) {
			double max = X.col(i).maxCoeff();
			double min = X.col(i).minCoeff();
			double r = max-min;
			if (!std::isfinite(r)) {
				X.col(i).setOnes();
				X.col(i) = X.col(i) * 0.5;
			} else {
				X.col(i) = (X.col(i) - CVec::Constant(X.col(i).size(),min)) / r;
			}
		}

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
 * Seleciona aleatoriamente sem repeticao numeros de 0 ateh certo valor, de maneira a obter #igual de repr. pra cada classe.
 * @param N valor max
 * @param M quantidade de numeros a serem escolhidos
 * @param Y vetor do qual obtem-se as amostras
 * @return vetor com numeros escolhidos
 */
std::vector<int> getBalancedSamples(int N, int M, CVec Y) {
	std::vector<int> vec;
	for (int i = 0; i < N; i++) {
		vec.push_back(i);
	}
	std::random_shuffle(vec.begin(),vec.end());
	std::vector<int> vec2;
	CVec classes(M);

	int* count = (int*) calloc(NUM_CLASSES, sizeof(int));

	int* maxAllowed = (int*) calloc(NUM_CLASSES, sizeof(int));
	maxAllowed[0] = M -  (NUM_CLASSES-1)*(M/NUM_CLASSES);
	for (int i = 1; i < NUM_CLASSES; i++) {
		maxAllowed[i] = M/NUM_CLASSES;
	}


	for (int i = 0; i < N && vec2.size() != M; i++) {
		int c = std::round(Y[vec[i]]);
		if (count[c] + 1 > maxAllowed[c]) {
			continue;
		}
		vec2.push_back(vec[i]);
		classes(vec2.size()-1) = c;
		count[c] = count[c] + 1;
	}
	free(count);
	free(maxAllowed);
	if(F_DEBUG){
		std::cout <<"Sample class frequency: " << getClassFreq(classes).transpose() << std::endl;
	}
	return vec2;
}



/**
 * Seleciona aleatoriamente sem repeticao numeros de 0 ateh certo valor (possivelmente desbalanceado)
 * @param N valor max
 * @param M quantidade de numeros a serem escolhidos
 * @param Y vetor do qual obtem-se as amostras
 * @return vetor com numeros escolhidos
 */
std::vector<int> getRawSamples(int N, int M, CVec Y) {
	std::vector<int> vec;
	for (int i = 0; i < N; i++) {
		vec.push_back(i);
	}
	std::random_shuffle(vec.begin(),vec.end());
	std::vector<int> vec2;
	CVec classes(M);

	int* count = (int*) calloc(NUM_CLASSES, sizeof(int));

	int* maxAllowed = (int*) calloc(NUM_CLASSES, sizeof(int));
	maxAllowed[0] = M -  (NUM_CLASSES-1)*(M/NUM_CLASSES);
	for (int i = 1; i < NUM_CLASSES; i++) {
		maxAllowed[i] = M/NUM_CLASSES;
	}


	for (int i = 0; i < N && vec2.size() != M; i++) {
		int c = std::round(Y[vec[i]]);
		vec2.push_back(vec[i]);
		classes(vec2.size()-1) = c;
		count[c] = count[c] + 1;
	}
	free(count);
	free(maxAllowed);
	if(F_DEBUG){
		std::cout <<"Sample class frequency: " << getClassFreq(classes).transpose() << std::endl;
	}
	return vec2;
}
/**
 * Seleciona aleatoriamente sem repeticao numeros de 0 ateh certo valor
 * @param N valor max
 * @param M quantidade de numeros a serem escolhidos
 * @param Y vetor do qual obtem-se as amostras
 * @param _seed uma seed para replicar experimentos
 * @return  vetor com numeros escolhidos
 */
std::vector<int> getBalancedSamples(int N, int M, CVec Y, unsigned int _seed) {
	std::srand(_seed);
	return getBalancedSamples(N,M,Y);
}

/**
 * Seleciona aleatoriamente sem repeticao numeros de 0 ateh certo valor (possivelmente desbalanceado)
 * @param N valor max
 * @param M quantidade de numeros a serem escolhidos
 * @param Y vetor do qual obtem-se as amostras
 * @param _seed uma seed para replicar experimentos
 * @return  vetor com numeros escolhidos
 */
std::vector<int> getRawSamples(int N, int M, CVec Y, unsigned int _seed) {
	std::srand(_seed);
	return getRawSamples(N,M,Y);
}



/**
 * Executa comando no bash.
 * @param cmd comando
 */
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}
void createSubfolderPath(std::string subfolderPath){
	//Comando para criar subfolder de entrada
	std::ostringstream commandCriaSub;
	commandCriaSub << "mkdir -p " << subfolderPath;
	std::cout << commandCriaSub.str() << std::endl;
	exec(commandCriaSub.str().c_str());
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
		Init(samples[i], std::round(Y(samples[i])))  = 1.0;
	}

	return Init;
}
/**
 * Obtém um vetor coluna com as classificações de classes a partir de uma matriz
 * nxc com probabilidades de cada uma das c classes para cada um dos n pontos.
 * @param probs matriz com probabilidades de classificação
 */
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

/**
 * Reporta a porcentagem de erros na classificação
 * @param Y classes desejadas
 * @param classes obtidas na classificação
 */
double getErrorRate(CVec &Y, CVec &classification) {
	double count = 0;
	for (int i = 0; i < Y.rows(); i++) {
		if (std::round(Y(i)) != std::round(classification(i)) ) {
			count++;
		}
	}
	return count/(double) Y.rows();
}


/**
 * Reporta a porcentagem de erros na classificação
 * @param Y classes desejadas
 * @param classes obtidas na classificação
 * @param test_indices indices a serem testados
 */
double getErrorRate(CVec &Y, CVec &classification, std::vector<int> &test_indices) {
	double count = 0;
	for (int i = 0; i < test_indices.size(); i++) {
		if (std::round(Y(test_indices[i])) != std::round(classification(test_indices[i])) ) {
			count++;
		}
	}

	return count/(double) test_indices.size();
}

/**
 * Debug. Joga logic_error caso haja entrada não finita na matriz.
 * @param M matriz
 */
void checkFinite(Mat& M){
	for (int i = 0; i < M.rows();i++){
		for (int j = 0; j < M.cols();j++) {
			if (!std::isfinite(M(i,j))) {
				throw std::logic_error("Non-finite value found in matrix\n");
			}
		}
	}
}
/**
 * Retorna uma medida alternativa de erro que depende da confiança no rótulo correto. O acerto
 * é determinado pela soma de probabilidades da decisão correta de cada caso.
 * @param Y vetor com valores corretos
 * @param C matriz cuja j-ésima entrada da i-ésima linha corresponde à probabilidade do ponto #i ser da classe #j
 */
double getProbabilisticError(CVec &Y, Mat C) {

	if (F_DEBUG) {
		if (C.cols() != NUM_CLASSES || C.rows()!=Y.rows()) {
			throw std::logic_error("Wrong probability matrix given to getProbabilisticError\n");
		}

	}

	Mat copy(C);
	for (int i = 0; i < C.rows();i++) {
		if (std::isfinite(1.0/(double)C.row(i).sum() * C.row(i).maxCoeff())) {
			C.row(i) = 1.0/(double)C.row(i).sum() * C.row(i);
		} else {
			C.row(i) = CVec::Ones(C.row(i).size()) / (double) C.row(i).size();
		}
	}

	double sum = 0;
	for (int i = 0; i < Y.rows(); i++) {
		int correct = std::round(Y(i));
		double d = C(i,correct);
		sum += C(i,correct);
	}

	return 1 - sum/(double) Y.rows();
}

/**
 * Conta a frequencia de cada valor em um vetor coluna
 * @param M vetor coluna
 * @return vetor coluna _freq tal que \forall c: c \in [0,max(M)] : _freq[c] = |i:M[i]==c|
 */
CVec getClassFreq(CVec M) {
	CVec freq = CVec::Zero(NUM_CLASSES);
	for (int i = 0; i < M.rows(); i++) {
		freq[std::round(M(i))]++;
	}
	return freq;
}



/**
 * Faz a leitura dos melhores valores pro algoritmo LGC
 * @param filenameBest arquivo com melhores params. Ele tem um header com atributos
 * 		"num_samples","set","best_alpha","best_sigma"
 * @param alpha pointeiro, cujo valor ao final será o do melhor alpha
   @param alpha pointeiro, cujo valor ao final será o do melhor sigma
*/
void readBestLGC(std::string filenameBest, double* alpha, double* sigma, int setNumber = setN){

		std::ifstream filestream;
		filestream.open (filenameBest.c_str(),std::ios::in);
		std::vector<std::string> attributes;

		std::string line;
		std::getline(filestream,line);
		std::istringstream initLineStream(line);


		std::string temp;
		while (initLineStream >> temp) {
			attributes.push_back(temp.substr(1, temp.size()-2));
		}
		int index_numsamples = std::find(attributes.begin(),attributes.end(),"num_samples") - attributes.begin();
		int index_set = std::find(attributes.begin(),attributes.end(),"set") - attributes.begin();
		int index_alpha = std::find(attributes.begin(),attributes.end(),"best_alpha") - attributes.begin();
		int index_sigma = std::find(attributes.begin(),attributes.end(),"best_sigma") - attributes.begin();


		while(std::getline(filestream,line)) {
			std::istringstream lineStream(line);
			std::find(attributes.begin(),attributes.end(),"num_samples") - attributes.begin();
			CVec v(attributes.size());
			for (int i = 0; i < attributes.size();i++) {
				lineStream >> v[i];
			}
			if (v[index_numsamples] == 100 && v[index_set] == setNumber) {
				std::cout << "Best alpha for " << v[index_set] << " is " << v[index_alpha] << std::endl;
				std::cout << "Best sigma for " << v[index_set] << " is " << v[index_sigma] << std::endl;
				*alpha = v[index_alpha];
				*sigma = v[index_sigma];
			}

		}

}

double getVariance(CVec Y) {

	static double mean = Y.mean();
	double var = 0;
	for (int i = 0; i < Y.size(); i++){
		var = var + (Y(i)-mean)*(Y(i)-mean);
	}
	std::cout << "var " <<  var / (double) (Y.size() - 1)  << std::endl;




	double sum = Y.sum();
	double sumSquared = Y.cwiseProduct(Y).sum();

	var = (sumSquared - sum*sum/(double)Y.size()   ) / (double) (Y.size() - 1);
	std::cout << "var " <<  var  << std::endl;
	return var;
}

/**
 * Retorna lista de valores do parâmetro alpha a serem usados no algoritmo LGC.
 */
std::vector<double> defaultAlphas (){
	std::vector<double> alphas({0.1,0.2,0.4,0.6,0.8});
	for (double alpha = 0.9; alpha < 1; alpha += 0.1) {
				alphas.push_back(alpha);
			}
	alphas.push_back(0.95);
	alphas.push_back(0.99);
	alphas.push_back(0.999);
	alphas.push_back(0.9999);
	return alphas;
}
/**
 * Retorna lista de valores de sigma a serem usados.
 */
std::vector<double> defaultSigmas() {

	std::vector<double> sigmas({0.005,0.05,0.1,0.15});



	for (double var = 0.2; var <= 1; var += 0.1) {
		sigmas.push_back(var);
	}
	sigmas.push_back(2);
	sigmas.push_back(4);
	return sigmas;
}
