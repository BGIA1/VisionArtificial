#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>


const double e = 2.71828182845904523536;
const double pi = 3.14159265358979311600;
double c = 1 / (2 * pi);

using namespace cv;
using namespace std;

//Kernel Gaussiano
double** crearKernel(int N, double sigma)
{
	double** kernel = new double* [N];	//Reservo memoria para el kernel
	int m = N / 2;	//Mitad del tamano del kernel
	c /= (sigma * sigma);

	for (int i = 0; i < N; i++)
	{
		kernel[i] = new double[N];	//Reservar memoria en la i-esima posicion del kernel
		for (int j = 0; j < N; j++)
		{
			double x = j - m;
			double y = (i - m) * -1;
			kernel[i][j] = c * pow(e, (-((x * x) + (y * y)) / (2 * (sigma * sigma))));	//Funcion Gaussiana

		}

	}

	return kernel;
}


Mat aplicarBordes(Mat imagen, int N, int borde)
{

	int rows = imagen.rows;	//Numero de filas
	int cols = imagen.cols; //Numero de columnas

	Mat imagenG(rows + borde * 2, cols + borde * 2, CV_8UC1);
	Mat imagenB(rows + borde * 2, cols + borde * 2, CV_8UC1);	//Mat para la imagen resultante con bordes

	cvtColor(imagen, imagenG, COLOR_BGR2GRAY);		//Se obtiene la matriz de niveles de gris de la imagen 

	double niv_gris;

	for (int i = 0; i < rows + borde * 2; i++) {
		for (int j = 0; j < cols + borde * 2; j++) {//recorrido de la matriz (x,y)

			if ((i >= rows + borde || i < borde) || (j >= cols + borde || j < borde)) {		//En caso de que i o j se encuentren en la parte de los bordes
				imagenB.at<uchar>(Point(i, j)) = uchar(0);	//Se coloca un nivel 0 
			}
			else {	//En caso contrario

				niv_gris = imagenG.at<uchar>(Point(j - borde, i - borde));
				imagenB.at<uchar>(Point(j, i)) = uchar(niv_gris);	//Coloca el nivel de gris de la imagen original
			}

		}
	}
	/*
	//Impresion de dimensiones de la imagen 
	cout << "Numero de filas imagen original " << rows << endl;
	cout << "Numero de columnas imagen original " << cols << endl;

	cout << "Numero de filas imagen con bordes " << imagenB.rows << endl;
	cout << "Numero de columnas imagen con bordes " << imagenB.cols << endl;
	*/
	return imagenB;
}


Mat reshape(Mat imagen, int borde)
{

	int rows = imagen.rows;	//Numero de filas
	int cols = imagen.cols; //Numero de columnas
	Mat imagenB(rows - 2 * borde, cols - 2 * borde, CV_8UC1); // Matriz para la imagen resultante


	double niv_gris;

	for (int i = borde; i < rows - borde; i++) {//Recorrido de la matriz (x,y)
		for (int j = borde; j < cols - borde; j++) {

			niv_gris = imagen.at<uchar>(Point(j, i));//Obtencion del nivel de gris 
			imagenB.at<uchar>(Point(j - borde, i - borde)) = uchar(niv_gris);//Copia del nivel de gris en la posicion corrspondiente
		}
	}
	/*
	//Impresion de dimensiones de la imagen 
	cout << "Numero de filas imagen redimensionada " << imagenB.rows << endl;
	cout << "Numero de columnas imagen redimensionada " << imagenB.cols << endl;
	*/
	return imagenB;
}

int aplicarConvolucion(Mat imagenBordes, double** kernel, int borde, int x, int y)
{
	int b = (2 * borde) + 1;
	int valor = 0;

	for (int i = 0; i < b; i++) {
		for (int j = 0; j < b; j++) {

			valor += int((imagenBordes.at<uchar>(Point(y - borde + i, x - borde + j))) * (kernel[i][j]));

		}
	}

	return valor;
}

Mat aplicarFiltro(Mat imagenBordes, double** kernel, int borde, int selec)
{
	int rows = imagenBordes.rows;//Numero de filas
	int cols = imagenBordes.cols; //Numero de columnas
	Mat imagenR(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; i++) {//recorrido de la matriz (x,y)
		for (int j = 0; j < cols; j++) {
			if (imagenBordes.at<uchar>(Point(i, j)) != uchar(0))//Aplicacion del filtro usando el kernel
			{
				if (selec == 0) {
					imagenR.at<uchar>(Point(i, j)) = uchar(aplicarConvolucion(imagenBordes, kernel, borde, j, i));
				}
				else
				imagenR.at<uchar>(Point(i, j)) = uchar(aplicarConvolucion(imagenBordes, kernel, borde, j, i)/selec);

				
			}
		}
	}
	/*
	//Impresion de dimensiones de la imagen 
	cout << "Numero de filas imagen procesada " << rows << endl;
	cout << "Numero de columnas imagen procesada " << cols << endl;
	*/
	return imagenR;
}



//kernels sobel manuales
double** crearKernelIn(int N)
{
	double** kernel = new double* [N];	//Reservo memoria para el kernel
	int x = 0;

	for (int i = 0; i < N; i++)
	{
		kernel[i] = new double[N];	//Reservar memoria en la i-esima posicion del kernel
		for (int j = 0; j < N; j++)
		{
			cout << "Ingrese el valor del kernel en la posicion [" << i << "][" << j << "]\n" << endl; //Valores del kernel
			cin >> x;
			kernel[i][j] = x;

		}

	}

	return kernel;
}

//kernels sobel automaticos de 3x3
double** crearKernelInPruebax(int N)
{
	double** kernel = new double* [N];	//Reservo memoria para el kernel

	for (int i = 0; i < N; i++)
	{
		kernel[i] = new double[N];	//Reservar memoria en la i-esima posicion del kernel


	}
	kernel[0][0] = -1;
	kernel[0][1] = 0;
	kernel[0][2] = 1;
	kernel[1][0] = -2;
	kernel[1][1] = 0;
	kernel[1][2] = 2;
	kernel[2][0] = -1;
	kernel[2][1] = 0;
	kernel[2][2] = 1;
	return kernel;
}

double** crearKernelInPruebay(int N)
{
	double** kernel = new double* [N];	//Reservo memoria para el kernel

	for (int i = 0; i < N; i++)
	{
		kernel[i] = new double[N];	//Reservar memoria en la i-esima posicion del kernel


	}
	kernel[0][0] = -1;
	kernel[0][1] = -2;
	kernel[0][2] = -1;
	kernel[1][0] = 0;
	kernel[1][1] = 0;
	kernel[1][2] = 0;
	kernel[2][0] = 1;
	kernel[2][1] = 2;
	kernel[2][2] = 1;
	return kernel;
}

Mat resultadoSobel(Mat imagenX, Mat imagenY)
{
	int rows = imagenX.rows;//Numero de filas
	int cols = imagenX.cols; //Numero de columnas
	Mat imagenR(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; i++) {//recorrido de la matriz (x,y)
		for (int j = 0; j < cols; j++) {

			{	//Combinacion de Gx y Gy para obtener la magnitud del gradiente 
				imagenR.at<uchar>(Point(i, j)) = uchar(sqrt((pow(imagenX.at<uchar>(Point(i, j)), 2) + pow(imagenY.at<uchar>(Point(i, j)), 2))));


			}
		}
	}
	/*
	//Impresion de dimensiones de la imagen 
	cout << "Numero de filas imagen procesada (Filtro sobel) " << rows << endl;
	cout << "Numero de columnas imagen procesada (Filtro sobel) " << cols << endl;
	*/
	return imagenR;
}

Mat aplicarBordesGray(Mat imagen, int N, int borde)
{

	int rows = imagen.rows;	//Numero de filas
	int cols = imagen.cols; //Numero de columnas

	Mat imagenB(rows + borde * 2, cols + borde * 2, CV_8UC1);	//Mat para la imagen resultante con bordes


	double niv_gris;

	for (int i = 0; i < rows + borde * 2; i++) {
		for (int j = 0; j < cols + borde * 2; j++) {//recorrido de la matriz (x,y)

			if ((i >= rows + borde || i < borde) || (j >= cols + borde || j < borde)) {		//En caso de que i o j se encuentren en la parte de los bordes
				imagenB.at<uchar>(Point(i, j)) = uchar(0);	//Se coloca un nivel 0 
			}
			else {	//En caso contrario

				niv_gris = imagen.at<uchar>(Point(j - borde, i - borde));
				imagenB.at<uchar>(Point(j, i)) = uchar(niv_gris);	//Coloca el nivel de gris de la imagen original
			}

		}
	}
	/*Impresion de dimensiones de la imagen 
	cout << "Numero de filas imagen original " << rows << endl;
	cout << "Numero de columnas imagen original " << cols << endl;

	cout << "Numero de filas imagen con bordes " << imagenB.rows << endl;
	cout << "Numero de columnas imagen con bordes " << imagenB.cols << endl;
	*/
	return imagenB;
}

Mat filtroSobel(Mat imagen, double** kernelX, double** kernelY, int bordeSobel, int N_Sobel, int selec) {

	Mat img, imageny, imagenx, rsobel, Resultado, bordes;
	img = aplicarBordesGray(imagen, N_Sobel, bordeSobel);
	imagenx = aplicarFiltro(img, kernelX, bordeSobel,selec);
	imageny = aplicarFiltro(img, kernelY, bordeSobel,selec);
	rsobel = resultadoSobel(imagenx, imageny);
	Resultado = reshape(rsobel, bordeSobel);
	/*
	imshow("Imagen X", imagenx);//Filtro sobel Gx
	imshow("Imagen Y", imageny);//Filtro sobel Gy
	*/
	return Resultado;

}


int main()
{

	/********Variables*********/
	char NombreImagen[] = "lena.png";
	int N = 0, borde; int N_Sobel = 3; int bordeSobel = 1; int selector = 0;
	double sigma;
	Mat imagen, imagenBordes, imagenResultante, imagenResultanteReshape, imagenEscalaGrises, imgSobel, imagenH, imagenCanny;


	/*********Lectura de la imagen*********/
	imagen = imread(NombreImagen);

	if (imagen.empty())
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		return -1;
	}


	/************Procesamiento*********/

	cout << "Ingrese el valor de N\n";
	cin >> N;
	//Obtencion de las dimenciones del kernel
	while (N % 2 == 0)
	{
		cout << "Por favor ingresa un valor impar para N\n";
		cin >> N;
	}



	cout << "Ingrese el valor de Sigma\n";
	cin >> sigma;
	//Obtencion del valor de sigma
	while (sigma == 0)
	{
		cout << "Por favor ingresa un valor distinto de 0 para sigma\n";
		cin >> sigma;
	}

	borde = N / 2;//Obtencion del tamaño de bordes necesarios para pasar el kernel sin desbordamiento

	double** kernel = crearKernel(N, sigma);//Creacion del kernel

	cvtColor(imagen, imagenEscalaGrises, COLOR_BGR2GRAY);//Imagen en escala de grises

	imagenBordes = aplicarBordesGray(imagenEscalaGrises, N, borde);	//Se añaden los bordes a la imagen  segun el kernel
	imagenResultante = aplicarFiltro(imagenBordes, kernel, borde, selector);//Se aplica el filtro de suavisado a la imagen con bordes añadidos
	imagenResultanteReshape = reshape(imagenResultante, borde);//Se eliminan los bordes extras de la imagen para obtener el tamaño original


/*
	//Valores manuales del kernel

	cout << "Ingrese el valor de N_sobel\n";
	cin >> N_Sobel;
	//Obtencion de las dimenciones del kernel
	while ((N_Sobel % 2 == 0) || (N_Sobel <= 0))
	{
		cout << "Por favor ingresa un valor valido N_sobel\n";
		cin >> N_Sobel;
	}
	bordeSobel = N_Sobel / 2;

	 //Valores manuales del kernel

	cout << "kernel sobel x\n";
		double** kernelx = crearKernelIn(N_Sobel);

		cout << "kernel sobel y\n";
		double** kernely = crearKernelIn(N_Sobel);
*/
		//kernels filtro sobel
	double** kernelx = crearKernelInPruebax(N_Sobel);
	double** kernely = crearKernelInPruebay(N_Sobel);
	
	//aplicacion del filtro sobel a la imagen suavisada
	// 
	//imgSobel = filtroSobel(imagenResultanteReshape, kernelx, kernely, bordeSobel, N_Sobel, selector);
	

	imagenH = imagenResultanteReshape;

	vector<float>probability_density(256, 0);
	vector<float>probability_distribution(256, 0);

	// Estadísticas de la densidad de probabilidad de la imagen original y la distribución de probabilidad de la imagen original


	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < imagenH.rows; j++)
		{
			for (int k = 0; k < imagenH.cols; k++)
			{
				if (imagenH.at<unsigned char>(j, k) == i)
				{
					probability_density[i]++;
				}
			}
		}
		probability_density[i] = probability_density[i] / (imagenH.rows * imagenH.cols);
		if (i != 0)
		{
			probability_distribution[i] = probability_distribution[i - 1] + probability_density[i];
		}
		else
		{
			probability_distribution[i] = probability_density[i];
		}
	}
	// Convertir la distribución de probabilidad de la imagen original (tipo flotante) a su entero más cercano
	for (int i = 0; i < 256; i++)
	{
		round(probability_distribution[i]);
	}
	// Usa la función de distribución como una función de transformación para la ecualización del histograma
	for (int i = 0; i < imagenH.rows; i++)
	{
		for (int j = 0; j < imagenH.cols; j++)
		{
			for (int k = 0; k < 256; k++)
			{
				if (imagenH.at<unsigned char>(i, j) == k)
				{
					imagenH.at<unsigned char>(i, j) = 255 * (probability_distribution[k]);
					break;
				}
			}
		}
	}

/*	//Imagenes resultantes
	imshow("Imagen Original", imagen);
	imshow("Imagen Original en escala de grises", imagenEscalaGrises);
	imshow("Imagen Original con bordes", imagenBordes);
	imshow("Imagen filtrada con bordes", imagenResultante);
	imshow("Imagen Suavisada", imagenResultanteReshape);
	imshow("Imagen Resultante Sobel (G)", imgSobel);
*/

	imgSobel = filtroSobel(imagenH, kernelx, kernely, bordeSobel, N_Sobel, selector);

	cv::Canny(imagenResultanteReshape,            // input image
		imagenCanny,                    // output image
		100,                        // low threshold
		200);                        // high threshold
	int im1, im2, im3, im4, im5, im6, im7, im8, im9, im10,im11,im12;


	cout << "Tamaño del kernel: " << N << "x" << N << endl;
	cout << "Valor de sigma: " << sigma << endl;

	im1 = imagen.rows;
	im2 = imagen.cols;
	cout << "Dimensiones imagen original: " << im1 << "x" << im2 << endl;

	im3 = imagenEscalaGrises.rows;
	im4 = imagenEscalaGrises.cols;
	cout << "Dimensiones imagen en escala de grises: " << im3 << "x" << im4 << endl;

	im5 = imagenResultanteReshape.rows;
	im6 = imagenResultanteReshape.cols;
	cout << "Dimensiones imagen suavisada (Gauus): " << im5 << "x" << im6 << endl;

	im7 = imagenH.rows;
	im8 = imagenH.cols;
	cout << "Dimensiones imagen ecualizada: " << im7 << "x" << im8 << endl;

	im9 = imgSobel.rows;
	im10 = imgSobel.cols;
	cout << "Dimensiones imagen procesada filtro Sobel (|G|): " << im9 << "x" << im10 << endl;

	im11 = imagenCanny.rows;
	im12 = imagenCanny.cols;
	cout << "Dimensiones imagen deteccion de borde(Canny): " << im11 << "x" << im12 << endl;



	imshow("Imagen Original", imagen);
	imshow("Imagen Original en escala de grises", imagenEscalaGrises);
	imshow("Imagen Suavisada", imagenResultanteReshape);
	imshow("ecualización de histograma", imagenH);
	imshow("Imagen Resultante Sobel (G)", imgSobel);
	imshow("Imagen Resultante Canny", imagenCanny);

	waitKey(0); //Funcion para esperar
	return 0;
}
