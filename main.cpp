#include <iostream>
#include <cmath>
#include <stdlib.h>


// activation function : Sigmoid function
double nonlin(double x, bool deriv=false)
{
	if (deriv)
	{
		return x *(1.0-x);
	}
	return 1.0 / (1.0 + exp(-x));
}

// network function
double neural_network_func(double X[], double y[], double weights[][1])
{
	// forward propagation
	double l0_weighted = 0;
	for(int i = 0; i < 3; i++)
	{
		l0_weighted += X[i] * weights[i][0];
	}
	double l1 = nonlin(l0_weighted);

	// calculate error
	double l1_error = y[0] - l1;

	// multiply error by the slope of 
	// the sigmoid at the values in l1
	double l1_delta = l1_error * nonlin(l1, true);

	// update weights
	for(int i = 0; i < 3; i++)
	{
		weights[i][0] += X[i] * l1_delta;
	}

	return l1;
}


int main (void)
{
	// |--------------------------|
	// | neural network			  |
	// | 3 inputs, 1 outputs	  |
	// | 4 test sets			  |
	// |--------------------------|

	std::cout << std::endl;

	// 3 inputs, 4 test sets
	double X[4][3] = {
		{0, 0, 1},
		{0, 1, 0},
		{1, 0, 1},
		{1, 1, 1}
	};

	// correct output for each test set
	double y[4][1] = {
		{0},
		{0},
		{1},
		{1}
	};

	// weights
	double weights[3][1];
	for(int row = 0; row < 3; row++)
	{
		weights[row][0] = (double)((rand()%100)/100);
	}

	//actual outputs
	double yHat[4][1];


	// train NN for 10000 iterations
	for(int j = 0; j<1000000; j++)
	{
		// for each test set
		for(int test_set = 0; test_set < 4; test_set++)
		{
			// output matrix value
			yHat[test_set][0] = neural_network_func(X[test_set], y[test_set], weights);
		}
	}

	// print output for each test set
	for(int test_set = 0; test_set < 4; test_set++)
	{
		std::cout << yHat[test_set][0] << std::endl;
	}

	// std::cout << std::endl;
	// std::cout << neural_network_func(X[3], y[3], weights) << std::endl;

	return 0;
}
