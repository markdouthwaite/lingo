package lingo

import (
	"math"
	"testing"
)

func bostonX() [][]float64 {
	return [][]float64{
		{5.0, 296.0, 16.6, 395.5, 9.04},
		{4.0, 254.0, 17.6, 396.9, 3.53},
		{4.0, 711.0, 20.1, 390.11, 18.07},
		{4.0, 305.0, 19.2, 390.91, 5.52},
		{24.0, 666.0, 20.2, 385.09, 17.27},
		{5.0, 398.0, 18.7, 373.66, 11.97},
		{4.0, 304.0, 18.4, 390.7, 18.33},
		{4.0, 437.0, 21.2, 388.08, 24.16},
		{24.0, 666.0, 20.2, 395.33, 12.87},
		{6.0, 391.0, 19.2, 396.9, 14.33},
	}
}

func bostonH() []float64 {
	return []float64{
		28.083266491089258, 31.451203791590125,
		10.08596698683239, 27.0644431779563,
		18.35145401733488, 21.43960601132347,
		17.94217898908129, 7.840914014972789,
		21.926758029071763, 19.607130944881817,
	}
}

func setupSimpleRegressor() *Regressor {
	theta := []float64{0.3574458, -0.01453134, -1.2603342, 0.00891642, -0.79181815}
	intercept := []float64{55.15045304131303}
	return NewRegressor(theta, intercept, 1)
}

func setupMinimalMultivariateRegressor() *Regressor {
	theta := []float64{
		0.3574458, -0.01453134,
		-1.2603342, 0.00891642,
		-0.79181815, 0.3574458,
		-0.01453134, -1.2603342,
		0.00891642, -0.79181815,
	}
	intercept := []float64{55.15045304131303, 55.15045304131303}
	return NewRegressor(theta, intercept, 2)
}

func TestSimpleRegressor_Predict(t *testing.T) {
	x := bostonX()
	h := bostonH()
	model := setupSimpleRegressor()
	for i := 0; i < len(h); i++ {
		o := model.Predict(x[i])
		if math.Abs(o[0]-h[i]) > ErrorThreshold {
			t.Errorf("Failed to return correct univariate response.")
		}
	}
}

func TestMinimalMultivariateLinearRegression_Predict(t *testing.T) {
	x := []float64{1., 296., 15.3, 396.9, 4.98}
	model := setupMinimalMultivariateRegressor()
	o := model.Predict(x)
	if ((o[0] - 31.519181652313026) > ErrorThreshold) || ((o[1] - 31.519181652313026) > ErrorThreshold) {
		t.Errorf("Failed to return correct multivariate response.")
	}
}
