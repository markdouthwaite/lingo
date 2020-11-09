package models

import "gonum.org/v1/gonum/mat"


// Create a new Linear Regression model
func CreateLinearRegression(coeffs []float64, intercept float64, nVars int) *LinearRegression {
	newCoeffs := mat.NewDense(len(coeffs), nVars, coeffs)
	return &LinearRegression{newCoeffs, intercept}
}


// Linear regression model struct
type LinearRegression struct {
	coeffs *mat.Dense
	intercept float64
}


// Predict the response for a single observation
func (m *LinearRegression) Predict(x []float64) *mat.Dense {
	vx := mat.NewDense(len(x),1, x)
	vx.Mul(m.coeffs, vx)
	return vx
}
