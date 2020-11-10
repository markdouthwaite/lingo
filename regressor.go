package lingo

import (
	"gonum.org/v1/gonum/mat"
)

// Create a new Linear Regression model
func CreateRegression(coeffs []float64, intercept []float64, nVars int) *Regressor {
	vecIntercept := mat.NewVecDense(nVars, intercept)
	newCoeffs := mat.NewDense(nVars, len(coeffs)/nVars, coeffs)
	return &Regressor{newCoeffs, vecIntercept}
}

// Linear regression model struct
type Regressor struct {
	coeffs    *mat.Dense
	intercept *mat.VecDense
}

// Predict the response for a single observation
func (m *Regressor) Predict(x []float64) []float64 {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.coeffs, m.intercept)
	return VecToArrayFloat64(H.RowView(0))
}
