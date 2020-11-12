package lingo

import (
	"gonum.org/v1/gonum/mat"
)

// Create a new Linear Regression model
func NewLinearRegressor(theta []float64, intercept []float64, nVars int) *LinearRegressor {
	vecIntercept := mat.NewVecDense(nVars, intercept)
	newTheta := mat.NewDense(nVars, len(theta)/nVars, theta)
	return &LinearRegressor{newTheta, vecIntercept}
}

// LinearRegressor
type LinearRegressor struct {
	Theta     *mat.Dense
	Intercept *mat.VecDense
}

// Predict runs inference for a single observation
func (m *LinearRegressor) Predict(x []float64) []float64 {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.Theta, m.Intercept)
	return VecToArrayFloat64(H.RowView(0))
}
