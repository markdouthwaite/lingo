package lingo

import (
	"gonum.org/v1/gonum/mat"
)

// Create a new Linear Regression model
func NewRegressor(theta []float64, intercept []float64, nVars int) *Regressor {

	vecIntercept := mat.NewVecDense(nVars, intercept)
	newTheta := mat.NewDense(nVars, len(theta)/nVars, theta)
	return &Regressor{newTheta, vecIntercept}
}

// Linear regression model struct
type Regressor struct {
	theta     *mat.Dense
	intercept *mat.VecDense
}

// Predict the response for a single observation
func (m *Regressor) Predict(x []float64) []float64 {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.theta, m.intercept)
	return VecToArrayFloat64(H.RowView(0))
}
