package lingo

import (
	"gonum.org/v1/gonum/mat"
)

// Create a new Linear Regression model
func NewClassifier(coeffs []float64, intercept []float64, nVars int) *Classifier {
	vecIntercept := mat.NewVecDense(nVars, intercept)
	newCoeffs := mat.NewDense(nVars, len(coeffs)/nVars, coeffs)

	return &Classifier{newCoeffs, vecIntercept}
}

// Linear regression model struct
type Classifier struct {
	coeffs    *mat.Dense
	intercept *mat.VecDense
}

// Predict the probability of classes for a single observation
func (m *Classifier) PredictProba(x []float64) []float64 {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.coeffs, m.intercept)
	return VecToArrayFloat64(Softmax(H).RowView(0))
}

// Predict the class for a single observation
func (m *Classifier) PredictClass(x []float64) []int {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.coeffs, m.intercept)
	c, _ := Argmax(H.RowView(0))
	h := []int{c}
	return h
}

// Predict the probability of classes for a single observation
func (m *Classifier) Predict(x []float64) []float64 {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.coeffs, m.intercept)
	n, _ := m.coeffs.Dims()
	c, _ := Argmax(H.RowView(0))
	h := make([]float64, n)
	h[c] = 1.0
	return h
}
