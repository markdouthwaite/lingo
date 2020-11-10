package lingo

import (
	"gonum.org/v1/gonum/mat"
)

// Create a new Linear Regression model
func NewClassifier(theta []float64, intercept []float64, nVars int) *Classifier {
	vecIntercept := mat.NewVecDense(nVars, intercept)
	newTheta := mat.NewDense(nVars, len(theta)/nVars, theta)

	return &Classifier{newTheta, vecIntercept}
}

// Linear regression model struct
type Classifier struct {
	theta     *mat.Dense
	intercept *mat.VecDense
}

// Predict the probability of classes for a single observation
func (m *Classifier) PredictProba(x []float64) []float64 {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.theta, m.intercept)
	return VecToArrayFloat64(Softmax(H).RowView(0))
}

// Predict the class for a single observation
func (m *Classifier) PredictClass(x []float64) []int {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.theta, m.intercept)
	c, _ := Argmax(H.RowView(0))
	h := []int{c}
	return h
}

// Predict the probability of classes for a single observation
func (m *Classifier) Predict(x []float64) []float64 {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.theta, m.intercept)
	n, _ := m.theta.Dims()
	c, _ := Argmax(H.RowView(0))
	h := make([]float64, n)
	h[c] = 1.0
	return h
}
