package lingo

import (
	"gonum.org/v1/gonum/mat"
)

// NewLinearClassifier initializes a new Linear LinearClassifier model
func NewLinearClassifier(theta []float64, intercept []float64, nVars int) *LinearClassifier {
	vecIntercept := mat.NewVecDense(nVars, intercept)
	newTheta := mat.NewDense(nVars, len(theta)/nVars, theta)

	return &LinearClassifier{newTheta, vecIntercept}
}

// LinearClassifier model
type LinearClassifier struct {
	Theta     *mat.Dense
	Intercept *mat.VecDense
}

// Predict the probability of classes for a single observation
func (m *LinearClassifier) PredictProba(x []float64) []float64 {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.Theta, m.Intercept)
	return VecToArrayFloat64(Softmax(H).RowView(0))
}

// Predict the class for a single observation
func (m *LinearClassifier) PredictClass(x []float64) []int {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.Theta, m.Intercept)
	c, _ := Argmax(H.RowView(0))
	h := []int{c}
	return h
}

// Predict the probability of classes for a single observation
func (m *LinearClassifier) Predict(x []float64) []float64 {
	vx := mat.NewDense(1, len(x), x)
	H := DecisionFunction(vx, m.Theta, m.Intercept)
	n, _ := m.Theta.Dims()
	c, _ := Argmax(H.RowView(0))
	h := make([]float64, n)
	h[c] = 1.0
	return h
}
